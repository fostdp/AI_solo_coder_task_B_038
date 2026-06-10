"""
冻干机群控调度微服务
基于电价和产能需求，用整数规划优化10台冻干机的启停和冻干曲线选择

优化目标：
1. 最小化电费成本（考虑峰谷电价）
2. 最大化产能
3. 平衡设备使用率

约束条件：
- 最大并发设备数
- 设备维护周期
- 人员可用时间
- 批次持续时间
"""

import asyncio
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from uuid import uuid4
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    MicroserviceBase,
    CHANNELS, SERVICE_IDS, MESSAGE_TYPES,
    FleetSchedule, FleetCommand, FleetStatus, BatchRecord,
    MessageFactory, validate_message, extract_payload,
    config_loader, FleetConfig
)


@dataclass
class DeviceState:
    """设备状态"""
    device_id: int
    status: str = "idle"  # idle, running, paused, maintenance, defrosting
    current_batch_id: Optional[str] = None
    current_formula_id: Optional[str] = None
    current_profile_id: Optional[int] = None
    current_phase: Optional[str] = None
    phase_start_time: Optional[float] = None
    estimated_completion_time: Optional[float] = None
    batches_completed: int = 0
    total_run_hours: float = 0.0
    last_maintenance_time: Optional[float] = None
    priority: int = 1


@dataclass
class ScheduledBatch:
    """调度的批次"""
    device_id: int
    batch_id: str
    formula_id: str
    profile_id: int
    start_time: float
    end_time: float
    energy_kwh: float
    priority: int = 0


@dataclass
class TimeSlot:
    """时间段"""
    start_hour: int
    end_hour: int
    price: float
    is_valley: bool


class IntegerProgrammingSolver:
    """整数规划求解器（简化版，使用贪心+启发式算法）"""
    
    def __init__(self, config: FleetConfig):
        self.config = config
        self.opt_config = config.optimization
        self.price_config = config.electricity_price
        self.constraints = config.constraints
        
        # 可用冻干曲线
        self.profiles = config.__dict__.get('freeze_profiles', [])
        
        # 初始化PuLP（如果可用）
        self._has_pulp = False
        try:
            import pulp
            self._pulp = pulp
            self._has_pulp = True
        except ImportError:
            self._has_pulp = False
            print("[FleetController] 警告: PuLP未安装，使用启发式算法")
    
    def solve(self, device_states: Dict[int, DeviceState], 
              required_batches: int,
              time_horizon_hours: int = 24,
              start_time: Optional[datetime] = None) -> Tuple[List[ScheduledBatch], float, float, str]:
        """
        求解调度问题
        
        返回：(调度计划, 预计电费, 预计节能, 求解状态)"""
        if start_time is None:
            start_time = datetime.now(timezone.utc)
        
        # 获取电价
        price_schedule = self._get_electricity_prices(start_time, time_horizon_hours)
        
        if self._has_pulp:
            try:
                return self._solve_with_pulp(device_states, required_batches, 
                                             time_horizon_hours, start_time, price_schedule)
            except Exception as e:
                self.logger.error(f"PuLP solver failed, falling back to heuristic: {e}")
                return self._solve_heuristic(device_states, required_batches,
                                             time_horizon_hours, start_time, price_schedule)
        else:
            return self._solve_heuristic(device_states, required_batches,
                                         time_horizon_hours, start_time, price_schedule)
    
    def _get_electricity_prices(self, start_time: datetime, hours: int) -> List[TimeSlot]:
        """获取电价表"""
        slots = []
        static_prices = self.price_config.get('static_prices', {})
        
        for hour_offset in range(hours):
            dt = start_time + timedelta(hours=hour_offset)
            hour = dt.hour
            
            if hour in static_prices.get('peak_hours', []):
                price = static_prices.get('peak', 1.2)
                is_valley = False
            elif hour in static_prices.get('flat_hours', []):
                price = static_prices.get('flat', 0.8)
                is_valley = False
            else:
                price = static_prices.get('valley', 0.4)
                is_valley = True
            
            slots.append(TimeSlot(hour_offset, hour_offset + 1, price, is_valley))
        
        return slots
    
    def _solve_with_pulp(self, device_states: Dict[int, DeviceState],
                         required_batches: int,
                         time_horizon_hours: int,
                         start_time: datetime,
                         price_schedule: List[TimeSlot]) -> Tuple[List[ScheduledBatch], float, float, str]:
        """使用PuLP求解整数规划"""
        try:
            pulp = self._pulp
            
            # 获取可用设备
            available_devices = [
                device_id for device_id, state in device_states.items()
                if state.status == "idle"
            ]
            
            if not available_devices:
                return [], 0.0, 0.0, "no_available"
            
            # 获取冻干曲线
            profiles = self.profiles
            if not profiles:
                profiles = [
                    {'formula_id': 'FORMULA-001', 'primary_drying_hours': 24, 'secondary_drying_hours': 8, 'energy_kwh': 120},
                ]
            
            # 时间粒度
            time_resolution = self.opt_config.get('time_resolution_minutes', 30)
            num_slots = time_horizon_hours * 60 // time_resolution
            
            # 创建问题
            prob = pulp.LpProblem("FleetScheduling", pulp.LpMinimize)
            
            # 决策变量：x[device, profile, time_slot] = 是否在该时间开始该批次
            x = {}
            for d in available_devices:
                for p_idx, profile in enumerate(profiles):
                    total_hours = profile['primary_drying_hours'] + profile['secondary_drying_hours']
                    duration_slots = int(total_hours * 60 // time_resolution)
                    for t in range(num_slots - duration_slots + 1):
                        x[(d, p_idx, t)] = pulp.LpVariable(
                            f"x_{d}_{p_idx}_{t}", cat='Binary')
            
            # 目标函数：最小化电费
            objective = []
            for (d, p_idx, t), var in x.items():
                profile = profiles[p_idx]
                duration_slots = int((profile['primary_drying_hours'] + profile['secondary_drying_hours']) * 60 // time_resolution)
                # 计算该批次在各个时段的电费
                cost = 0
                for i in range(duration_slots):
                    slot_idx = t + i
                    if slot_idx < len(price_schedule):
                        hour_idx = slot_idx * time_resolution // 60
                        if hour_idx < len(price_schedule):
                            cost += price_schedule[hour_idx].price * profile['energy_kwh'] * (time_resolution / 60) / (profile['primary_drying_hours'] + profile['secondary_drying_hours'])
                objective.append(cost * var)
            
            prob += pulp.lpSum(objective)
            
            # 约束1：满足产能需求
            prob += pulp.lpSum(x.values()) >= required_batches
            
            # 约束2：每台设备同一时间只能运行一个批次
            for d in available_devices:
                for t in range(num_slots):
                    vars_at_t = []
                    for p_idx, profile in enumerate(profiles):
                        duration_slots = int((profile['primary_drying_hours'] + profile['secondary_drying_hours']) * 60 // time_resolution)
                        for start_t in range(max(0, t - duration_slots + 1), t + 1):
                            if (d, p_idx, start_t) in x:
                                vars_at_t.append(x[(d, p_idx, start_t)])
                    if vars_at_t:
                        prob += pulp.lpSum(vars_at_t) <= 1
            
            # 约束3：最大并发设备数
            max_concurrent = self.constraints.get('max_concurrent_devices', 10)
            for t in range(num_slots):
                vars_at_t = []
                for d in available_devices:
                    for p_idx, profile in enumerate(profiles):
                        duration_slots = int((profile['primary_drying_hours'] + profile['secondary_drying_hours']) * 60 // time_resolution)
                        for start_t in range(max(0, t - duration_slots + 1), t + 1):
                            if (d, p_idx, start_t) in x:
                                vars_at_t.append(x[(d, p_idx, start_t)])
                if vars_at_t:
                    prob += pulp.lpSum(vars_at_t) <= max_concurrent
            
            # 求解
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                status = 'optimal'
            elif pulp.LpStatus[prob.status] == 'Not Solved':
                return self._solve_heuristic(device_states, required_batches,
                                             time_horizon_hours, start_time, price_schedule)
            else:
                status = 'suboptimal'
            
            # 提取解
            schedule = []
            total_cost = 0.0
            for (d, p_idx, t), var in x.items():
                if var.value() and var.value() > 0.5:
                    profile = profiles[p_idx]
                    start_ts = start_time.timestamp() + t * time_resolution * 60
                    duration_hours = profile['primary_drying_hours'] + profile['secondary_drying_hours']
                    end_ts = start_ts + duration_hours * 3600
                    
                    # 计算该批次的电费
                    cost = 0
                    duration_slots = int(duration_hours * 60 // time_resolution)
                    for i in range(duration_slots):
                        slot_idx = t + i
                        hour_idx = slot_idx * time_resolution // 60
                        if hour_idx < len(price_schedule):
                            cost += price_schedule[hour_idx].price * profile['energy_kwh'] * (time_resolution / 60) / duration_hours
                    
                    batch_id = f"BATCH-{start_time.strftime('%Y%m%d')}-{len(schedule) + 1:03d}"
                    
                    schedule.append(ScheduledBatch(
                        device_id=d,
                        batch_id=batch_id,
                        formula_id=profile['formula_id'],
                        profile_id=p_idx + 1,
                        start_time=start_ts,
                        end_time=end_ts,
                        energy_kwh=profile['energy_kwh'],
                        priority=profile.get('priority', 1)
                    ))
                    total_cost += cost
            
            # 计算基准成本（不优化，按顺序调度）
            baseline_cost = self._calculate_baseline_cost(
                device_states, required_batches, time_horizon_hours, start_time, price_schedule, profiles
            )
            energy_saving = baseline_cost - total_cost
            
            return schedule, total_cost, energy_saving, status
            
        except Exception as e:
            print(f"[FleetController] PuLP求解失败: {e}, 使用启发式算法")
            return self._solve_heuristic(device_states, required_batches,
                                         time_horizon_hours, start_time, price_schedule)
    
    def _calculate_baseline_cost(self, device_states: Dict[int, DeviceState],
                                required_batches: int,
                                time_horizon_hours: int,
                                start_time: datetime,
                                price_schedule: List[TimeSlot],
                                profiles: List[Dict]) -> float:
        """计算基准成本（不优化的顺序调度）"""
        available_devices = sorted([
            (d, s) for d, s in device_states.items()
            if s.status == "idle"
        ], key=lambda x: x[0])
        
        total_cost = 0.0
        batches_scheduled = 0
        device_next_available = {d: 0.0 for d, _ in available_devices}
        
        for batch_idx in range(required_batches):
            if not available_devices:
                break
            
            # 按顺序选择设备
            device_idx = batch_idx % len(available_devices)
            device_id = available_devices[device_idx][0]
            profile = profiles[0]  # 默认第一个配方
            
            start_hour = int(device_next_available[device_id] // 3600)
            if start_hour >= time_horizon_hours:
                break
            
            duration_hours = profile['primary_drying_hours'] + profile['secondary_drying_hours']
            
            cost = 0
            for h in range(start_hour, min(start_hour + int(duration_hours), time_horizon_hours)):
                if h < len(price_schedule):
                    cost += price_schedule[h].price * profile['energy_kwh'] / duration_hours
            
            total_cost += cost
            device_next_available[device_id] = (start_hour + duration_hours) * 3600
            batches_scheduled += 1
        
        return total_cost
    
    def _solve_heuristic(self, device_states: Dict[int, DeviceState],
                           required_batches: int,
                           time_horizon_hours: int,
                           start_time: datetime,
                           price_schedule: List[TimeSlot]) -> Tuple[List[ScheduledBatch], float, float, str]:
        """启发式算法：优先在谷电时段调度高能耗批次"""
        available_devices = sorted([
            (d, s) for d, s in device_states.items()
            if s.status == "idle"
        ], key=lambda x: x[1].priority, reverse=True)
        
        if not available_devices:
            return [], 0.0, 0.0, "no_available"
        
        # 获取所有配方
        profiles = self.profiles
        if not profiles:
            profiles = [
                {'formula_id': 'FORMULA-001', 'primary_drying_hours': 24, 'secondary_drying_hours': 8, 'energy_kwh': 120, 'priority': 1},
            ]
        
        # 按能耗排序配方（高能耗优先谷电）
        sorted_profiles = sorted(profiles, key=lambda p: p['energy_kwh'], reverse=True)
        
        schedule = []
        total_cost = 0.0
        device_next_available = {d: 0.0 for d, _ in available_devices}
        
        # 找出谷电时段
        valley_hours = [i for i, slot in enumerate(price_schedule) if slot.is_valley]
        
        batches_scheduled = 0
        start_ts = start_time.timestamp()
        
        while batches_scheduled < required_batches:
            scheduled_this_round = False
            
            for profile in sorted_profiles:
                if batches_scheduled >= required_batches:
                    break
                
                # 找最合适的设备和时间
                best_device = None
                best_start_hour = None
                best_cost = float('inf')
                
                for device_id, state in available_devices:
                    next_avail = device_next_available[device_id]
                    next_avail_hour = int(next_avail) // 3600
                    
                    duration_hours = profile['primary_drying_hours'] + profile['secondary_drying_hours']
                    
                    # 优先尝试谷电时段开始
                    candidate_hours = []
                    for valley_h in valley_hours:
                        if valley_h >= next_avail_hour and valley_h + duration_hours <= time_horizon_hours:
                            candidate_hours.append(valley_h)
                    
                    # 如果没有合适的谷电时段，尝试任意可用时间
                    if not candidate_hours:
                        for h in range(next_avail_hour, time_horizon_hours - int(duration_hours) + 1):
                            candidate_hours.append(h)
                    
                    for start_hour in candidate_hours:
                        # 计算该时段的电费
                        cost = 0
                        for h in range(start_hour, min(start_hour + int(duration_hours), time_horizon_hours)):
                            if h < len(price_schedule):
                                cost += price_schedule[h].price * profile['energy_kwh'] / duration_hours
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_device = device_id
                            best_start_hour = start_hour
                
                if best_device is not None and best_start_hour is not None:
                    batch_id = f"BATCH-{start_time.strftime('%Y%m%d')}-{batches_scheduled + 1:03d}"
                    batch_start = start_ts + best_start_hour * 3600
                    batch_end = batch_start + duration_hours * 3600
                    
                    schedule.append(ScheduledBatch(
                        device_id=best_device,
                        batch_id=batch_id,
                        formula_id=profile['formula_id'],
                        profile_id=sorted_profiles.index(profile) + 1,
                        start_time=batch_start,
                        end_time=batch_end,
                        energy_kwh=profile['energy_kwh'],
                        priority=profile.get('priority', 1)
                    ))
                    
                    total_cost += best_cost
                    device_next_available[best_device] = (best_start_hour + duration_hours) * 3600
                    batches_scheduled += 1
                    scheduled_this_round = True
            
            if not scheduled_this_round:
                break
        
        # 计算基准成本
        baseline_cost = self._calculate_baseline_cost(
            device_states, required_batches, time_horizon_hours, start_time, price_schedule, profiles
        )
        energy_saving = max(0.0, baseline_cost - total_cost)
        
        return schedule, total_cost, energy_saving, "heuristic"


class FleetControllerService(MicroserviceBase):
    """群控调度服务"""
    
    def __init__(self):
        super().__init__(SERVICE_IDS['FLEET_CONTROLLER'], 'fleet_controller')
        self.config: FleetConfig = config_loader.load_fleet_config()
        self.device_states: Dict[int, DeviceState] = {}
        self.solver = IntegerProgrammingSolver(self.config)
        self.current_schedule: List[ScheduledBatch] = []
        self._init_states()
    
    def _init_states(self):
        """初始化设备状态"""
        priorities = self.config.device_priorities
        default_priority = priorities.get('default', 1)
        
        for device_id in range(1, 11):
            priority = priorities.get(str(device_id), default_priority)
            self.device_states[device_id] = DeviceState(
                device_id=device_id,
                priority=priority
            )
    
    async def _subscribe_channels(self):
        """订阅频道"""
        await self.subscribe(CHANNELS['FLEET_STATUS'], self._handle_fleet_status)
        await self.subscribe(CHANNELS['FLEET_COMMAND'], self._handle_fleet_command)
        await self.subscribe(CHANNELS['ENDPOINT_DETECTION'], self._handle_endpoint)
        await self.subscribe(CHANNELS['CONFIG_UPDATE'], self._handle_config_update)
    
    async def _on_start(self):
        """启动时执行"""
        print(f"[{self.service_id}] 冻干机群控调度服务启动")
        print(f"  - 调度间隔: {self.config.schedule_interval_minutes}分钟")
        print(f"  - 求解器: {'PuLP' if self.solver._has_pulp else 'Heuristic'}")
        print(f"  - 优化目标: {self.opt_config.get('objective', 'energy_cost')}")
        
        # 启动调度循环
        asyncio.create_task(self._scheduling_loop())
        asyncio.create_task(self._status_monitoring_loop())
    
    async def _scheduling_loop(self):
        """调度循环"""
        while self._running:
            try:
                # 计算需要调度的批次（示例：每台设备每天2批次）
                required_batches = 20  # 10台设备 × 2批次
                time_horizon = self.opt_config.get('time_horizon_hours', 24)
                
                await self._run_optimization(required_batches, time_horizon)
                
                await asyncio.sleep(self.config.schedule_interval_minutes * 60)
                
            except Exception as e:
                print(f"[{self.service_id}] 调度循环异常: {e}")
                await asyncio.sleep(60)
    
    async def _status_monitoring_loop(self):
        """状态监控循环"""
        while self._running:
            try:
                # 检查是否有批次应该开始
                now = time.time()
                
                for batch in list(self.current_schedule):
                    if batch.start_time <= now and not self.device_states[batch.device_id].current_batch_id:
                        await self._start_batch(batch)
                
                # 检查是否有批次应该结束
                for device_id, state in self.device_states.items():
                    if state.estimated_completion_time and now >= state.estimated_completion_time:
                        if state.current_batch_id:
                            print(f"[{self.service_id}] 设备{device_id} 批次{state.current_batch_id} 预计完成")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"[{self.service_id}] 状态监控异常: {e}")
                await asyncio.sleep(5)
    
    async def _run_optimization(self, required_batches: int, time_horizon_hours: int):
        """运行优化"""
        if not self.config.enabled:
            return
        
        now = datetime.now(timezone.utc)
        
        # 只考虑空闲设备
        schedule, total_cost, energy_saving, status = self.solver.solve(
            self.device_states, required_batches, time_horizon_hours, now
        )
        
        if not schedule:
            return
        
        self.current_schedule = schedule
        
        # 发布调度计划
        details = []
        for batch in schedule:
            details.append({
                'device_id': batch.device_id,
                'batch_id': batch.batch_id,
                'formula_id': batch.formula_id,
                'freeze_profile_id': batch.profile_id,
                'start_time': datetime.fromtimestamp(batch.start_time, tz=timezone.utc).isoformat(),
                'end_time': datetime.fromtimestamp(batch.end_time, tz=timezone.utc).isoformat(),
                'estimated_cycle_hours': (batch.end_time - batch.start_time) / 3600,
                'estimated_energy_kwh': batch.energy_kwh,
                'priority': batch.priority,
            })
        
        fleet_schedule = FleetSchedule(
            schedule_id=str(uuid4()),
            schedule_date=now.date().isoformat(),
            total_required_batches=required_batches,
            estimated_energy_cost=round(total_cost, 2),
            optimized_energy_saving=round(energy_saving, 2),
            solver_status=status,
            details=details,
            timestamp=now.isoformat()
        )
        
        message = MessageFactory.create_fleet_schedule(fleet_schedule, self.service_id)
        await self.publish(CHANNELS['FLEET_SCHEDULE'], message)
        self._increment_metric("messages_published")
        
        print(f"[{self.service_id}] 调度完成: {len(schedule)}个批次, "
              f"预计电费: {total_cost:.2f}元, 预计节能: {energy_saving:.2f}元, "
              f"求解状态: {status}")
    
    async def _start_batch(self, batch: ScheduledBatch):
        """启动批次"""
        state = self.device_states[batch.device_id]
        
        if state.status != "idle":
            return
        
        # 发布启动命令
        cmd = FleetCommand(
            device_id=batch.device_id,
            command='start_batch',
            timestamp=datetime.now(timezone.utc).isoformat(),
            batch_id=batch.batch_id,
            formula_id=batch.formula_id,
            freeze_profile_id=batch.profile_id,
            priority=batch.priority
        )
        
        message = MessageFactory.create_fleet_command(cmd, self.service_id)
        await self.publish(CHANNELS['FLEET_COMMAND'], message)
        self._increment_metric("messages_published")
        
        # 更新状态
        state.status = "running"
        state.current_batch_id = batch.batch_id
        state.current_formula_id = batch.formula_id
        state.current_profile_id = batch.profile_id
        state.current_phase = "freezing"
        state.phase_start_time = time.time()
        state.estimated_completion_time = batch.end_time
        
        # 发布批次记录
        record = BatchRecord(
            device_id=batch.device_id,
            batch_id=batch.batch_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            update_type='start',
            freeze_profile_id=batch.profile_id,
            formula_id=batch.formula_id,
            start_time=datetime.fromtimestamp(batch.start_time, tz=timezone.utc).isoformat(),
            batch_status='running'
        )
        
        msg = MessageFactory.create_batch_record(record, self.service_id)
        await self.publish(CHANNELS['DB_WRITE'], msg)
        self._increment_metric("messages_published")
        
        print(f"[{self.service_id}] 启动批次: 设备{batch.device_id}, "
              f"批次{batch.batch_id}, 配方{batch.formula_id}")
    
    async def _handle_fleet_status(self, message: Dict):
        """处理设备状态更新"""
        try:
            if not validate_message(message, MESSAGE_TYPES['FLEET_STATUS']):
                return
            
            payload = extract_payload(message)
            status = FleetStatus(**payload)
            
            state = self.device_states.get(status.device_id)
            if not state:
                return
            
            self._increment_metric("messages_received")
            
            # 更新状态
            if status.batch_status == "completed":
                state.status = "idle"
                state.batches_completed += 1
                state.total_run_hours += (time.time() - state.phase_start_time) / 3600 if state.phase_start_time else 0
                
                # 发布批次完成记录
                if state.current_batch_id:
                    record = BatchRecord(
                        device_id=status.device_id,
                        batch_id=state.current_batch_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        update_type='complete',
                        end_time=datetime.now(timezone.utc).isoformat(),
                        total_cycle_hours=(time.time() - state.phase_start_time) / 3600 if state.phase_start_time else 0,
                        batch_status='completed'
                    )
                    
                    msg = MessageFactory.create_batch_record(record, self.service_id)
                    await self.publish(CHANNELS['DB_WRITE'], msg)
                    self._increment_metric("messages_published")
                
                state.current_batch_id = None
                state.current_formula_id = None
                state.current_profile_id = None
                state.current_phase = None
                state.phase_start_time = None
                state.estimated_completion_time = None
                
                print(f"[{self.service_id}] 设备{status.device_id} 批次完成, "
                      f"累计完成: {state.batches_completed}")
                
            else:
                state.status = status.batch_status
                state.current_phase = status.current_phase
                if status.estimated_completion_time:
                    try:
                        state.estimated_completion_time = datetime.fromisoformat(
                            status.estimated_completion_time.replace('Z', '+00:00')
                        ).timestamp()
                    except:
                        pass
                
        except Exception as e:
            print(f"[{self.service_id}] 处理状态更新失败: {e}")
            self._increment_metric("errors")
    
    async def _handle_fleet_command(self, message: Dict):
        """处理手动命令"""
        try:
            if not validate_message(message, MESSAGE_TYPES['FLEET_COMMAND']):
                return
            
            payload = extract_payload(message)
            cmd = FleetCommand(**payload)
            
            state = self.device_states.get(cmd.device_id)
            if not state:
                return
            
            if cmd.command == 'stop_batch' and state.current_batch_id:
                state.status = "idle"
                state.current_batch_id = None
                state.current_phase = None
                print(f"[{self.service_id}] 设备{cmd.device_id} 批次已停止")
                
        except Exception as e:
            print(f"[{self.service_id}] 处理命令失败: {e}")
    
    async def _handle_endpoint(self, message: Dict):
        """处理终点检测"""
        try:
            if not validate_message(message, MESSAGE_TYPES['ENDPOINT']):
                return
            
            payload = extract_payload(message)
            device_id = payload.get('device_id')
            phase = payload.get('cycle_phase')
            
            state = self.device_states.get(device_id)
            if not state:
                return
            
            if phase == 'primary_drying':
                state.current_phase = "secondary_drying"
                state.phase_start_time = time.time()
            elif phase == 'secondary_drying':
                state.current_phase = "completed"
                
        except Exception as e:
            print(f"[{self.service_id}] 处理终点检测失败: {e}")
    
    async def _handle_config_update(self, message: Dict):
        """处理配置更新"""
        try:
            payload = extract_payload(message)
            if payload.get('config_type') == 'fleet':
                self.config = config_loader.load_fleet_config()
                self.solver = IntegerProgrammingSolver(self.config)
                print(f"[{self.service_id}] 配置已更新")
        except Exception as e:
            print(f"[{self.service_id}] 配置更新失败: {e}")


if __name__ == "__main__":
    service = FleetControllerService()
    
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        asyncio.run(service.stop())
    except Exception as e:
        print(f"服务异常退出: {e}")
