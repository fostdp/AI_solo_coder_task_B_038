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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    MicroserviceBase,
    CHANNELS, SERVICE_IDS, MESSAGE_TYPES,
    FleetSchedule, FleetCommand, FleetStatus, BatchRecord,
    MessageFactory, validate_message, extract_payload,
    config_loader, FleetConfig
)

from modules.cluster_scheduler import IntegerProgrammingSolver, SchedulingWorker, SolverConfig


@dataclass
class DeviceState:
    device_id: int
    status: str = "idle"
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
class UrgentBatch:
    batch_id: str
    formula_id: str
    priority: int
    deadline_hours: float
    requested_start_time: Optional[float] = None
    energy_kwh: float = 0.0
    min_start_delay: float = 0.0


@dataclass
class ScheduledBatch:
    device_id: int
    batch_id: str
    formula_id: str
    profile_id: int
    start_time: float
    end_time: float
    energy_kwh: float
    priority: int = 0
    is_urgent: bool = False
    original_schedule: Optional['ScheduledBatch'] = None
    rescheduled: bool = False


@dataclass
class TimeSlot:
    start_hour: int
    end_hour: int
    price: float
    is_valley: bool


class FleetControllerService(MicroserviceBase):
    
    def __init__(self):
        super().__init__(SERVICE_IDS['FLEET_CONTROLLER'], 'fleet_controller')
        self.config: FleetConfig = config_loader.load_fleet_config()
        self.device_states: Dict[int, DeviceState] = {}
        self.opt_config = self.config.optimization
        
        solver_config = SolverConfig(
            optimization=self.config.optimization,
            electricity_price=self.config.electricity_price,
            constraints=self.config.constraints,
            freeze_profiles=self.config.__dict__.get('freeze_profiles', [])
        )
        self.solver = IntegerProgrammingSolver(solver_config)
        self.worker = SchedulingWorker(solver_config)
        self.current_schedule: List[ScheduledBatch] = []
        self._pending_task_id: Optional[str] = None
        self._init_states()
    
    def _init_states(self):
        priorities = self.config.device_priorities
        default_priority = priorities.get('default', 1)
        
        for device_id in range(1, 11):
            priority = priorities.get(str(device_id), default_priority)
            self.device_states[device_id] = DeviceState(
                device_id=device_id,
                priority=priority
            )
    
    async def _subscribe_channels(self):
        await self.subscribe(CHANNELS['FLEET_STATUS'], self._handle_fleet_status)
        await self.subscribe(CHANNELS['FLEET_COMMAND'], self._handle_fleet_command)
        await self.subscribe(CHANNELS['ENDPOINT_DETECTION'], self._handle_endpoint)
        await self.subscribe(CHANNELS['CONFIG_UPDATE'], self._handle_config_update)
    
    async def _on_start(self):
        print(f"[{self.service_id}] 冻干机群控调度服务启动")
        print(f"  - 调度间隔: {self.config.schedule_interval_minutes}分钟")
        print(f"  - 求解器: {'PuLP' if self.solver._has_pulp else 'Heuristic'}")
        print(f"  - Worker进程: 启动中")
        
        self.worker.start()
        print(f"  - Worker进程: 已启动 (PID: {self.worker._process.pid if self.worker._process else 'N/A'})")
        
        asyncio.create_task(self._scheduling_loop())
        asyncio.create_task(self._status_monitoring_loop())
        asyncio.create_task(self._worker_result_polling_loop())
    
    async def _on_stop(self):
        if self.worker.is_alive():
            self.worker.stop()
            print(f"[{self.service_id}] Worker进程已停止")
    
    async def _scheduling_loop(self):
        while self._running:
            try:
                required_batches = 20
                time_horizon = self.opt_config.get('time_horizon_hours', 24)
                
                await self._run_optimization(required_batches, time_horizon)
                
                await asyncio.sleep(self.config.schedule_interval_minutes * 60)
                
            except Exception as e:
                print(f"[{self.service_id}] 调度循环异常: {e}")
                await asyncio.sleep(60)
    
    async def _status_monitoring_loop(self):
        while self._running:
            try:
                now = time.time()
                
                for batch in list(self.current_schedule):
                    if batch.start_time <= now and not self.device_states[batch.device_id].current_batch_id:
                        await self._start_batch(batch)
                
                for device_id, state in self.device_states.items():
                    if state.estimated_completion_time and now >= state.estimated_completion_time:
                        if state.current_batch_id:
                            print(f"[{self.service_id}] 设备{device_id} 批次{state.current_batch_id} 预计完成")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                print(f"[{self.service_id}] 状态监控异常: {e}")
                await asyncio.sleep(5)
    
    async def _worker_result_polling_loop(self):
        while self._running:
            try:
                results = self.worker.get_all_results(timeout=0.0)
                for result in results:
                    if result['success']:
                        task_type = self.worker._active_tasks.get(result['task_id'], {}).get('task_type', 'unknown')
                        if task_type == 'solve':
                            await self._handle_solve_result(result['result'])
                        elif task_type == 'reschedule':
                            await self._handle_reschedule_result(result['result'])
                    else:
                        print(f"[{self.service_id}] Worker任务失败: {result.get('error', 'unknown')}")
                
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"[{self.service_id}] Worker结果轮询异常: {e}")
                await asyncio.sleep(5)
    
    async def _handle_solve_result(self, result: Dict):
        schedule = result.get('schedule', [])
        total_cost = result.get('total_cost', 0.0)
        energy_saving = result.get('energy_saving', 0.0)
        status = result.get('status', 'unknown')
        
        if not schedule:
            return
        
        self.current_schedule = schedule
        await self._publish_schedule(schedule, total_cost, energy_saving, status)
    
    async def _handle_reschedule_result(self, result: Dict):
        new_schedule = result.get('new_schedule', [])
        urgent_cost = result.get('urgent_cost', 0.0)
        cost_delta = result.get('cost_delta', 0.0)
        status = result.get('status', 'unknown')
        
        if not new_schedule:
            return
        
        self.current_schedule = new_schedule
        time_horizon = self.opt_config.get('time_horizon_hours', 24)
        await self._publish_schedule_update(new_schedule, time_horizon, urgent_cost, cost_delta)
    
    async def _run_optimization(self, required_batches: int, time_horizon_hours: int):
        if not self.config.enabled:
            return
        
        now = datetime.now(timezone.utc)
        
        schedule, total_cost, energy_saving, status = self.solver.solve(
            self.device_states, required_batches, time_horizon_hours, now
        )
        
        if not schedule:
            return
        
        self.current_schedule = schedule
        
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
    
    async def _publish_schedule(self, schedule, total_cost, energy_saving, status):
        now = datetime.now(timezone.utc)
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
            total_required_batches=len(schedule),
            estimated_energy_cost=round(total_cost, 2),
            optimized_energy_saving=round(energy_saving, 2),
            solver_status=status,
            details=details,
            timestamp=now.isoformat()
        )
        
        message = MessageFactory.create_fleet_schedule(fleet_schedule, self.service_id)
        await self.publish(CHANNELS['FLEET_SCHEDULE'], message)
        self._increment_metric("messages_published")
    
    async def _start_batch(self, batch: ScheduledBatch):
        state = self.device_states[batch.device_id]
        
        if state.status != "idle":
            return
        
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
        
        state.status = "running"
        state.current_batch_id = batch.batch_id
        state.current_formula_id = batch.formula_id
        state.current_profile_id = batch.profile_id
        state.current_phase = "freezing"
        state.phase_start_time = time.time()
        state.estimated_completion_time = batch.end_time
        
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
        try:
            if not validate_message(message, MESSAGE_TYPES['FLEET_STATUS']):
                return
            
            payload = extract_payload(message)
            status = FleetStatus(**payload)
            
            state = self.device_states.get(status.device_id)
            if not state:
                return
            
            self._increment_metric("messages_received")
            
            if status.batch_status == "completed":
                state.status = "idle"
                state.batches_completed += 1
                state.total_run_hours += (time.time() - state.phase_start_time) / 3600 if state.phase_start_time else 0
                
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
        try:
            if not validate_message(message, MESSAGE_TYPES['FLEET_COMMAND']):
                return
            
            payload = extract_payload(message)
            cmd = FleetCommand(**payload)
            
            if cmd.command == 'urgent_insert':
                await self._handle_urgent_insert(cmd, payload)
                return
            
            state = self.device_states.get(cmd.device_id)
            if not state:
                return
            
            if cmd.command == 'stop_batch' and state.current_batch_id:
                state.status = "idle"
                state.current_batch_id = None
                state.current_phase = None
                print(f"[{self.service_id}] 设备{cmd.device_id} 批次已停止")
            
            elif cmd.command == 'force_reschedule':
                await self._run_optimization(
                    self.opt_config.get('default_batches_per_cycle', 20),
                    self.opt_config.get('time_horizon_hours', 24)
                )
                print(f"[{self.service_id}] 收到强制重新调度命令")
                
        except Exception as e:
            print(f"[{self.service_id}] 处理命令失败: {e}")
    
    async def _handle_urgent_insert(self, cmd: FleetCommand, payload: Dict):
        try:
            batch_id = payload.get('batch_id', f"URGENT-{int(time.time())}")
            formula_id = cmd.formula_id or 'FORMULA-001'
            priority = payload.get('priority', 10)
            deadline_hours = payload.get('deadline_hours', 8)
            
            urgent_batch = UrgentBatch(
                batch_id=batch_id,
                formula_id=formula_id,
                priority=priority,
                deadline_hours=deadline_hours,
                requested_start_time=time.time()
            )
            
            print(f"[{self.service_id}] 收到紧急插单: {batch_id}, 配方: {formula_id}, 优先级: {priority}, 时限: {deadline_hours}h")
            
            current_time = datetime.now(timezone.utc)
            time_horizon = self.opt_config.get('time_horizon_hours', 24)
            
            new_schedule, urgent_cost, cost_delta, status = self.solver.reschedule_for_urgent_batch(
                self.current_schedule,
                self.device_states,
                urgent_batch,
                time_horizon,
                current_time
            )
            
            if status == "no_available":
                print(f"[{self.service_id}] 紧急插单失败: 无可用设备，所有设备优先级均 >= {priority}")
                await self._publish_urgent_result(batch_id, False, "no_available_devices")
                return
            elif status == "no_slot":
                print(f"[{self.service_id}] 紧急插单失败: 在时间窗口内无法找到可用时段")
                await self._publish_urgent_result(batch_id, False, "no_time_slot")
                return
            
            is_valid, violations = self.solver.validate_schedule(
                new_schedule, time_horizon, current_time
            )
            
            if not is_valid:
                print(f"[{self.service_id}] 紧急插单调度验证失败: {violations}")
                await self._publish_urgent_result(batch_id, False, "validation_failed", violations)
                return
            
            self.current_schedule = new_schedule
            
            urgent_batches = [b for b in new_schedule if b.batch_id == batch_id]
            if urgent_batches:
                urgent = urgent_batches[0]
                state = self.device_states.get(urgent.device_id)
                if state and state.status == "idle":
                    await self._start_batch(urgent)
                    print(f"[{self.service_id}] 紧急批次 {batch_id} 已立即启动在设备 {urgent.device_id}")
                else:
                    print(f"[{self.service_id}] 紧急批次 {batch_id} 已安排在设备 {urgent.device_id}, "
                          f"开始时间: {datetime.fromtimestamp(urgent.start_time, tz=timezone.utc).isoformat()}")
            
            await self._publish_schedule_update(new_schedule, time_horizon, urgent_cost, cost_delta)
            
            await self._publish_urgent_result(
                batch_id, True, status,
                {
                    'device_id': urgent_batches[0].device_id if urgent_batches else None,
                    'start_time': datetime.fromtimestamp(urgent_batches[0].start_time, tz=timezone.utc).isoformat() if urgent_batches else None,
                    'end_time': datetime.fromtimestamp(urgent_batches[0].end_time, tz=timezone.utc).isoformat() if urgent_batches else None,
                    'urgent_cost': round(urgent_cost, 2),
                    'cost_delta': round(cost_delta, 2),
                    'rescheduled_batches': sum(1 for b in new_schedule if b.rescheduled)
                }
            )
            
            self._increment_metric("urgent_batches_handled")
            
        except Exception as e:
            print(f"[{self.service_id}] 处理紧急插单失败: {e}")
            self._increment_metric("errors")
    
    async def _publish_urgent_result(self, batch_id: str, success: bool, 
                                      status: str, details: Optional[Dict] = None):
        try:
            result = {
                'batch_id': batch_id,
                'success': success,
                'status': status,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            if details:
                result.update(details)
            
            message = MessageFactory.create(
                MESSAGE_TYPES['URGENT_RESULT'],
                result,
                self.service_id
            )
            await self.publish(CHANNELS.get('URGENT_RESULT', 'fleet.urgent_result'), message)
            self._increment_metric("messages_published")
            
        except Exception as e:
            print(f"[{self.service_id}] 发布紧急插单结果失败: {e}")
    
    async def _publish_schedule_update(self, schedule: List[ScheduledBatch], 
                                        time_horizon: int, urgent_cost: float, cost_delta: float):
        try:
            now = datetime.now(timezone.utc)
            
            details = []
            total_cost = urgent_cost
            for batch in schedule:
                batch_cost = batch.energy_kwh * 0.8
                if not batch.is_urgent:
                    total_cost += batch_cost
                
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
                    'is_urgent': batch.is_urgent,
                    'rescheduled': batch.rescheduled
                })
            
            fleet_schedule = FleetSchedule(
                schedule_id=str(uuid4()),
                schedule_date=now.date().isoformat(),
                total_required_batches=len(schedule),
                estimated_energy_cost=round(total_cost, 2),
                optimized_energy_saving=round(abs(cost_delta), 2),
                solver_status='dynamic_reschedule',
                details=details,
                timestamp=now.isoformat()
            )
            
            message = MessageFactory.create_fleet_schedule(fleet_schedule, self.service_id)
            await self.publish(CHANNELS['FLEET_SCHEDULE'], message)
            self._increment_metric("messages_published")
            
            print(f"[{self.service_id}] 动态重调度完成: {len(schedule)}个批次, "
                  f"紧急批次成本: {urgent_cost:.2f}元, 成本变化: {cost_delta:.2f}元")
            
        except Exception as e:
            print(f"[{self.service_id}] 发布调度更新失败: {e}")
    
    async def _handle_endpoint(self, message: Dict):
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
        try:
            payload = extract_payload(message)
            if payload.get('config_type') == 'fleet':
                self.config = config_loader.load_fleet_config()
                solver_config = SolverConfig(
                    optimization=self.config.optimization,
                    electricity_price=self.config.electricity_price,
                    constraints=self.config.constraints,
                    freeze_profiles=self.config.__dict__.get('freeze_profiles', [])
                )
                self.solver = IntegerProgrammingSolver(solver_config)
                self.worker.stop()
                self.worker = SchedulingWorker(solver_config)
                self.worker.start()
                print(f"[{self.service_id}] 配置已更新，Worker已重启")
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
