"""
冷阱除霜优化微服务
基于冷阱温度趋势估算结霜厚度，优化除霜周期和加热功率
"""

import asyncio
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass, field, asdict
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    MicroserviceBase,
    CHANNELS, SERVICE_IDS, MESSAGE_TYPES,
    TelemetryData, DefrostOptimization, DefrostCommand, DefrostStatus,
    MessageFactory, validate_message, extract_payload,
    config_loader, DefrostConfig
)

from modules.defrost_optimizer import MultiSensorFusion, FrostThicknessEstimator, DefrostOptimizer


@dataclass
class DeviceDefrostState:
    """设备除霜状态"""
    device_id: int
    batch_id: Optional[str] = None
    
    # 多传感器冷阱温度历史（每个传感器独立历史）
    # sensor_id -> deque of (timestamp, temperature)
    multi_sensor_history: Dict[int, Deque[Tuple[float, float]]] = field(
        default_factory=lambda: {
            i: deque(maxlen=720) for i in range(1, 6)  # 支持5个传感器
        }
    )
    
    # 兼容单传感器接口（主传感器历史）
    cold_trap_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=720)  # 2小时（10秒间隔）
    )
    
    # 传感器元数据
    sensor_positions: Dict[int, str] = field(default_factory=lambda: {
        1: "inlet",      # 入口（结霜最严重）
        2: "coil_1",     # 盘管1
        3: "coil_2",     # 盘管2
        4: "coil_3",     # 盘管3
        5: "outlet"      # 出口（结霜最轻）
    })
    
    sensor_weights: Dict[int, float] = field(default_factory=lambda: {
        1: 0.35,  # 入口权重最高
        2: 0.20,
        3: 0.20,
        4: 0.15,
        5: 0.10   # 出口权重最低
    })
    
    sensor_health: Dict[int, float] = field(default_factory=lambda: {
        i: 1.0 for i in range(1, 6)  # 0-1，1表示传感器正常
    })
    
    # 融合后的温度历史
    fused_temperature_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=720)
    )
    
    # 结霜厚度分布估算（每个传感器位置）
    frost_thickness_distribution: Dict[int, float] = field(default_factory=lambda: {
        i: 0.0 for i in range(1, 6)
    })
    
    # 结霜估算
    estimated_frost_thickness: float = 0.0
    last_defrost_time: Optional[float] = None
    cumulative_running_hours: float = 0.0
    
    # 除霜状态
    defrost_in_progress: bool = False
    defrost_phase: Optional[str] = None  # preheat, main_heating, soak, cooldown
    defrost_start_time: Optional[float] = None
    defrost_power_pct: float = 0.0
    target_defrost_temp: float = 10.0  # 除霜目标温度
    current_cold_trap_temp: float = -80.0
    
    # 调度状态
    next_scheduled_defrost: Optional[float] = None
    recommended_interval_hours: float = 12.0


class DefrostOptimizerService(MicroserviceBase):
    """冷阱除霜优化服务"""
    
    def __init__(self):
        super().__init__(SERVICE_IDS['DEFROST_OPTIMIZER'], 'defrost_optimizer')
        self.config: DefrostConfig = config_loader.load_defrost_config()
        self.optimizer = DefrostOptimizer(asdict(self.config))
        self.device_states: Dict[int, DeviceDefrostState] = {}
        self._init_states()
    
    def _init_states(self):
        """初始化设备状态"""
        for device_id in range(1, 11):
            self.device_states[device_id] = DeviceDefrostState(device_id=device_id)
    
    async def _subscribe_channels(self):
        """订阅频道"""
        await self.subscribe(CHANNELS['TELEMETRY_RAW'], self._handle_telemetry)
        await self.subscribe(CHANNELS['ENDPOINT_DETECTION'], self._handle_endpoint)
        await self.subscribe(CHANNELS['DEFROST_COMMAND'], self._handle_defrost_command)
        await self.subscribe(CHANNELS['CONFIG_UPDATE'], self._handle_config_update)
    
    async def _on_start(self):
        """启动时执行"""
        print(f"[{self.service_id}] 冷阱除霜优化服务启动")
        print(f"  - 检查间隔: {self.config.check_interval_seconds}s")
        print(f"  - 最大结霜厚度: {self.config.frost_thickness_estimation.get('max_frost_thickness_mm', 5.0)}mm")
        
        # 启动优化循环
        asyncio.create_task(self._optimization_loop())
        asyncio.create_task(self._defrost_control_loop())
    
    async def _optimization_loop(self):
        """优化循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval_seconds)
                
                for device_id, state in self.device_states.items():
                    if state.defrost_in_progress:
                        continue
                    
                    await self._optimize_defrost(device_id, state)
                    
            except Exception as e:
                print(f"[{self.service_id}] 优化循环异常: {e}")
                await asyncio.sleep(5)
    
    async def _defrost_control_loop(self):
        """除霜控制循环"""
        while self._running:
            try:
                await asyncio.sleep(5)
                
                for device_id, state in self.device_states.items():
                    if state.defrost_in_progress:
                        await self._control_defrost_process(device_id, state)
                    
            except Exception as e:
                print(f"[{self.service_id}] 除霜控制循环异常: {e}")
                await asyncio.sleep(5)
    
    async def _handle_telemetry(self, message: Dict):
        """处理遥测数据"""
        try:
            if not validate_message(message, MESSAGE_TYPES['TELEMETRY']):
                return
            
            payload = extract_payload(message)
            telemetry = TelemetryData(**payload)
            device_id = telemetry.device_id
            
            state = self.device_states.get(device_id)
            if not state:
                return
            
            self._increment_metric("messages_received")
            
            # 更新批次ID
            if telemetry.batch_id:
                state.batch_id = telemetry.batch_id
            
            current_time = time.time()
            
            # 处理多传感器冷阱温度数据
            # 优先使用多传感器数据（如果遥测数据中有）
            if hasattr(telemetry, 'cold_trap_temperatures') and telemetry.cold_trap_temperatures:
                # telemetry包含多传感器温度数据
                for sensor_id, temp in enumerate(telemetry.cold_trap_temperatures, start=1):
                    if sensor_id in state.multi_sensor_history:
                        state.multi_sensor_history[sensor_id].append((current_time, temp))
                # 使用主传感器（第一个）作为兼容
                primary_temp = telemetry.cold_trap_temperatures[0] if telemetry.cold_trap_temperatures else telemetry.cold_trap_temp
            else:
                # 单传感器数据，分配到所有传感器位置（模拟多传感器）
                primary_temp = telemetry.cold_trap_temp
                # 根据位置添加偏移，模拟结霜不均匀性
                position_offsets = {
                    1: 3.0,   # 入口温度最高（结霜最厚）
                    2: 1.0,   # 盘管1
                    3: 0.0,   # 盘管2（基准）
                    4: -0.5,  # 盘管3
                    5: -2.0,  # 出口温度最低（结霜最薄）
                }
                for sensor_id, offset in position_offsets.items():
                    sensor_temp = primary_temp + offset
                    state.multi_sensor_history[sensor_id].append((current_time, sensor_temp))
            
            # 同时记录单传感器历史（兼容旧接口）
            state.cold_trap_history.append((current_time, primary_temp))
            state.current_cold_trap_temp = primary_temp
            
            # 累积运行时间
            if state.last_defrost_time:
                state.cumulative_running_hours += 10.0 / 3600.0  # 10秒数据点
                
        except Exception as e:
            print(f"[{self.service_id}] 处理遥测失败: {e}")
            self._increment_metric("errors")
    
    async def _handle_endpoint(self, message: Dict):
        """处理终点检测（批次完成时调度除霜）"""
        try:
            if not validate_message(message, MESSAGE_TYPES['ENDPOINT']):
                return
            
            payload = extract_payload(message)
            device_id = payload.get('device_id')
            phase = payload.get('cycle_phase')
            
            state = self.device_states.get(device_id)
            if not state:
                return
            
            # 二次干燥完成，批次结束，调度除霜
            if phase == 'secondary_drying':
                now = time.time()
                state.next_scheduled_defrost = now + 5 * 60  # 5分钟后除霜
                print(f"[{self.service_id}] 设备{device_id} 批次完成，调度5分钟后除霜")
                
        except Exception as e:
            print(f"[{self.service_id}] 处理终点检测失败: {e}")
    
    async def _handle_defrost_command(self, message: Dict):
        """处理除霜命令"""
        try:
            if not validate_message(message, MESSAGE_TYPES['DEFROST_COMMAND']):
                return
            
            payload = extract_payload(message)
            cmd = DefrostCommand(**payload)
            
            state = self.device_states.get(cmd.device_id)
            if not state:
                return
            
            if cmd.command == 'start' and not state.defrost_in_progress:
                await self._start_defrost(cmd.device_id, state, cmd.heating_power_pct, cmd.max_duration_minutes)
            elif cmd.command == 'stop' and state.defrost_in_progress:
                await self._stop_defrost(cmd.device_id, state, completed=False)
            elif cmd.command == 'cancel' and state.next_scheduled_defrost:
                state.next_scheduled_defrost = None
                print(f"[{self.service_id}] 设备{cmd.device_id} 取消调度的除霜")
                
        except Exception as e:
            print(f"[{self.service_id}] 处理除霜命令失败: {e}")
    
    async def _optimize_defrost(self, device_id: int, state: DeviceDefrostState):
        """优化除霜决策"""
        # 获取当前电价
        now = datetime.now(timezone.utc)
        hour_of_day = now.hour
        
        # 简化的电价模型（实际应从数据库查询）
        if hour_of_day in [7, 8, 9, 10, 18, 19, 20, 21, 22, 23]:
            electricity_price = 1.2  # 峰时
        elif hour_of_day in [11, 12, 13, 14, 15, 16, 17]:
            electricity_price = 0.8  # 平时
        else:
            electricity_price = 0.4  # 谷时
        
        # 判断是否有批次在运行
        is_batch_running = state.batch_id is not None
        
        # 检查调度的除霜
        if state.next_scheduled_defrost and time.time() >= state.next_scheduled_defrost:
            state.next_scheduled_defrost = None
            need_defrost = True
            recommended_interval = 12.0
            recommended_power = self.power_profile.get('main_heating_power_pct', 80.0)
            estimated_saving = 10.0
        else:
            # 运行优化器
            need_defrost, recommended_interval, recommended_power, estimated_saving = \
                self.optimizer.optimize(state, electricity_price, hour_of_day, is_batch_running)
        
        # 计算平均温度和趋势
        if len(state.cold_trap_history) >= 10:
            times = np.array([t[0] for t in state.cold_trap_history])
            temps = np.array([t[1] for t in state.cold_trap_history])
            temp_avg = float(np.mean(temps[-60:]))
            temp_trend = self.optimizer.frost_estimator._calculate_trend(times, temps)
        else:
            temp_avg = state.current_cold_trap_temp
            temp_trend = 0.0
        
        # 发布优化建议
        optimization = DefrostOptimization(
            device_id=device_id,
            batch_id=state.batch_id,
            timestamp=now.isoformat(),
            estimated_frost_thickness_mm=round(state.estimated_frost_thickness, 2),
            cold_trap_temp_avg=round(temp_avg, 2),
            cold_trap_temp_trend=round(temp_trend, 3),
            recommended_defrost_interval_hours=round(recommended_interval, 1),
            recommended_heating_power_pct=round(recommended_power, 1),
            estimated_energy_saving=estimated_saving,
            defrost_status='in_progress' if state.defrost_in_progress else ('pending' if need_defrost else 'idle')
        )
        
        message = MessageFactory.create_defrost_optimization(optimization, self.service_id)
        await self.publish(CHANNELS['DEFROST_OPTIMIZATION'], message)
        self._increment_metric("messages_published")
        
        # 如果需要除霜，自动启动
        if need_defrost and not state.defrost_in_progress:
            await self._start_defrost(device_id, state, recommended_power)
    
    async def _start_defrost(self, device_id: int, state: DeviceDefrostState, 
                              power_pct: float, max_duration: Optional[int] = None):
        """启动除霜"""
        state.defrost_in_progress = True
        state.defrost_phase = 'preheat'
        state.defrost_start_time = time.time()
        state.defrost_power_pct = power_pct
        
        now = datetime.now(timezone.utc)
        status = DefrostStatus(
            device_id=device_id,
            timestamp=now.isoformat(),
            status='defrosting',
            progress_pct=0.0,
            current_temp=state.current_cold_trap_temp,
            target_temp=state.target_defrost_temp,
            batch_id=state.batch_id
        )
        
        message = MessageFactory.create_defrost_status(status, self.service_id)
        await self.publish(CHANNELS['DEFROST_STATUS'], message)
        self._increment_metric("messages_published")
        
        print(f"[{self.service_id}] 设备{device_id} 开始除霜, 功率: {power_pct:.1f}%, "
              f"最大厚度: {state.estimated_frost_thickness:.1f}mm")
    
    async def _control_defrost_process(self, device_id: int, state: DeviceDefrostState):
        """控制除霜过程"""
        if not state.defrost_in_progress or state.defrost_start_time is None:
            return
        
        elapsed = time.time() - state.defrost_start_time
        total_duration = (
            self.power_profile.get('preheat_duration_minutes', 10) +
            self.power_profile.get('main_duration_minutes', 30) +
            self.power_profile.get('soak_duration_minutes', 10) +
            self.power_profile.get('cooldown_duration_minutes', 10)
        ) * 60
        
        # 计算进度
        progress = min(100.0, (elapsed / total_duration) * 100)
        
        # 阶段判断
        preheat_end = self.power_profile.get('preheat_duration_minutes', 10) * 60
        main_end = preheat_end + self.power_profile.get('main_duration_minutes', 30) * 60
        soak_end = main_end + self.power_profile.get('soak_duration_minutes', 10) * 60
        
        if elapsed < preheat_end:
            state.defrost_phase = 'preheat'
            power = self.power_profile.get('preheat_power_pct', 30.0)
        elif elapsed < main_end:
            state.defrost_phase = 'main_heating'
            power = self.power_profile.get('main_heating_power_pct', 80.0)
        elif elapsed < soak_end:
            state.defrost_phase = 'soak'
            power = self.power_profile.get('soak_power_pct', 50.0)
        else:
            state.defrost_phase = 'cooldown'
            power = 0.0
            
            # 冷却完成，结束除霜
            if elapsed >= total_duration:
                await self._stop_defrost(device_id, state, completed=True)
                return
        
        # 自适应功率调整
        if self.power_profile.get('adaptive_power_enabled', True):
            temp_diff = state.target_defrost_temp - state.current_cold_trap_temp
            if temp_diff > 20.0:
                power *= 1.2
            elif temp_diff < 5.0:
                power *= 0.8
        
        power = min(power, self.power_profile.get('max_power_pct', 80.0))
        state.defrost_power_pct = power
        
        # 发布状态
        now = datetime.now(timezone.utc)
        status = DefrostStatus(
            device_id=device_id,
            timestamp=now.isoformat(),
            status='defrosting',
            progress_pct=round(progress, 1),
            current_temp=state.current_cold_trap_temp,
            target_temp=state.target_defrost_temp,
            energy_consumed_kwh=round(elapsed / 3600.0 * 5.0 * (power / 100.0), 2),
            batch_id=state.batch_id
        )
        
        message = MessageFactory.create_defrost_status(status, self.service_id)
        await self.publish(CHANNELS['DEFROST_STATUS'], message)
        self._increment_metric("messages_published")
    
    async def _stop_defrost(self, device_id: int, state: DeviceDefrostState, completed: bool):
        """停止除霜"""
        energy_consumed = 0.0
        if state.defrost_start_time:
            elapsed = time.time() - state.defrost_start_time
            energy_consumed = elapsed / 3600.0 * 5.0 * (state.defrost_power_pct / 100.0)
        
        state.defrost_in_progress = False
        state.defrost_phase = None
        state.last_defrost_time = time.time()
        state.estimated_frost_thickness = 0.0
        state.cumulative_running_hours = 0.0
        state.defrost_power_pct = 0.0
        
        now = datetime.now(timezone.utc)
        status = DefrostStatus(
            device_id=device_id,
            timestamp=now.isoformat(),
            status='defrost_complete' if completed else 'idle',
            progress_pct=100.0 if completed else 0.0,
            current_temp=state.current_cold_trap_temp,
            energy_consumed_kwh=round(energy_consumed, 2),
            batch_id=state.batch_id
        )
        
        message = MessageFactory.create_defrost_status(status, self.service_id)
        await self.publish(CHANNELS['DEFROST_STATUS'], message)
        self._increment_metric("messages_published")
        
        print(f"[{self.service_id}] 设备{device_id} 除霜{'完成' if completed else '中止'}, "
              f"能耗: {energy_consumed:.2f}kWh")
    
    async def _handle_config_update(self, message: Dict):
        """处理配置更新"""
        try:
            payload = extract_payload(message)
            if payload.get('config_type') == 'defrost':
                self.config = config_loader.load_defrost_config()
                self.optimizer = DefrostOptimizer(asdict(self.config))
                print(f"[{self.service_id}] 配置已更新")
        except Exception as e:
            print(f"[{self.service_id}] 配置更新失败: {e}")


if __name__ == "__main__":
    service = DefrostOptimizerService()
    
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        asyncio.run(service.stop())
    except Exception as e:
        print(f"服务异常退出: {e}")
