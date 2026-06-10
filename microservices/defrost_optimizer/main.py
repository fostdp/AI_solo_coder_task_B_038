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
from dataclasses import dataclass, field
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    MicroserviceBase,
    CHANNELS, SERVICE_IDS, MESSAGE_TYPES,
    TelemetryData, DefrostOptimization, DefrostCommand, DefrostStatus,
    MessageFactory, validate_message, extract_payload,
    config_loader, DefrostConfig
)


@dataclass
class DeviceDefrostState:
    """设备除霜状态"""
    device_id: int
    batch_id: Optional[str] = None
    
    # 冷阱温度历史
    cold_trap_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=720)  # 2小时（10秒间隔）
    )
    
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


class FrostThicknessEstimator:
    """结霜厚度估算器"""
    
    def __init__(self, config: Dict):
        self.method = config.get('method', 'thermal_resistance')
        self.base_temp = config.get('base_cold_trap_temp', -80.0)
        self.max_thickness = config.get('max_frost_thickness_mm', 5.0)
        self.calibration_factor = config.get('calibration_factor', 1.2)
        self.temp_window = config.get('temp_window_minutes', 60)
    
    def estimate(self, temp_history: List[Tuple[float, float]]) -> float:
        """估算结霜厚度"""
        if len(temp_history) < 10:
            return 0.0
        
        times = np.array([t[0] for t in temp_history])
        temps = np.array([t[1] for t in temp_history])
        
        if self.method == 'thermal_resistance':
            return self._thermal_resistance_method(times, temps)
        else:
            return self._empirical_method(times, temps)
    
    def _thermal_resistance_method(self, times: np.ndarray, temps: np.ndarray) -> float:
        """基于热阻的估算方法"""
        # 计算平均温度和温度趋势
        avg_temp = np.mean(temps[-60:])  # 最近10分钟平均
        temp_trend = self._calculate_trend(times, temps)
        
        # 温度差越大，结霜越厚
        temp_diff = avg_temp - self.base_temp
        
        # 厚度与温度差成正比（简化模型）
        thickness = max(0.0, temp_diff * self.calibration_factor * 0.1)
        
        # 趋势修正：温度上升越快，结霜越严重
        if temp_trend > 0:
            thickness += temp_trend * 5.0
        
        return min(thickness, self.max_thickness)
    
    def _empirical_method(self, times: np.ndarray, temps: np.ndarray) -> float:
        """经验公式法"""
        # 基于运行时间和温度的经验公式
        avg_temp = np.mean(temps[-30:])
        temp_std = np.std(temps[-30:])
        
        # 经验公式：厚度 = 系数 * 温度偏差 * 时间因子
        temp_deviation = max(0.0, avg_temp - self.base_temp)
        runtime_factor = min(1.0, len(temps) / 360.0)  # 2小时数据归一化
        
        thickness = temp_deviation * runtime_factor * self.calibration_factor * 0.08
        
        # 温度波动大表示结霜不稳定
        if temp_std > 1.0:
            thickness *= 1.2
        
        return min(thickness, self.max_thickness)
    
    def _calculate_trend(self, times: np.ndarray, values: np.ndarray) -> float:
        """计算趋势斜率（℃/hour）"""
        if len(values) < 10:
            return 0.0
        
        # 取最近60个点（10分钟）
        recent_times = times[-60:]
        recent_values = values[-60:]
        
        # 线性拟合
        dt = (recent_times - recent_times[0]) / 3600.0  # 转换为小时
        slope, _ = np.polyfit(dt, recent_values, 1)
        
        return float(slope)


class DefrostOptimizer:
    """除霜优化器"""
    
    def __init__(self, config: DefrostConfig):
        self.optimization_config = config.optimization
        self.thresholds = config.thresholds
        self.power_profile = config.power_profile
        self.energy_model = config.__dict__.get('energy_model', {})
        self.frost_estimator = FrostThicknessEstimator(config.frost_thickness_estimation)
    
    def optimize(self, state: DeviceDefrostState, electricity_price: float, 
                 hour_of_day: int, is_batch_running: bool) -> Tuple[bool, float, float, float]:
        """
        优化除霜决策
        返回：(是否需要除霜, 推荐间隔小时, 推荐加热功率%, 预计节能)
        """
        # 估算结霜厚度
        temp_history = list(state.cold_trap_history)
        frost_thickness = self.frost_estimator.estimate(temp_history)
        state.estimated_frost_thickness = frost_thickness
        
        # 计算温度趋势
        if len(temp_history) >= 10:
            times = np.array([t[0] for t in temp_history])
            temps = np.array([t[1] for t in temp_history])
            temp_trend = self.frost_estimator._calculate_trend(times, temps)
        else:
            temp_trend = 0.0
        
        # 判断是否需要除霜
        need_defrost = self._check_defrost_needed(state, frost_thickness, temp_trend)
        
        # 计算推荐间隔
        recommended_interval = self._calculate_optimal_interval(
            state, frost_thickness, temp_trend, electricity_price, hour_of_day
        )
        
        # 计算推荐功率
        recommended_power = self._calculate_optimal_power(
            frost_thickness, temp_trend, electricity_price
        )
        
        # 计算预计节能
        estimated_saving = self._calculate_energy_saving(
            frost_thickness, recommended_power, recommended_interval,
            state, is_batch_running
        )
        
        # 批次运行中不允许除霜（除非配置允许）
        if is_batch_running and not self.optimization_config.get('allow_defrost_during_batches', False):
            need_defrost = False
        
        return need_defrost, recommended_interval, recommended_power, estimated_saving
    
    def _check_defrost_needed(self, state: DeviceDefrostState, 
                               frost_thickness: float, temp_trend: float) -> bool:
        """检查是否需要除霜"""
        # 最短运行时间限制
        min_running = self.optimization_config.get('min_running_hours_before_defrost', 8.0)
        if state.last_defrost_time:
            hours_since = (time.time() - state.last_defrost_time) / 3600.0
            if hours_since < min_running:
                return False
        
        # 最短除霜间隔
        min_interval = self.thresholds.get('min_defrost_interval_hours', 4.0)
        if state.last_defrost_time:
            hours_since = (time.time() - state.last_defrost_time) / 3600.0
            if hours_since < min_interval:
                return False
        
        # 最大除霜间隔（强制执行）
        max_interval = self.thresholds.get('max_defrost_interval_hours', 24.0)
        if state.last_defrost_time:
            hours_since = (time.time() - state.last_defrost_time) / 3600.0
            if hours_since >= max_interval:
                return True
        
        # 厚度阈值
        thickness_trigger = self.thresholds.get('frost_thickness_trigger_mm', 3.0)
        if frost_thickness >= thickness_trigger:
            return True
        
        # 温度趋势阈值
        temp_trend_trigger = self.thresholds.get('temp_trend_trigger', 0.5)
        if temp_trend >= temp_trend_trigger:
            return True
        
        return False
    
    def _calculate_optimal_interval(self, state: DeviceDefrostState, 
                                     frost_thickness: float, temp_trend: float,
                                     electricity_price: float, hour_of_day: int) -> float:
        """计算最佳除霜间隔"""
        base_interval = 12.0  # 基础间隔12小时
        
        # 根据结霜厚度调整
        if frost_thickness < 1.0:
            interval = base_interval * 1.5
        elif frost_thickness < 2.0:
            interval = base_interval * 1.2
        elif frost_thickness < 3.0:
            interval = base_interval
        else:
            interval = base_interval * 0.8
        
        # 根据温度趋势调整
        if temp_trend > 1.0:
            interval *= 0.7
        elif temp_trend > 0.5:
            interval *= 0.85
        
        # 优先谷电时段
        if self.optimization_config.get('prefer_valley_electricity', True):
            # 谷电时段（0-6点）延长间隔到谷电
            if 6 <= hour_of_day < 22:  # 非谷电时段
                hours_to_valley = (24 - hour_of_day) % 24
                if hours_to_valley < interval:
                    interval = max(interval, hours_to_valley + 1)
        
        return max(self.thresholds.get('min_defrost_interval_hours', 4.0),
                   min(interval, self.thresholds.get('max_defrost_interval_hours', 24.0)))
    
    def _calculate_optimal_power(self, frost_thickness: float, temp_trend: float,
                                  electricity_price: float) -> float:
        """计算最佳加热功率"""
        max_power = self.power_profile.get('max_power_pct', 80.0)
        min_power = self.power_profile.get('min_power_pct', 20.0)
        main_power = self.power_profile.get('main_heating_power_pct', 80.0)
        
        # 基础功率
        power = main_power
        
        # 根据结霜厚度调整
        if frost_thickness < 1.5:
            power *= 0.7
        elif frost_thickness < 3.0:
            power *= 0.85
        elif frost_thickness < 4.0:
            power *= 1.0
        else:
            power *= 1.1
        
        # 根据电价调整（谷电用满功率，峰电降低功率）
        if self.power_profile.get('adaptive_power_enabled', True):
            # 电价归一化到 [0.4, 1.2]
            normalized_price = (electricity_price - 0.4) / (1.2 - 0.4)
            adjustment = 1.0 - normalized_price * self.power_profile.get('power_adjustment_factor', 0.1)
            power *= adjustment
        
        return max(min_power, min(power, max_power))
    
    def _calculate_energy_saving(self, frost_thickness: float, power_pct: float,
                                  interval_hours: float, state: DeviceDefrostState,
                                  is_batch_running: bool) -> float:
        """计算预计节能"""
        # 计算当前运行方式的能耗
        specific_energy = self.energy_model.get('specific_energy_kwh_per_mm_frost', 0.5)
        standby_power = self.energy_model.get('standby_power_kw', 2.0)
        efficiency = self.energy_model.get('efficiency_coefficient', 0.85)
        
        # 除霜能耗
        defrost_energy = frost_thickness * specific_energy * (power_pct / 100.0)
        
        # 结霜导致的额外能耗（制冷效率降低）
        penalty_factor = 1.0 + frost_thickness * 0.1  # 每毫米霜增加10%能耗
        extra_energy_per_hour = standby_power * (penalty_factor - 1.0)
        
        # 提前除霜节省的能耗
        hours_saved = max(0.0, 12.0 - interval_hours)
        total_saving = extra_energy_per_hour * hours_saved * efficiency
        
        # 批次中除霜的能耗惩罚（暂停生产）
        if is_batch_running:
            total_saving *= 0.5  # 减半，因为会影响生产
        
        return round(total_saving, 2)


class DefrostOptimizerService(MicroserviceBase):
    """冷阱除霜优化服务"""
    
    def __init__(self):
        super().__init__(SERVICE_IDS['DEFROST_OPTIMIZER'], 'defrost_optimizer')
        self.config: DefrostConfig = config_loader.load_defrost_config()
        self.optimizer = DefrostOptimizer(self.config)
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
            
            # 记录冷阱温度
            state.cold_trap_history.append((time.time(), telemetry.cold_trap_temp))
            state.current_cold_trap_temp = telemetry.cold_trap_temp
            
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
                self.optimizer = DefrostOptimizer(self.config)
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
