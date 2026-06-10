import numpy as np
from typing import Dict, Tuple
import time

from .types import DeviceDefrostState
from .thickness_estimator import FrostThicknessEstimator


class DefrostOptimizer:
    """除霜优化器"""
    
    def __init__(self, config: Dict):
        self.optimization_config = config.get('optimization', {})
        self.thresholds = config.get('thresholds', {})
        self.power_profile = config.get('power_profile', {})
        self.energy_model = config.get('energy_model', {})
        self.frost_estimator = FrostThicknessEstimator(config.get('frost_thickness_estimation', {}))
    
    def optimize(self, state: DeviceDefrostState, electricity_price: float, 
                 hour_of_day: int, is_batch_running: bool) -> Tuple[bool, float, float, float]:
        """
        优化除霜决策
        返回：(是否需要除霜, 推荐间隔小时, 推荐加热功率%, 预计节能)
        """
        current_time = time.time()
        
        # 估算结霜厚度（优先使用多传感器融合）
        if self.frost_estimator.use_multi_sensor:
            frost_thickness, thickness_dist, quality_score = self.frost_estimator.estimate_multi_sensor(
                state, current_time
            )
        else:
            temp_history = list(state.cold_trap_history)
            frost_thickness = self.frost_estimator.estimate(temp_history)
            thickness_dist = {1: frost_thickness}
            quality_score = 0.5
        
        state.estimated_frost_thickness = frost_thickness
        
        # 计算温度趋势（使用融合后的温度历史）
        if len(state.fused_temperature_history) >= 10:
            temp_history = list(state.fused_temperature_history)
        elif len(state.cold_trap_history) >= 10:
            temp_history = list(state.cold_trap_history)
        else:
            temp_history = []
        
        if temp_history:
            times = np.array([t[0] for t in temp_history])
            temps = np.array([t[1] for t in temp_history])
            temp_trend = self.frost_estimator._calculate_trend(times, temps)
        else:
            temp_trend = 0.0
        
        # 如果除霜正在进行，根据阶段返回功率
        if state.defrost_in_progress and state.defrost_phase:
            phase = state.defrost_phase
            if phase == 'cooldown':
                return False, 12.0, 0.0, 0.0
            
            if phase == 'preheat':
                recommended_power = self.power_profile.get('preheat_power_pct', 30.0)
            elif phase == 'main_heating':
                recommended_power = self.power_profile.get('main_heating_power_pct', 80.0)
            elif phase == 'soak':
                recommended_power = self.power_profile.get('soak_power_pct', 50.0)
            else:
                recommended_power = self._calculate_optimal_power(
                    frost_thickness, temp_trend, electricity_price
                )
            
            # 自适应功率调整
            if self.power_profile.get('adaptive_power_enabled', True):
                temp_diff = state.target_defrost_temp - state.current_cold_trap_temp
                if temp_diff > 20.0:
                    recommended_power *= 1.2
                elif temp_diff < 5.0:
                    recommended_power *= 0.8
                recommended_power = min(recommended_power, self.power_profile.get('max_power_pct', 80.0))
            
            return False, 12.0, recommended_power, 0.0
        
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
