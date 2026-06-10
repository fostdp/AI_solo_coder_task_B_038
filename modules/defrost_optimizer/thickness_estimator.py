import numpy as np
from typing import Dict, List, Tuple
import time

from .types import DeviceDefrostState
from .sensor_fusion import MultiSensorFusion


class FrostThicknessEstimator:
    """结霜厚度估算器 - 支持多传感器融合"""
    
    def __init__(self, config: Dict):
        self.method = config.get('method', 'thermal_resistance')
        self.base_temp = config.get('base_cold_trap_temp', -80.0)
        self.max_thickness = config.get('max_frost_thickness_mm', 5.0)
        self.calibration_factor = config.get('calibration_factor', 1.2)
        self.temp_window = config.get('temp_window_minutes', 60)
        
        # 多传感器融合
        self.multi_sensor_fusion = MultiSensorFusion(config.get('multi_sensor', {}))
        self.use_multi_sensor = config.get('use_multi_sensor', True)
    
    def estimate(self, temp_history: List[Tuple[float, float]]) -> float:
        """估算结霜厚度（单传感器兼容接口）"""
        if len(temp_history) < 10:
            return 0.0
        
        times = np.array([t[0] for t in temp_history])
        temps = np.array([t[1] for t in temp_history])
        
        if self.method == 'thermal_resistance':
            return self._thermal_resistance_method(times, temps)
        else:
            return self._empirical_method(times, temps)
    
    def estimate_multi_sensor(self,
                              state: DeviceDefrostState,
                              current_timestamp: float) -> Tuple[float, Dict[int, float], float]:
        """
        多传感器融合估算结霜厚度
        返回：(综合厚度, 各位置厚度分布, 融合质量评分)
        """
        # 1. 获取各传感器当前温度
        sensor_temps = {}
        historical_values = {}
        for sensor_id, history in state.multi_sensor_history.items():
            if len(history) > 0:
                sensor_temps[sensor_id] = history[-1][1]
                historical_values[sensor_id] = [h[1] for h in history]
        
        if len(sensor_temps) == 0:
            # 没有多传感器数据，回退到单传感器
            if len(state.cold_trap_history) >= 10:
                thickness = self.estimate(list(state.cold_trap_history))
                return thickness, {1: thickness}, 0.5
            return 0.0, {1: 0.0}, 0.0
        
        # 2. 更新传感器健康状态
        for sensor_id in sensor_temps:
            if sensor_id in historical_values:
                state.sensor_health[sensor_id] = self.multi_sensor_fusion.calculate_sensor_health(
                    sensor_id, sensor_temps[sensor_id], historical_values[sensor_id]
                )
        
        # 3. 多传感器温度融合
        fused_temp, corrected_temps = self.multi_sensor_fusion.fuse_temperatures(
            sensor_temps,
            state.sensor_weights,
            state.sensor_health,
            state.sensor_positions
        )
        
        # 4. 计算融合质量评分
        consistency = self.multi_sensor_fusion.check_temperature_consistency(sensor_temps)
        health_scores = list(state.sensor_health.values())
        avg_health = np.mean(health_scores) if health_scores else 1.0
        quality_score = float(0.6 * consistency + 0.4 * avg_health)
        
        # 5. 估算各位置厚度分布
        thickness_dist = self.multi_sensor_fusion.estimate_thickness_distribution(
            corrected_temps,
            self.base_temp,
            self.calibration_factor
        )
        
        state.frost_thickness_distribution = thickness_dist
        
        # 6. 综合厚度（取入口和最大厚度的加权平均，因为入口结霜最严重）
        max_thickness = max(thickness_dist.values()) if thickness_dist else 0.0
        inlet_thickness = thickness_dist.get(1, max_thickness)  # 入口传感器
        combined_thickness = float(0.6 * max_thickness + 0.4 * inlet_thickness)
        combined_thickness = min(combined_thickness, self.max_thickness)
        
        # 7. 更新融合温度历史
        state.fused_temperature_history.append((current_timestamp, fused_temp))
        state.current_cold_trap_temp = fused_temp
        
        # 同时更新兼容的单传感器历史
        state.cold_trap_history.append((current_timestamp, fused_temp))
        
        return combined_thickness, thickness_dist, quality_score
    
    def _thermal_resistance_method(self, times: np.ndarray, temps: np.ndarray) -> float:
        """基于热阻的估算方法"""
        # 计算平均温度和温度趋势
        avg_temp = np.mean(temps[-60:])  # 最近10分钟平均
        temp_trend = self._calculate_trend(times, temps)
        
        # 温度差越大，结霜越厚
        temp_diff = avg_temp - self.base_temp
        
        # 厚度与温度差成正比（30℃温差对应5.0mm厚度）
        thickness = max(0.0, temp_diff * self.calibration_factor * (5.0 / 36.0))
        
        # 趋势修正：温度上升越快，结霜越严重
        if temp_trend > 0:
            thickness += temp_trend * 2.0
        
        return min(thickness, self.max_thickness)
    
    def _empirical_method(self, times: np.ndarray, temps: np.ndarray) -> float:
        """经验公式法（与热阻法基于相同原理）"""
        # 计算平均温度和温度趋势
        avg_temp = np.mean(temps[-30:])
        temp_trend = self._calculate_trend(times, temps)
        temp_std = np.std(temps[-30:])
        
        # 温度差越大，结霜越厚（与热阻法相同的系数）
        temp_diff = max(0.0, avg_temp - self.base_temp)
        thickness = temp_diff * self.calibration_factor * (5.0 / 36.0)
        
        # 趋势修正：温度上升越快，结霜越严重（与热阻法相同的系数）
        if temp_trend > 0:
            thickness += temp_trend * 1.0
        
        # 温度波动大表示结霜不稳定
        if temp_std > 1.0:
            thickness *= 1.1
        
        # 时间因子：数据越多估算越准确
        runtime_factor = min(1.0, len(temps) / 360.0)
        thickness *= (0.8 + 0.2 * runtime_factor)
        
        return min(thickness, self.max_thickness)
    
    def _calculate_trend(self, times: np.ndarray, values: np.ndarray) -> float:
        """计算趋势斜率（℃/hour）"""
        if len(values) < 10:
            return 0.0
        
        # 取最近60个点（10分钟）
        recent_times = times[-60:]
        recent_values = values[-60:]
        
        # 按时间排序（确保时间递增）
        sorted_indices = np.argsort(recent_times)
        recent_times = recent_times[sorted_indices]
        recent_values = recent_values[sorted_indices]
        
        # 线性拟合
        dt = (recent_times - recent_times[0]) / 3600.0  # 转换为小时
        slope, _ = np.polyfit(dt, recent_values, 1)
        
        return float(slope)
