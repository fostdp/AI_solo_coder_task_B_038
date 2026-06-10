import numpy as np
from typing import Dict, List, Tuple


class MultiSensorFusion:
    """多传感器数据融合 - 处理冷阱多个温度传感器数据"""
    
    def __init__(self, config: Dict = None):
        config = config or {}
        self.num_sensors = config.get('num_sensors', 5)
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        self.min_valid_sensors = config.get('min_valid_sensors', 3)
        self.temperature_consistency_threshold = config.get('temperature_consistency_threshold', 5.0)  # ℃
        
        # 默认位置权重（入口结霜最严重，权重最高）
        self.default_weights = {
            1: 0.35,  # inlet
            2: 0.20,  # coil_1
            3: 0.20,  # coil_2
            4: 0.15,  # coil_3
            5: 0.10,  # outlet
        }
        
        # 位置修正因子（不同位置结霜速率不同）
        self.position_correction = {
            'inlet': 1.5,    # 入口结霜快50%
            'coil_1': 1.1,
            'coil_2': 1.0,
            'coil_3': 0.9,
            'outlet': 0.7,   # 出口结霜慢30%
        }
    
    def detect_outliers(self, sensor_readings: Dict[int, float]) -> Tuple[Dict[int, float], Dict[int, bool]]:
        """检测异常传感器"""
        if len(sensor_readings) < 3:
            return sensor_readings, {k: True for k in sensor_readings}
        
        values = list(sensor_readings.values())
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        threshold = self.outlier_threshold * mad / 0.6745
        
        is_valid = {}
        cleaned = {}
        for sensor_id, value in sensor_readings.items():
            deviation = abs(value - median)
            is_valid[sensor_id] = deviation <= threshold
            if is_valid[sensor_id]:
                cleaned[sensor_id] = value
        
        return cleaned, is_valid
    
    def calculate_sensor_health(self, sensor_id: int, 
                                current_value: float, 
                                historical_values: List[float]) -> float:
        """计算传感器健康度评分（0-1）"""
        if len(historical_values) < 10:
            return 1.0
        
        historical = np.array(historical_values)
        mean_hist = np.mean(historical)
        std_hist = np.std(historical)
        
        # 检查当前值是否在合理范围内
        z_score = abs(current_value - mean_hist) / (std_hist + 1e-10)
        if z_score > 5.0:
            return 0.3
        
        # 检查信号噪声水平
        noise_level = std_hist / (abs(mean_hist) + 1e-10)
        health = max(0.0, 1.0 - noise_level * 2.0)
        
        return float(health)
    
    def check_temperature_consistency(self, sensor_temps: Dict[int, float]) -> float:
        """检查传感器之间的温度一致性（0-1）"""
        if len(sensor_temps) < 2:
            return 1.0
        
        values = list(sensor_temps.values())
        temp_range = max(values) - min(values)
        
        # 温度范围越大，一致性越差
        consistency = max(0.0, 1.0 - temp_range / self.temperature_consistency_threshold)
        return float(consistency)
    
    def fuse_temperatures(self, 
                          sensor_temps: Dict[int, float],
                          sensor_weights: Dict[int, float],
                          sensor_health: Dict[int, float],
                          sensor_positions: Dict[int, str]) -> Tuple[float, Dict[int, float]]:
        """
        多传感器温度融合
        返回：(融合后的温度, 每个传感器的修正后温度)
        """
        # 1. 异常值检测
        cleaned_temps, is_valid = self.detect_outliers(sensor_temps)
        
        if len(cleaned_temps) < self.min_valid_sensors:
            # 有效传感器太少，使用中位数作为降级方案
            median_temp = float(np.median(list(sensor_temps.values())))
            corrected_temps = {k: median_temp for k in sensor_temps}
            return median_temp, corrected_temps
        
        # 2. 计算一致性评分
        consistency = self.check_temperature_consistency(cleaned_temps)
        
        # 3. 动态调整权重
        weights = {}
        for sensor_id in cleaned_temps:
            base_weight = sensor_weights.get(sensor_id, 1.0 / len(cleaned_temps))
            health = sensor_health.get(sensor_id, 1.0)
            valid_factor = 1.0 if is_valid.get(sensor_id, True) else 0.0
            weights[sensor_id] = base_weight * max(health, 0.01) * valid_factor
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # 所有权重为0，使用平均权重
            avg_weight = 1.0 / len(cleaned_temps)
            weights = {k: avg_weight for k in cleaned_temps}
        
        # 4. 加权融合
        fused_temp = 0.0
        corrected_temps = {}
        for sensor_id, temp in cleaned_temps.items():
            position = sensor_positions.get(sensor_id, 'coil_2')
            correction = self.position_correction.get(position, 1.0)
            
            # 根据位置修正温度（入口温度偏高需要修正）
            corrected_temp = temp / correction
            corrected_temps[sensor_id] = corrected_temp
            
            fused_temp += weights[sensor_id] * corrected_temp
        
        return float(fused_temp), corrected_temps
    
    def estimate_thickness_distribution(self,
                                        sensor_temps: Dict[int, float],
                                        base_temp: float,
                                        calibration_factor: float) -> Dict[int, float]:
        """估算每个传感器位置的结霜厚度分布"""
        thickness_dist = {}
        
        for sensor_id, temp in sensor_temps.items():
            temp_diff = temp - base_temp
            # 温度每升高2°C，结霜厚度约增加1mm
            thickness = max(0.0, temp_diff * calibration_factor)
            thickness_dist[sensor_id] = min(thickness, 10.0)
        
        return thickness_dist
