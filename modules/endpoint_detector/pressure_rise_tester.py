"""
压力升测试（Pressure Rise Test, PRT）
通过关闭真空系统，测量干燥室压力变化速率来判定干燥终点
"""

import numpy as np
from typing import Dict, Optional, Tuple, Deque
from collections import deque
from .types import PRTState, PRTResult


class PressureRiseTester:
    """压力升测试器"""

    def __init__(self, config: Dict):
        self.enabled = config.get('enabled', True)
        self.test_duration = config.get('test_duration_seconds', 120)
        self.measurement_interval = config.get('measurement_interval_seconds', 5)
        self.endpoint_threshold = config.get('endpoint_threshold_pa_per_min', 0.05)
        self.min_interval = config.get('min_interval_between_tests_minutes', 30) * 60
        self.auto_trigger = config.get('auto_trigger_enabled', True)
        self.min_data_quality = config.get('min_data_quality', 0.6)
        self.confirmation_count = config.get('confirmation_count', 2)

        # 测试结果历史（用于多级确认）
        self._test_results: Deque[bool] = deque(maxlen=5)
        self._confirmed_endpoint = False

    def start_test(self, prt_state: PRTState, initial_pressure: float, current_time: float) -> bool:
        """
        开始压力升测试

        Args:
            prt_state: PRT状态对象
            initial_pressure: 初始压力（Pa）
            current_time: 当前时间戳

        Returns:
            bool: 是否成功启动测试
        """
        if not self.enabled:
            return False

        if current_time - prt_state.last_test_time < self.min_interval:
            return False

        if prt_state.in_progress:
            return False

        prt_state.in_progress = True
        prt_state.start_time = current_time
        prt_state.initial_pressure = initial_pressure
        prt_state.measurements = [(current_time, initial_pressure)]
        prt_state.last_test_time = current_time

        return True

    def record_measurement(self, prt_state: PRTState, pressure: float, current_time: float) -> None:
        """
        记录测量值

        Args:
            prt_state: PRT状态对象
            pressure: 压力测量值（Pa）
            current_time: 当前时间戳
        """
        if not prt_state.in_progress:
            return

        prt_state.measurements.append((current_time, pressure))

    def _filter_pressure_data(self, times: np.ndarray, pressures: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """过滤压力数据并计算质量评分"""
        if len(pressures) < 5:
            return times, pressures, 1.0

        # 1. 异常值剔除
        median = np.median(pressures)
        mad = np.median(np.abs(pressures - median))
        threshold = 3.0 * mad / 0.6745

        mask = np.abs(pressures - median) <= threshold
        filtered_times = times[mask]
        filtered_pressures = pressures[mask]

        if len(filtered_pressures) < 5:
            return times, pressures, 0.3

        # 2. 滑动平均平滑
        if len(filtered_pressures) >= 7:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(filtered_pressures, kernel, mode='same')
        else:
            smoothed = filtered_pressures

        # 3. 计算数据质量评分
        # 基于线性拟合的R²
        dt = (filtered_times - filtered_times[0]) / 60.0
        slope, intercept = np.polyfit(dt, filtered_pressures, 1)
        predicted = slope * dt + intercept
        ss_res = np.sum((filtered_pressures - predicted) ** 2)
        ss_tot = np.sum((filtered_pressures - np.mean(filtered_pressures)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        # 基于测量值的一致性
        consistency = 1.0 - min(1.0, np.std(filtered_pressures) / (np.mean(np.abs(filtered_pressures)) + 1e-10))

        quality_score = float(max(0.0, min(1.0, (r_squared * 0.6 + consistency * 0.4))))

        return filtered_times, smoothed, quality_score

    def check_test_complete(self, prt_state: PRTState, current_time: float) -> Optional[PRTResult]:
        """
        检查测试是否完成并计算结果

        Args:
            prt_state: PRT状态对象
            current_time: 当前时间戳

        Returns:
            Optional[PRTResult]: 测试完成返回结果，否则返回None
        """
        if not prt_state.in_progress:
            return None

        if prt_state.start_time is None:
            prt_state.in_progress = False
            return None

        elapsed = current_time - prt_state.start_time

        if elapsed < self.test_duration:
            return None

        # 计算压力升速率
        measurements = prt_state.measurements
        if len(measurements) < 5:
            prt_state.in_progress = False
            prt_state.measurements = []
            return None

        times = np.array([m[0] for m in measurements])
        pressures = np.array([m[1] for m in measurements])

        # 数据滤波和质量检查
        filtered_times, filtered_pressures, quality_score = self._filter_pressure_data(times, pressures)

        # 数据质量过低，测试无效
        if quality_score < self.min_data_quality:
            prt_state.in_progress = False
            prt_state.measurements = []
            return None

        # 线性拟合计算压力升速率（Pa/min）
        dt = (filtered_times - filtered_times[0]) / 60.0  # 转换为分钟
        slope, intercept = np.polyfit(dt, filtered_pressures, 1)

        pressure_rise_rate = float(slope)
        is_endpoint = pressure_rise_rate < self.endpoint_threshold

        # 多级确认
        self._test_results.append(is_endpoint)
        if len(self._test_results) >= self.confirmation_count:
            recent_results = list(self._test_results)[-self.confirmation_count:]
            if all(recent_results):
                self._confirmed_endpoint = True

        # 计算检测置信度
        detection_confidence = float(quality_score * min(1.0, max(0.0, 1.0 - pressure_rise_rate / max(self.endpoint_threshold, 1e-10))))

        result = PRTResult(
            test_start_time=float(prt_state.start_time),
            test_end_time=current_time,
            initial_pressure_pa=float(filtered_pressures[0]),
            final_pressure_pa=float(filtered_pressures[-1]),
            pressure_rise_pa_per_min=pressure_rise_rate,
            test_duration_seconds=int(elapsed),
            is_endpoint_detected=self._confirmed_endpoint,
            detection_confidence=detection_confidence,
            data_quality_score=quality_score,
            test_status="completed"
        )

        # 重置测试状态
        prt_state.in_progress = False
        prt_state.measurements = []

        return result

    def reset(self):
        """重置测试状态"""
        self._test_results.clear()
        self._confirmed_endpoint = False
