"""
一阶导数法终点检测
通过检测温度曲线的一阶导数变化来判定干燥终点
"""

import numpy as np
from typing import Dict, List, Tuple


class FirstDerivativeDetector:
    """一阶导数法终点检测"""

    def __init__(self, config: Dict):
        self.window_size = config.get('window_size', 10)
        self.poly_order = config.get('poly_order', 2)
        self.primary_threshold = config.get('primary_drying_threshold', 0.05)
        self.secondary_threshold = config.get('secondary_drying_threshold', 0.02)
        self.consecutive_points = config.get('consecutive_points', 5)
        self.confirmation_window = config.get('confirmation_window', 3)  # 多级确认窗口
        self.min_stability_score = config.get('min_stability_score', 0.3)
        self._consecutive_count_primary = 0
        self._consecutive_count_secondary = 0
        self._primary_confirmed = False
        self._secondary_confirmed = False

    def savitzky_golay(self, y: np.ndarray, window_size: int, poly_order: int) -> np.ndarray:
        """Savitzky-Golay平滑滤波"""
        if len(y) < window_size:
            return y

        order = min(poly_order, window_size - 1)
        half_window = (window_size - 1) // 2

        coeffs = np.zeros(len(y))
        for i in range(len(y)):
            start = max(0, i - half_window)
            end = min(len(y), i + half_window + 1)
            segment = y[start:end]
            x = np.arange(len(segment)) - (i - start)

            if len(segment) >= order + 1:
                poly_coeffs = np.polyfit(x, segment, order)
                coeffs[i] = np.polyval(poly_coeffs, 0)
            else:
                coeffs[i] = y[i]

        return coeffs

    def compute_derivative(self, timestamps: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算一阶导数"""
        if len(timestamps) < 2:
            return np.array([]), np.array([])

        # 先平滑
        smoothed = self.savitzky_golay(values, self.window_size, self.poly_order)

        # 计算导数（℃/min）
        dt = np.diff(timestamps) / 60.0  # 转换为分钟
        dy = np.diff(smoothed)

        # 避免除以零
        dt = np.where(dt == 0, 1e-10, dt)
        derivatives = dy / dt

        return derivatives, smoothed

    def detect_primary_endpoint(self, timestamps: List[float], temps: List[float],
                                stability_score: float = 1.0) -> Tuple[bool, float, float]:
        """检测一次干燥终点（温度变化率下降到阈值以下）"""
        if len(timestamps) < self.window_size * 2:
            return False, 0.0, 0.0

        # 信号质量过低时不进行判定
        if stability_score < self.min_stability_score:
            return False, 0.0, 0.0

        ts_arr = np.array(timestamps)
        temp_arr = np.array(temps)

        derivatives, smoothed = self.compute_derivative(ts_arr, temp_arr)

        if len(derivatives) == 0:
            return False, 0.0, 0.0

        recent_deriv = derivatives[-self.consecutive_points:]
        avg_derivative = np.mean(np.abs(recent_deriv))

        # 根据信号质量动态调整阈值（质量越差阈值越严格）
        # 质量差时提高阈值要求，需要更稳定的信号才会判定
        quality_factor = 1.0 + (1.0 - stability_score) * 2.0
        adjusted_threshold = self.primary_threshold * quality_factor

        # 一次干燥终点：温度变化率 < 阈值（冰升华完成，温度变化趋于稳定）
        is_endpoint = avg_derivative < adjusted_threshold

        if is_endpoint:
            self._consecutive_count_primary += 1
        else:
            self._consecutive_count_primary = max(0, self._consecutive_count_primary - 1)

        # 多级确认：需要连续足够次数才判定
        # 只有第一次达到确认条件时返回True，之后返回False避免重复触发
        detected = False
        if self._consecutive_count_primary >= self.consecutive_points:
            if not self._primary_confirmed:
                self._primary_confirmed = True
            detected = True

        return detected, float(avg_derivative), float(smoothed[-1]) if len(smoothed) > 0 else 0.0

    def detect_secondary_endpoint(self, timestamps: List[float], temps: List[float],
                                  stability_score: float = 1.0) -> Tuple[bool, float, float]:
        """检测二次干燥终点（温度变化率极小）"""
        if len(timestamps) < self.window_size * 2:
            return False, 0.0, 0.0

        # 信号质量过低时不进行判定
        if stability_score < 0.3:
            return False, 0.0, 0.0

        ts_arr = np.array(timestamps)
        temp_arr = np.array(temps)

        derivatives, smoothed = self.compute_derivative(ts_arr, temp_arr)

        if len(derivatives) == 0:
            return False, 0.0, 0.0

        recent_deriv = derivatives[-self.consecutive_points:]
        avg_derivative = np.mean(np.abs(recent_deriv))

        # 根据信号质量动态调整阈值
        adjusted_threshold = self.secondary_threshold * (1.0 + (1.0 - stability_score) * 0.5)

        # 二次干燥终点：温度变化率 < 更小的阈值（解析干燥完成）
        is_endpoint = avg_derivative < adjusted_threshold

        if is_endpoint:
            self._consecutive_count_secondary += 1
        else:
            self._consecutive_count_secondary = max(0, self._consecutive_count_secondary - 1)

        # 多级确认
        detected = False
        if self._consecutive_count_secondary >= self.consecutive_points:
            if not self._secondary_confirmed:
                self._secondary_confirmed = True
                detected = True
            else:
                detected = True

        return detected, float(avg_derivative), float(smoothed[-1]) if len(smoothed) > 0 else 0.0

    def reset(self):
        """重置检测状态"""
        self._consecutive_count_primary = 0
        self._consecutive_count_secondary = 0
        self._primary_confirmed = False
        self._secondary_confirmed = False
