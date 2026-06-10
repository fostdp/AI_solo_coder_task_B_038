"""
信号滤波器
处理真空度和温度信号的噪声和波动
"""

import numpy as np
from typing import Dict, Tuple, Deque
from collections import deque


class SignalFilter:
    """信号滤波器 - 处理真空度和温度信号的噪声和波动"""

    def __init__(self, config: Dict = None):
        config = config or {}
        self.median_window = config.get('median_window', 5)
        self.ma_window = config.get('moving_average_window', 7)
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        self.min_stability_score = config.get('min_stability_score', 0.3)

        # 历史缓存
        self._raw_history: Deque[float] = deque(maxlen=60)
        self._filtered_history: Deque[float] = deque(maxlen=60)

    def median_filter(self, data: np.ndarray) -> np.ndarray:
        """中值滤波 - 去除脉冲噪声"""
        if len(data) < self.median_window:
            return data

        filtered = np.zeros_like(data)
        half_win = self.median_window // 2

        for i in range(len(data)):
            start = max(0, i - half_win)
            end = min(len(data), i + half_win + 1)
            filtered[i] = np.median(data[start:end])

        return filtered

    def moving_average(self, data: np.ndarray) -> np.ndarray:
        """滑动平均滤波 - 平滑高频噪声"""
        if len(data) < self.ma_window:
            return data

        kernel = np.ones(self.ma_window) / self.ma_window
        return np.convolve(data, kernel, mode='same')

    def remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """异常值剔除 - 基于3σ原则"""
        if len(data) < 10:
            return data

        median = np.median(data)
        mad = np.median(np.abs(data - median))
        threshold = self.outlier_threshold * mad / 0.6745  # 转换为近似标准差

        cleaned = np.copy(data)
        outliers = np.abs(data - median) > threshold
        cleaned[outliers] = median

        return cleaned

    def calculate_stability_score(self, data: np.ndarray) -> float:
        """计算信号稳定性评分（0-1，1最稳定）"""
        if len(data) < 10:
            return 1.0

        # 基于变异系数
        mean_val = np.mean(data)
        if abs(mean_val) < 1e-10:
            return 1.0

        cv = np.std(data) / abs(mean_val)

        # 变异系数越小越稳定，映射到0-1
        score = max(0.0, min(1.0, 1.0 - cv * 10.0))
        return float(score)

    def filter(self, raw_value: float) -> Tuple[float, float]:
        """
        完整滤波流程 - 使用滚动窗口处理
        返回：(滤波后的值, 稳定性评分)
        """
        self._raw_history.append(raw_value)

        if len(self._raw_history) < self.median_window + self.ma_window:
            self._filtered_history.append(raw_value)
            return raw_value, 1.0

        # 使用最近的滚动窗口数据，而不是全部历史
        window_size = self.median_window + self.ma_window + 10
        data = np.array(self._raw_history)[-window_size:]

        # 先对趋势进行去趋势处理（差分），滤波后再还原
        if len(data) > 5:
            # 使用线性回归去除趋势
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)
            trend = slope * x + intercept
            detrended = data - trend

            # 1. 异常值剔除（对去趋势后的数据）
            cleaned = self.remove_outliers(detrended)
            # 2. 中值滤波
            median_filtered = self.median_filter(cleaned)
            # 3. 滑动平均
            ma_filtered = self.moving_average(median_filtered)

            # 恢复趋势
            filtered_data = ma_filtered + trend
            filtered_value = float(filtered_data[-1])
        else:
            # 数据不足，简单处理
            cleaned = self.remove_outliers(data)
            median_filtered = self.median_filter(cleaned)
            ma_filtered = self.moving_average(median_filtered)
            filtered_value = float(ma_filtered[-1])

        # 使用最近的窗口计算稳定性
        recent_data = np.array(self._raw_history)[-20:]
        stability_score = self.calculate_stability_score(recent_data)

        self._filtered_history.append(filtered_value)

        return filtered_value, stability_score
