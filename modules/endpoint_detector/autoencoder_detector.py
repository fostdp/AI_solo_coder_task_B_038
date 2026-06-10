"""
自编码器异常检测
通过学习正常干燥模式，检测模式变化来判定终点
"""

import numpy as np
from typing import Dict, List, Tuple, Deque
from collections import deque


class AutoEncoderDetector:
    """自编码器异常检测"""

    def __init__(self, config: Dict):
        self.input_dim = config.get('input_dim', 11)
        self.latent_dim = config.get('latent_dim', 4)
        self.hidden_layers = config.get('hidden_layers', [32, 16])
        self.threshold = config.get('threshold', 0.1)
        self.min_training_samples = config.get('min_training_samples', 100)
        self.confirmation_count = config.get('confirmation_count', 3)
        self.min_stability = config.get('min_stability_for_detection', 0.4)

        # 简化的自编码器（不依赖外部深度学习库，使用numpy实现）
        self._encoder_weights = None
        self._decoder_weights = None
        self._mean = None
        self._std = None
        self._trained = False
        self._recon_errors: Deque[float] = deque(maxlen=100)
        self._consecutive_anomalies = 0
        self._confirmed_endpoint = False

    def extract_features(self, temps: List[float], vacuums: List[float],
                        cold_trap: float, powers: List[float]) -> np.ndarray:
        """从遥测数据提取特征"""
        temp_arr = np.array(temps)
        vacuum_arr = np.array(vacuums)
        power_arr = np.array(powers)

        features = np.array([
            np.mean(temp_arr),       # 平均温度
            np.std(temp_arr),        # 温度标准差
            np.max(temp_arr) - np.min(temp_arr),  # 温度差
            np.mean(vacuum_arr),     # 平均真空度
            np.std(vacuum_arr),      # 真空度标准差
            cold_trap,               # 冷阱温度
            np.mean(power_arr),      # 平均功率
            np.std(power_arr),       # 功率标准差
            temp_arr[-1] - temp_arr[0] if len(temp_arr) > 1 else 0,  # 温度趋势
            vacuum_arr[-1] - vacuum_arr[0] if len(vacuum_arr) > 1 else 0,  # 真空趋势
            power_arr[-1] - power_arr[0] if len(power_arr) > 1 else 0,    # 功率趋势
        ])

        return features

    def train(self, training_data: List[np.ndarray]) -> None:
        """训练自编码器（简化的PCA-based自编码器）"""
        if len(training_data) < self.min_training_samples:
            return

        data_matrix = np.array(training_data)

        # 标准化
        self._mean = np.mean(data_matrix, axis=0)
        self._std = np.std(data_matrix, axis=0) + 1e-10
        normalized = (data_matrix - self._mean) / self._std

        # PCA降维作为编码器
        cov_matrix = np.cov(normalized.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 取前latent_dim个主成分
        idx = np.argsort(eigenvalues)[::-1]
        self._encoder_weights = eigenvectors[:, idx[:self.latent_dim]].real
        self._decoder_weights = self._encoder_weights.T

        # 计算重构误差阈值
        recon_errors = []
        for sample in normalized:
            encoded = sample @ self._encoder_weights
            decoded = encoded @ self._decoder_weights
            error = np.mean((sample - decoded) ** 2)
            recon_errors.append(error)

        # 阈值设为均值 + 3*标准差
        self.threshold = float(np.mean(recon_errors) + 3 * np.std(recon_errors))
        self._trained = True

    def predict(self, features: np.ndarray, stability_score: float = 1.0) -> Tuple[float, bool]:
        """预测重构误差"""
        if not self._trained:
            return 0.0, False

        # 信号质量过低时不进行判定
        if stability_score < self.min_stability:
            return 0.0, False

        # 标准化
        normalized = (features - self._mean) / self._std

        # 编码解码
        encoded = normalized @ self._encoder_weights
        decoded = encoded @ self._decoder_weights

        # 计算重构误差（MSE）
        recon_error = float(np.mean((normalized - decoded) ** 2))
        self._recon_errors.append(recon_error)

        # 根据信号质量动态调整阈值
        adjusted_threshold = self.threshold * (1.0 + (1.0 - stability_score) * 0.3)

        # 检测模式变化（误差显著增大表示终点到达）
        is_anomaly = recon_error > adjusted_threshold

        # 多级确认机制
        if is_anomaly:
            self._consecutive_anomalies += 1
        else:
            self._consecutive_anomalies = max(0, self._consecutive_anomalies - 1)

        if self._consecutive_anomalies >= self.confirmation_count:
            self._confirmed_endpoint = True
            return recon_error, True

        return recon_error, self._confirmed_endpoint

    def reset(self):
        """重置检测状态"""
        self._consecutive_anomalies = 0
        self._confirmed_endpoint = False
        self._trained = False
        self._encoder_weights = None
        self._decoder_weights = None
        self._mean = None
        self._std = None
        self._recon_errors.clear()
