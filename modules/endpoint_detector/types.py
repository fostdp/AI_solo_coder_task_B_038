"""
数据类型定义
包含干燥终点检测算法所需的数据类
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Deque
from collections import deque
import numpy as np


@dataclass
class DeviceState:
    """设备状态 - 仅包含算法相关的字段"""
    device_id: int
    batch_id: Optional[str] = None
    current_phase: str = "idle"  # idle, freezing, primary_drying, secondary_drying, completed
    phase_start_time: Optional[float] = None

    # 数据缓存
    temp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    vacuum_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    timestamp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))

    # 滤波后的数据缓存
    filtered_temp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    filtered_vacuum_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))

    # 信号质量指标
    vacuum_stability_score: float = 1.0  # 0-1，1表示最稳定
    signal_quality_history: Deque[float] = field(default_factory=lambda: deque(maxlen=60))

    # 判定状态
    primary_endpoint_detected: bool = False
    secondary_endpoint_detected: bool = False
    primary_endpoint_time: Optional[float] = None
    secondary_endpoint_time: Optional[float] = None

    # 多级确认计数器
    primary_confirmation_count: int = 0
    secondary_confirmation_count: int = 0
    autoencoder_confirmation_count: int = 0
    prt_confirmation_count: int = 0

    # 自编码器训练数据
    training_data: List[np.ndarray] = field(default_factory=list)
    autoencoder_trained: bool = False


@dataclass
class PRTState:
    """压力升测试状态 - 独立的PRT状态管理"""
    in_progress: bool = False
    start_time: Optional[float] = None
    initial_pressure: Optional[float] = None
    measurements: List[Tuple[float, float]] = field(default_factory=list)
    last_test_time: float = 0.0


@dataclass
class PRTResult:
    """压力升测试结果"""
    test_start_time: float
    test_end_time: float
    initial_pressure_pa: float
    final_pressure_pa: float
    pressure_rise_pa_per_min: float
    test_duration_seconds: int
    is_endpoint_detected: bool
    detection_confidence: float
    data_quality_score: float
    test_status: str = "completed"
