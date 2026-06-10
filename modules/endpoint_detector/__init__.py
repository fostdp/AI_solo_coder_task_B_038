"""
干燥终点检测算法模块
包含四种核心算法：
1. SignalFilter - 信号滤波器
2. FirstDerivativeDetector - 一阶导数法终点检测
3. AutoEncoderDetector - 自编码器异常检测
4. PressureRiseTester - 压力升测试
"""

from .types import (
    DeviceState,
    PRTState,
    PRTResult,
)

from .signal_filter import SignalFilter
from .derivative_detector import FirstDerivativeDetector
from .autoencoder_detector import AutoEncoderDetector
from .pressure_rise_tester import PressureRiseTester

__all__ = [
    'DeviceState',
    'PRTState',
    'PRTResult',
    'SignalFilter',
    'FirstDerivativeDetector',
    'AutoEncoderDetector',
    'PressureRiseTester',
]

__version__ = '1.0.0'
