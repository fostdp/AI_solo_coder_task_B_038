"""
冷阱除霜优化算法模块
包含多传感器融合、结霜厚度估算和除霜优化核心算法
"""

from .types import DeviceDefrostState
from .sensor_fusion import MultiSensorFusion
from .thickness_estimator import FrostThicknessEstimator
from .optimizer import DefrostOptimizer

__all__ = [
    'DeviceDefrostState',
    'MultiSensorFusion',
    'FrostThicknessEstimator',
    'DefrostOptimizer',
]

__version__ = '1.0.0'
