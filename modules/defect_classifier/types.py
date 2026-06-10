"""
缺陷分类器数据类型
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class ImageQualityResult:
    """图像质量检查结果"""
    passed: bool
    brightness: float
    contrast: float
    blurriness: float
    resolution: Tuple[int, int]
    error_message: Optional[str] = None


@dataclass
class BoundingBox:
    """边界框"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    defect_type: str


@dataclass
class DefectCandidate:
    """缺陷候选"""
    defect_type: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    features: Optional[np.ndarray] = None


@dataclass
class BatchDefectStats:
    """批次缺陷统计"""
    batch_id: str
    total_images: int = 0
    defect_images: int = 0
    defect_rate: float = 0.0
    defect_counts: Dict[str, int] = field(default_factory=dict)
    quality_score: float = 1.0
    needs_review: bool = False


@dataclass
class ClassifierConfig:
    """分类器配置"""
    image_preprocessing: Dict = field(default_factory=dict)
    cnn_model: Dict = field(default_factory=dict)
    post_processing: Dict = field(default_factory=dict)
    confidence_threshold: float = 0.7
    defect_types: Dict = field(default_factory=dict)
    manual_review: Dict = field(default_factory=dict)
    illumination_robustness: Dict = field(default_factory=dict)
    batch_processing: Dict = field(default_factory=dict)
    onnx_model_path: Optional[str] = None
    use_onnx: bool = True
