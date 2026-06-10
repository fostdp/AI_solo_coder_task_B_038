"""
缺陷分类器模块
=============

提供基于CNN的冻干制品外观缺陷检测功能，包括：
- 光照归一化和域适应预处理
- ONNX Runtime高速推理
- 非极大值抑制(NMS)和缺陷聚类后处理

使用示例：
---------
```python
from modules.defect_classifier import (
    CNNClassifier,
    ImagePreprocessor,
    ONNXEngine,
    DefectPostprocessor,
    ClassifierConfig,
    DefectCandidate,
    BatchDefectStats
)

# 创建配置
config = ClassifierConfig(
    confidence_threshold=0.7,
    onnx_model_path='./models/defect_classifier.onnx',
    use_onnx=True
)

# 创建分类器
classifier = CNNClassifier(config, use_gpu=False)

# 分类图像
image = classifier.preprocessor.load_image('sample.jpg')
candidates = classifier.classify_image(image)

# 后处理
postprocessor = DefectPostprocessor()
processed = postprocessor.process(candidates, image.shape)

# 查看结果
for candidate in processed:
    print(f"{candidate.defect_type}: {candidate.confidence:.3f}")
```

模块结构：
---------
- types.py: 数据类型定义
- preprocessor.py: 图像预处理器（光照归一化、数据增强、域适应）
- onnx_engine.py: ONNX Runtime推理引擎
- classifier.py: CNN分类器主类
- postprocessor.py: 后处理器（NMS、边界框平滑、聚类）
"""

from .types import (
    ClassifierConfig,
    ImageQualityResult,
    BoundingBox,
    DefectCandidate,
    BatchDefectStats
)

from .preprocessor import ImagePreprocessor
from .onnx_engine import ONNXEngine
from .classifier import CNNClassifier
from .postprocessor import DefectPostprocessor

__all__ = [
    'ClassifierConfig',
    'ImageQualityResult',
    'BoundingBox',
    'DefectCandidate',
    'BatchDefectStats',
    'ImagePreprocessor',
    'ONNXEngine',
    'CNNClassifier',
    'DefectPostprocessor',
]

__version__ = '1.0.0'
