"""
CNN缺陷分类器
集成图像预处理、ONNX推理和后处理
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from .types import ClassifierConfig, DefectCandidate, BoundingBox, BatchDefectStats
from .preprocessor import ImagePreprocessor
from .onnx_engine import ONNXEngine


class CNNClassifier:
    """
    CNN缺陷分类器
    
    功能：
    1. 图像预处理（加载、质量检查、光照归一化）
    2. ONNX Runtime高速推理
    3. 测试时增强(TTA)
    4. 缺陷分类和置信度评估
    """
    
    DEFECT_TYPES = ['normal', 'collapse', 'shrinkage', 'cracking']
    
    def __init__(self, config: ClassifierConfig, model_path: Optional[str] = None, use_gpu: bool = False):
        self.config = config
        self.confidence_threshold = config.confidence_threshold
        self.defect_types = config.defect_types or {
            'normal': '正常',
            'collapse': '塌陷',
            'shrinkage': '萎缩',
            'cracking': '开裂'
        }
        
        self.preprocessor = ImagePreprocessor(config)
        
        onnx_path = model_path or config.onnx_model_path
        self.engine = ONNXEngine(onnx_path, use_gpu)
        
        self._use_tta = config.illumination_robustness.get('test_time_augmentation', {}).get('enabled', False)
        self._tta_num_views = config.illumination_robustness.get('test_time_augmentation', {}).get('num_views', 5)
    
    def classify_image(self, image: np.ndarray) -> List[DefectCandidate]:
        """
        对单张图像进行缺陷分类
        
        Args:
            image: 输入图像，shape: [height, width, channels]
            
        Returns:
            缺陷候选列表
        """
        quality = self.preprocessor.check_quality(image)
        if not quality.passed:
            return [DefectCandidate(
                defect_type='unknown',
                confidence=0.0,
                features=None
            )]
        
        if self._use_tta:
            return self._classify_with_tta(image)
        else:
            return self._classify_single(image)
    
    def _classify_single(self, image: np.ndarray) -> List[DefectCandidate]:
        """单张图像分类（无TTA）"""
        preprocessed = self.preprocessor.preprocess(image)

        probs = self.engine.predict(preprocessed)

        return self._probs_to_candidates(probs[0])
    
    def _classify_with_tta(self, image: np.ndarray) -> List[DefectCandidate]:
        """使用测试时增强(TTA)进行分类"""
        tta_config = {
            'enabled': True,
            'num_variants': self._tta_num_views,
            'brightness_range': [-0.15, 0.15],
            'contrast_range': [-0.1, 0.1],
            'noise_std': 3.0,
        }
        augmented = self.preprocessor.augment_image(image, tta_config)

        all_probs = []
        for aug_img in augmented:
            preprocessed = self.preprocessor.preprocess(aug_img)
            probs = self.engine.predict(preprocessed)
            all_probs.append(probs[0])

        avg_probs = np.mean(all_probs, axis=0)

        return self._probs_to_candidates(avg_probs)
    
    def _probs_to_candidates(self, probs: np.ndarray) -> List[DefectCandidate]:
        """将概率转换为缺陷候选"""
        candidates = []
        
        for i, prob in enumerate(probs):
            defect_type = self.DEFECT_TYPES[i] if i < len(self.DEFECT_TYPES) else f'class_{i}'
            
            if prob >= self.confidence_threshold or defect_type == 'normal':
                candidate = DefectCandidate(
                    defect_type=defect_type,
                    confidence=float(prob),
                    features=None
                )
                candidates.append(candidate)
        
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates
    
    def classify_batch(self, images: List[np.ndarray], batch_id: str = '') -> Tuple[List[List[DefectCandidate]], BatchDefectStats]:
        """
        批量分类图像
        
        Args:
            images: 图像列表
            batch_id: 批次ID
            
        Returns:
            (每个图像的缺陷候选列表, 批次统计信息)
        """
        all_candidates = []
        defect_counts = {dt: 0 for dt in self.DEFECT_TYPES}
        defect_images = 0
        
        for image in images:
            candidates = self.classify_image(image)
            all_candidates.append(candidates)
            
            if candidates:
                top = candidates[0]
                defect_counts[top.defect_type] = defect_counts.get(top.defect_type, 0) + 1
                
                if top.defect_type != 'normal' and top.confidence >= self.confidence_threshold:
                    defect_images += 1
        
        total_images = len(images)
        defect_rate = defect_images / total_images if total_images > 0 else 0.0
        
        review_config = self.config.manual_review
        needs_review = defect_rate > review_config.get('defect_rate_threshold', 0.1)
        
        stats = BatchDefectStats(
            batch_id=batch_id,
            total_images=total_images,
            defect_images=defect_images,
            defect_rate=defect_rate,
            defect_counts=defect_counts,
            quality_score=1.0 - defect_rate,
            needs_review=needs_review
        )
        
        return all_candidates, stats
    
    def classify_from_paths(self, image_paths: List[str], batch_id: str = '') -> Tuple[List[List[DefectCandidate]], BatchDefectStats]:
        """
        从文件路径批量分类
        
        Args:
            image_paths: 图像路径列表
            batch_id: 批次ID
            
        Returns:
            (每个图像的缺陷候选列表, 批次统计信息)
        """
        images = []
        for path in image_paths:
            img = self.preprocessor.load_image(path)
            if img is not None:
                images.append(img)
        
        return self.classify_batch(images, batch_id)
    
    def get_defect_type_name(self, defect_type: str) -> str:
        """获取缺陷类型的中文名称"""
        return self.defect_types.get(defect_type, defect_type)
    
    def get_inference_metrics(self) -> Dict:
        """获取推理性能指标"""
        return {
            'onnx_available': self.engine.is_available,
            'model_loaded': self.engine.is_loaded,
            'backend': 'onnxruntime' if self.engine.is_available else 'numpy'
        }
    
    def profile_inference(self, image_size: Tuple[int, int] = (224, 224), iterations: int = 10) -> Dict:
        """
        性能测试
        
        Args:
            image_size: 图像尺寸
            iterations: 迭代次数
            
        Returns:
            性能指标
        """
        x = np.random.randn(1, 3, image_size[0], image_size[1]).astype(np.float32)
        timing = self.engine.get_inference_time(x, iterations)
        
        return {
            **timing,
            'image_size': image_size,
            'iterations': iterations
        }
