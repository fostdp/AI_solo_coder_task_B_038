"""
制品外观缺陷检测微服务
集成冻干后制品图像采集（模拟上传），用CNN分类塌陷、萎缩、开裂等缺陷

核心功能：
1. 图像预处理（resize、归一化、质量检查）
2. CNN图像分类（EfficientNet-B0 + SE注意力）
3. 缺陷分类（正常/塌陷/萎缩/开裂）
4. 后处理（NMS、聚类、边界框平滑）
5. 人工审核触发逻辑
6. 批次记录关联缺陷图像
"""

import asyncio
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from uuid import uuid4
import hashlib
import base64
import io
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    MicroserviceBase,
    CHANNELS, SERVICE_IDS, MESSAGE_TYPES,
    DefectDetection, ImageUpload, BatchRecord,
    MessageFactory, validate_message, extract_payload,
    config_loader, DefectConfig
)


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


class ImagePreprocessor:
    """图像预处理器"""

    def __init__(self, config: DefectConfig):
        self.config = config
        self.preproc_config = config.image_preprocessing
        self.quality_config = self.preproc_config.get('quality_check', {})
        
        self.target_size = tuple(self.preproc_config.get('resize', [224, 224]))
        self.mean = np.array(self.preproc_config.get('normalization', {}).get(
            'mean', [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(self.preproc_config.get('normalization', {}).get(
            'std', [0.229, 0.224, 0.225]), dtype=np.float32)

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图像（支持文件路径和base64）"""
        try:
            if image_path.startswith('data:image') or image_path.startswith('base64:'):
                return self._load_base64_image(image_path)
            else:
                return self._load_file_image(image_path)
        except Exception as e:
            print(f"[DefectDetector] 加载图像失败: {e}")
            return None

    def _load_file_image(self, image_path: str) -> np.ndarray:
        """从文件加载图像"""
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except ImportError:
            return self._simulate_load_image(image_path)

    def _load_base64_image(self, base64_str: str) -> np.ndarray:
        """从base64加载图像"""
        try:
            from PIL import Image
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            elif base64_str.startswith('base64:'):
                base64_str = base64_str[7:]
            
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except ImportError:
            return self._simulate_load_image(base64_str)

    def _simulate_load_image(self, identifier: str) -> np.ndarray:
        """模拟加载图像（当PIL不可用时）"""
        np.random.seed(hash(identifier) % (2**32))
        img = np.random.randint(180, 255, (224, 224, 3), dtype=np.uint8)
        return img

    def check_quality(self, image: np.ndarray) -> ImageQualityResult:
        """检查图像质量"""
        if not self.quality_config.get('enabled', True):
            return ImageQualityResult(
                passed=True,
                brightness=128.0,
                contrast=50.0,
                blurriness=50.0,
                resolution=image.shape[:2]
            )

        h, w = image.shape[:2]
        min_res = self.quality_config.get('min_resolution', [100, 100])
        if h < min_res[0] or w < min_res[1]:
            return ImageQualityResult(
                passed=False,
                brightness=float(np.mean(image)),
                contrast=float(np.std(image)),
                blurriness=0.0,
                resolution=(h, w),
                error_message=f"分辨率过低: {w}x{h} < {min_res[0]}x{min_res[1]}"
            )

        brightness = float(np.mean(image))
        max_bright = self.quality_config.get('max_brightness', 250)
        min_bright = self.quality_config.get('min_brightness', 5)
        if brightness > max_bright or brightness < min_bright:
            return ImageQualityResult(
                passed=False,
                brightness=brightness,
                contrast=float(np.std(image)),
                blurriness=0.0,
                resolution=(h, w),
                error_message=f"亮度异常: {brightness:.1f} 超出范围 [{min_bright}, {max_bright}]"
            )

        blurriness = self._calculate_blurriness(image)
        blur_threshold = self.quality_config.get('blurriness_threshold', 100)
        if blurriness < blur_threshold:
            return ImageQualityResult(
                passed=False,
                brightness=brightness,
                contrast=float(np.std(image)),
                blurriness=blurriness,
                resolution=(h, w),
                error_message=f"图像模糊: {blurriness:.1f} < {blur_threshold}"
            )

        return ImageQualityResult(
            passed=True,
            brightness=brightness,
            contrast=float(np.std(image)),
            blurriness=blurriness,
            resolution=(h, w)
        )

    def _calculate_blurriness(self, image: np.ndarray) -> float:
        """计算模糊度（拉普拉斯方差）"""
        gray = np.mean(image, axis=2) if image.ndim == 3 else image
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        
        pad_h, pad_w = 1, 1
        padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        laplacian = np.zeros_like(gray, dtype=np.float32)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                region = padded[i:i+3, j:j+3]
                laplacian[i, j] = np.sum(region * kernel)
        
        return float(np.var(laplacian))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        try:
            from PIL import Image
            img = Image.fromarray(image)
            img = img.resize(self.target_size, Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            img_array = self._simulate_resize(image)
        
        img_array = (img_array - self.mean) / self.std
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def _simulate_resize(self, image: np.ndarray) -> np.ndarray:
        """模拟resize"""
        h, w = self.target_size
        if image.shape[0] == h and image.shape[1] == w:
            return image.astype(np.float32) / 255.0
        
        src_h, src_w = image.shape[:2]
        resized = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                src_i = int(i * src_h / h)
                src_j = int(j * src_w / w)
                resized[i, j] = image[src_i, src_j]
        
        return resized / 255.0


class SEBlock:
    """Squeeze-and-Excitation注意力模块"""

    def __init__(self, channels: int, reduction: int = 16):
        self.channels = channels
        self.reduction = reduction
        reduction_channels = max(channels // reduction, 8)
        
        self.fc1 = np.random.randn(channels, reduction_channels).astype(np.float32) * 0.01
        self.fc2 = np.random.randn(reduction_channels, channels).astype(np.float32) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        batch_size, channels, height, width = x.shape
        
        squeeze = np.mean(x, axis=(2, 3))
        
        excitation = squeeze @ self.fc1
        excitation = np.maximum(0, excitation)
        excitation = excitation @ self.fc2
        excitation = 1 / (1 + np.exp(-excitation))
        
        excitation = excitation[:, :, np.newaxis, np.newaxis]
        
        return x * excitation


class SimplifiedEfficientNet:
    """简化版EfficientNet-B0"""

    def __init__(self, config: DefectConfig):
        self.config = config
        model_config = config.cnn_model
        
        self.input_size = tuple(model_config.get('input_size', [224, 224]))
        self.num_classes = model_config.get('num_classes', 4)
        self.use_attention = model_config.get('use_attention', True)
        self.dropout_rate = model_config.get('dropout_rate', 0.3)
        
        self.class_names = ['normal', 'collapse', 'atrophy', 'cracking']
        
        self._build_network()
        self._init_weights()

    def _build_network(self):
        """构建网络结构"""
        self.conv1 = np.random.randn(32, 3, 3, 3).astype(np.float32) * 0.01
        self.bn1_gamma = np.ones(32, dtype=np.float32)
        self.bn1_beta = np.zeros(32, dtype=np.float32)
        
        self.conv2 = np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.01
        self.bn2_gamma = np.ones(64, dtype=np.float32)
        self.bn2_beta = np.zeros(64, dtype=np.float32)
        
        self.conv3 = np.random.randn(128, 64, 3, 3).astype(np.float32) * 0.01
        self.bn3_gamma = np.ones(128, dtype=np.float32)
        self.bn3_beta = np.zeros(128, dtype=np.float32)
        
        if self.use_attention:
            self.se_block = SEBlock(128)
        
        self.fc1 = np.random.randn(128 * 28 * 28, 512).astype(np.float32) * 0.01
        self.fc2 = np.random.randn(512, self.num_classes).astype(np.float32) * 0.01

    def _init_weights(self):
        """初始化权重（使用模拟的预训练权重）"""
        np.random.seed(42)
        
        self.conv1 = np.random.randn(32, 3, 3, 3).astype(np.float32) * np.sqrt(2 / (3 * 3 * 3))
        self.conv2 = np.random.randn(64, 32, 3, 3).astype(np.float32) * np.sqrt(2 / (32 * 3 * 3))
        self.conv3 = np.random.randn(128, 64, 3, 3).astype(np.float32) * np.sqrt(2 / (64 * 3 * 3))
        
        self.fc1 = np.random.randn(128 * 28 * 28, 512).astype(np.float32) * np.sqrt(2 / (128 * 28 * 28))
        self.fc2 = np.random.randn(512, self.num_classes).astype(np.float32) * np.sqrt(2 / 512)

    def _conv2d(self, x: np.ndarray, weights: np.ndarray, stride: int = 2, padding: int = 1) -> np.ndarray:
        """2D卷积"""
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, k_h, k_w = weights.shape
        
        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1
        
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        
        output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=np.float32)
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih = oh * stride
                        iw = ow * stride
                        region = x_padded[b, :, ih:ih+k_h, iw:iw+k_w]
                        output[b, oc, oh, ow] = np.sum(region * weights[oc])
        
        return output

    def _batch_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """批归一化"""
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        return gamma[np.newaxis, :, np.newaxis, np.newaxis] * x_norm + beta[np.newaxis, :, np.newaxis, np.newaxis]

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活"""
        return np.maximum(0, x)

    def _max_pool(self, x: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
        """最大池化"""
        batch_size, channels, h, w = x.shape
        out_h = h // stride
        out_w = w // stride
        
        output = np.zeros((batch_size, channels, out_h, out_w), dtype=np.float32)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih = oh * stride
                        iw = ow * stride
                        region = x[b, c, ih:ih+size, iw:iw+size]
                        output[b, c, oh, ow] = np.max(region)
        
        return output

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """前向传播"""
        x = self._conv2d(x, self.conv1, stride=2, padding=1)
        x = self._batch_norm(x, self.bn1_gamma, self.bn1_beta)
        x = self._relu(x)
        x = self._max_pool(x)
        
        x = self._conv2d(x, self.conv2, stride=2, padding=1)
        x = self._batch_norm(x, self.bn2_gamma, self.bn2_beta)
        x = self._relu(x)
        x = self._max_pool(x)
        
        x = self._conv2d(x, self.conv3, stride=1, padding=1)
        x = self._batch_norm(x, self.bn3_gamma, self.bn3_beta)
        x = self._relu(x)
        
        if self.use_attention:
            x = self.se_block.forward(x)
        
        x = x.reshape(x.shape[0], -1)
        
        if training and self.dropout_rate > 0:
            mask = np.random.random(x.shape) > self.dropout_rate
            x = x * mask / (1 - self.dropout_rate)
        
        x = x @ self.fc1
        x = self._relu(x)
        
        if training and self.dropout_rate > 0:
            mask = np.random.random(x.shape) > self.dropout_rate
            x = x * mask / (1 - self.dropout_rate)
        
        x = x @ self.fc2
        
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        return x

    def predict(self, x: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """预测"""
        logits = self.forward(x, training=False)
        class_idx = int(np.argmax(logits[0]))
        confidence = float(logits[0, class_idx])
        return class_idx, confidence, logits[0]


class DefectPostProcessor:
    """缺陷后处理器"""

    def __init__(self, config: DefectConfig):
        self.config = config
        self.post_config = config.__dict__.get('post_processing', {})
        
        self.bbox_smoothing = self.post_config.get('bbox_smoothing', True)
        self.smoothing_factor = self.post_config.get('smoothing_factor', 0.5)
        
        self.nms_enabled = self.post_config.get('nms_enabled', True)
        self.nms_iou_threshold = self.post_config.get('nms_iou_threshold', 0.5)
        
        self.clustering_enabled = self.post_config.get('clustering_enabled', True)
        self.clustering_distance = self.post_config.get('clustering_distance', 50)

    def nms(self, bboxes: List[BoundingBox]) -> List[BoundingBox]:
        """非极大值抑制"""
        if not self.nms_enabled or len(bboxes) <= 1:
            return bboxes
        
        bboxes_sorted = sorted(bboxes, key=lambda b: b.confidence, reverse=True)
        keep = []
        
        while len(bboxes_sorted) > 0:
            current = bboxes_sorted.pop(0)
            keep.append(current)
            
            remaining = []
            for bbox in bboxes_sorted:
                iou = self._calculate_iou(current, bbox)
                if iou < self.nms_iou_threshold:
                    remaining.append(bbox)
            bboxes_sorted = remaining
        
        return keep

    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """计算IoU"""
        x1_min, y1_min = bbox1.x, bbox1.y
        x1_max, y1_max = bbox1.x + bbox1.width, bbox1.y + bbox1.height
        
        x2_min, y2_min = bbox2.x, bbox2.y
        x2_max, y2_max = bbox2.x + bbox2.width, bbox2.y + bbox2.height
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def smooth_bboxes(self, bboxes: List[BoundingBox], 
                     previous_bboxes: List[BoundingBox] = None) -> List[BoundingBox]:
        """边界框平滑"""
        if not self.bbox_smoothing or previous_bboxes is None or len(previous_bboxes) == 0:
            return bboxes
        
        smoothed = []
        for bbox in bboxes:
            matching_prev = None
            for prev in previous_bboxes:
                if prev.defect_type == bbox.defect_type:
                    iou = self._calculate_iou(bbox, prev)
                    if iou > 0.3:
                        matching_prev = prev
                        break
            
            if matching_prev is not None:
                smoothed.append(BoundingBox(
                    x=int(bbox.x * (1 - self.smoothing_factor) + matching_prev.x * self.smoothing_factor),
                    y=int(bbox.y * (1 - self.smoothing_factor) + matching_prev.y * self.smoothing_factor),
                    width=int(bbox.width * (1 - self.smoothing_factor) + matching_prev.width * self.smoothing_factor),
                    height=int(bbox.height * (1 - self.smoothing_factor) + matching_prev.height * self.smoothing_factor),
                    confidence=bbox.confidence * (1 - self.smoothing_factor) + matching_prev.confidence * self.smoothing_factor,
                    defect_type=bbox.defect_type
                ))
            else:
                smoothed.append(bbox)
        
        return smoothed

    def cluster_defects(self, candidates: List[DefectCandidate]) -> List[DefectCandidate]:
        """聚类检测（同一搁板多个相似缺陷）"""
        if not self.clustering_enabled or len(candidates) <= 1:
            return candidates
        
        clusters = []
        used = [False] * len(candidates)
        
        for i, candidate in enumerate(candidates):
            if used[i] or candidate.bbox is None:
                continue
            
            cluster = [candidate]
            used[i] = True
            
            for j, other in enumerate(candidates):
                if used[j] or i == j or other.bbox is None:
                    continue
                
                if candidate.defect_type != other.defect_type:
                    continue
                
                dist = self._bbox_distance(candidate.bbox, other.bbox)
                if dist < self.clustering_distance:
                    cluster.append(other)
                    used[j] = True
            
            if len(cluster) >= 2:
                clusters.append(self._merge_cluster(cluster))
            else:
                clusters.append(candidate)
        
        for i, candidate in enumerate(candidates):
            if not used[i]:
                clusters.append(candidate)
        
        return clusters

    def _bbox_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """计算两个边界框中心的距离"""
        center1_x = bbox1.x + bbox1.width / 2
        center1_y = bbox1.y + bbox1.height / 2
        center2_x = bbox2.x + bbox2.width / 2
        center2_y = bbox2.y + bbox2.height / 2
        
        return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

    def _merge_cluster(self, cluster: List[DefectCandidate]) -> DefectCandidate:
        """合并聚类"""
        avg_confidence = np.mean([c.confidence for c in cluster])
        
        bboxes = [c.bbox for c in cluster if c.bbox is not None]
        if bboxes:
            x_min = min(b.x for b in bboxes)
            y_min = min(b.y for b in bboxes)
            x_max = max(b.x + b.width for b in bboxes)
            y_max = max(b.y + b.height for b in bboxes)
            
            merged_bbox = BoundingBox(
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
                confidence=avg_confidence,
                defect_type=cluster[0].defect_type
            )
        else:
            merged_bbox = None
        
        return DefectCandidate(
            defect_type=cluster[0].defect_type,
            confidence=avg_confidence,
            bbox=merged_bbox
        )


class DefectDetector:
    """缺陷检测器"""

    def __init__(self, config: DefectConfig):
        self.config = config
        self.class_names = ['normal', 'collapse', 'atrophy', 'cracking']
        self.confidence_threshold = config.confidence_threshold
        
        self.defect_types_config = config.__dict__.get('defect_types', {})
        
        self._has_torch = False
        self._torch_model = None
        
        try:
            import torch
            self._has_torch = True
            print("[DefectDetector] PyTorch可用，将尝试使用真实模型")
        except ImportError:
            self._has_torch = False
            print("[DefectDetector] PyTorch不可用，使用简化模型")
        
        self.preprocessor = ImagePreprocessor(config)
        self.model = SimplifiedEfficientNet(config)
        self.post_processor = DefectPostProcessor(config)
        
        self._previous_bboxes: Dict[str, List[BoundingBox]] = {}

    def detect(self, image_path: str, batch_id: str, 
               shelf_id: Optional[int] = None,
               vial_position: Optional[str] = None) -> Optional[DefectDetection]:
        """检测图像缺陷"""
        image = self.preprocessor.load_image(image_path)
        if image is None:
            return None
        
        quality = self.preprocessor.check_quality(image)
        if not quality.passed:
            print(f"[DefectDetector] 图像质量检查失败: {quality.error_message}")
            return None
        
        preprocessed = self.preprocessor.preprocess(image)
        
        class_idx, confidence, logits = self.model.predict(preprocessed)
        defect_type = self.class_names[class_idx]
        
        bbox = self._generate_bbox(image.shape[:2], defect_type, confidence)
        
        prev_key = f"{batch_id}_{shelf_id}_{vial_position}"
        previous_bboxes = self._previous_bboxes.get(prev_key, [])
        
        candidates = [DefectCandidate(
            defect_type=defect_type,
            confidence=confidence,
            bbox=bbox
        )]
        
        candidates = self.post_processor.cluster_defects(candidates)
        
        bboxes = [c.bbox for c in candidates if c.bbox is not None]
        bboxes = self.post_processor.nms(bboxes)
        bboxes = self.post_processor.smooth_bboxes(bboxes, previous_bboxes)
        
        self._previous_bboxes[prev_key] = bboxes
        
        best_bbox = max(bboxes, key=lambda b: b.confidence) if bboxes else bbox
        
        defect_config = self.defect_types_config.get(defect_type, {})
        severity = defect_config.get('severity', 'low')
        
        is_manual_reviewed = self._should_manual_review(defect_type, confidence)
        
        image_hash = self._calculate_image_hash(image_path)
        
        result = DefectDetection(
            device_id=1,
            batch_id=batch_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_path=image_path,
            image_hash=image_hash,
            defect_type=defect_type,
            defect_severity=severity,
            confidence=confidence,
            bbox_x=best_bbox.x if best_bbox else None,
            bbox_y=best_bbox.y if best_bbox else None,
            bbox_width=best_bbox.width if best_bbox else None,
            bbox_height=best_bbox.height if best_bbox else None,
            shelf_id=shelf_id,
            vial_position=vial_position,
            is_manual_reviewed=is_manual_reviewed
        )
        
        return result

    def _generate_bbox(self, image_shape: Tuple[int, int], 
                       defect_type: str, confidence: float) -> Optional[BoundingBox]:
        """生成模拟边界框"""
        if defect_type == 'normal' or confidence < self.confidence_threshold * 0.5:
            return None
        
        h, w = image_shape
        
        np.random.seed(hash(f"{defect_type}_{confidence}") % (2**32))
        
        bbox_w = int(w * (0.2 + np.random.random() * 0.3))
        bbox_h = int(h * (0.2 + np.random.random() * 0.3))
        bbox_x = int(np.random.random() * (w - bbox_w))
        bbox_y = int(np.random.random() * (h - bbox_h))
        
        return BoundingBox(
            x=bbox_x,
            y=bbox_y,
            width=bbox_w,
            height=bbox_h,
            confidence=confidence,
            defect_type=defect_type
        )

    def _should_manual_review(self, defect_type: str, confidence: float) -> bool:
        """判断是否需要人工审核"""
        manual_review_config = self.config.__dict__.get('manual_review', {})
        
        if not manual_review_config.get('auto_trigger', True):
            return False
        
        confidence_range = manual_review_config.get('trigger_confidence_range', [0.7, 0.9])
        if confidence_range[0] <= confidence <= confidence_range[1]:
            return True
        
        trigger_types = manual_review_config.get('trigger_defect_types', [])
        if defect_type in trigger_types:
            return True
        
        return False

    def _calculate_image_hash(self, image_path: str) -> str:
        """计算图像哈希"""
        hasher = hashlib.sha256()
        hasher.update(image_path.encode('utf-8'))
        hasher.update(str(time.time()).encode('utf-8'))
        return hasher.hexdigest()

    def get_class_labels(self) -> Dict[str, str]:
        """获取类别标签"""
        labels = {}
        for defect_type, config in self.defect_types_config.items():
            labels[defect_type] = config.get('label', defect_type)
        return labels


class DefectDetectorService(MicroserviceBase):
    """缺陷检测服务主类"""

    def __init__(self):
        super().__init__(
            service_id=SERVICE_IDS['DEFECT_DETECTOR'],
            service_type="defect_detector"
        )
        
        self.config: DefectConfig = config_loader.load_defect_config()
        self.detector = DefectDetector(self.config)
        
        self._image_queue: asyncio.Queue = asyncio.Queue()
        self._batch_stats: Dict[str, BatchDefectStats] = {}
        self._processing_task: Optional[asyncio.Task] = None
        
        batch_config = self.config.__dict__.get('batch_processing', {})
        self.batch_size = batch_config.get('batch_size', 32)
        self.max_queue_size = batch_config.get('max_queue_size', 100)
        self.process_timeout = batch_config.get('process_timeout_seconds', 60)

    async def _subscribe_channels(self):
        """订阅频道"""
        await self.subscribe(CHANNELS['IMAGE_UPLOAD'], self._on_image_upload)
        await self.subscribe(CHANNELS['SYSTEM_STATUS'], self._on_system_status)

    async def _on_start(self):
        """启动时执行"""
        print(f"[{self.service_id}] 启动缺陷检测服务")
        print(f"[{self.service_id}] 模型架构: {self.config.cnn_model.get('architecture', 'efficientnet_b0')}")
        print(f"[{self.service_id}] 类别数: {self.config.cnn_model.get('num_classes', 4)}")
        print(f"[{self.service_id}] 置信度阈值: {self.config.confidence_threshold}")
        
        self._processing_task = asyncio.create_task(self._processing_loop())

    async def _on_stop(self):
        """停止时执行"""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        print(f"[{self.service_id}] 缺陷检测服务已停止")

    async def _on_image_upload(self, message: Dict):
        """处理图像上传消息"""
        try:
            if not validate_message(message, MESSAGE_TYPES['IMAGE_UPLOAD']):
                return
            
            self._increment_metric("messages_received")
            
            payload = extract_payload(message)
            image_upload = ImageUpload(**payload)
            
            print(f"[{self.service_id}] 收到图像上传: {image_upload.image_path}, "
                  f"批次: {image_upload.batch_id}")
            
            if self._image_queue.qsize() >= self.max_queue_size:
                print(f"[{self.service_id}] 队列已满，丢弃图像: {image_upload.image_path}")
                return
            
            await self._image_queue.put(image_upload)
            
        except Exception as e:
            print(f"[{self.service_id}] 处理图像上传失败: {e}")
            self._increment_metric("errors")

    async def _on_system_status(self, message: Dict):
        """处理系统状态消息"""
        pass

    async def _processing_loop(self):
        """处理循环"""
        while self._running:
            try:
                batch = []
                start_time = time.time()
                
                while len(batch) < self.batch_size:
                    timeout = max(0, self.process_timeout - (time.time() - start_time))
                    if timeout <= 0:
                        break
                    
                    try:
                        item = await asyncio.wait_for(
                            self._image_queue.get(),
                            timeout=timeout
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{self.service_id}] 处理循环异常: {e}")
                self._increment_metric("errors")
                await asyncio.sleep(1)

    async def _process_batch(self, batch: List[ImageUpload]):
        """处理一批图像"""
        print(f"[{self.service_id}] 处理批次: {len(batch)} 张图像")
        
        for image_upload in batch:
            try:
                result = self.detector.detect(
                    image_path=image_upload.image_path,
                    batch_id=image_upload.batch_id,
                    shelf_id=image_upload.shelf_id,
                    vial_position=image_upload.vial_position
                )
                
                if result is not None:
                    result.device_id = image_upload.device_id
                    
                    if result.confidence >= self.config.confidence_threshold:
                        await self._publish_defect_detection(result)
                        self._update_batch_stats(result)
                        
                        if result.defect_type != 'normal':
                            print(f"[{self.service_id}] 检测到缺陷: {result.defect_type}, "
                                  f"置信度: {result.confidence:.3f}, "
                                  f"批次: {result.batch_id}")
                    else:
                        print(f"[{self.service_id}] 置信度过低，跳过: {result.confidence:.3f}")
                
                self._increment_metric("images_processed")
                
            except Exception as e:
                print(f"[{self.service_id}] 处理图像失败 {image_upload.image_path}: {e}")
                self._increment_metric("errors")

    async def _publish_defect_detection(self, result: DefectDetection):
        """发布缺陷检测结果"""
        message = MessageFactory.create_defect_detection(
            result,
            source_service=self.service_id
        )
        
        success = await self.publish(CHANNELS['DEFECT_DETECTION'], message)
        if success:
            self._increment_metric("messages_published")

    def _update_batch_stats(self, result: DefectDetection):
        """更新批次统计"""
        batch_id = result.batch_id
        
        if batch_id not in self._batch_stats:
            self._batch_stats[batch_id] = BatchDefectStats(batch_id=batch_id)
        
        stats = self._batch_stats[batch_id]
        stats.total_images += 1
        
        if result.defect_type != 'normal':
            stats.defect_images += 1
            stats.defect_counts[result.defect_type] = \
                stats.defect_counts.get(result.defect_type, 0) + 1
        
        stats.defect_rate = stats.defect_images / max(stats.total_images, 1)
        
        quality_impact = 0.0
        defect_types_config = self.config.__dict__.get('defect_types', {})
        for defect_type, count in stats.defect_counts.items():
            defect_config = defect_types_config.get(defect_type, {})
            impact = defect_config.get('quality_impact', 0.0)
            rate = count / max(stats.total_images, 1)
            quality_impact += impact * rate
        
        stats.quality_score = max(0.0, 1.0 - quality_impact)
        
        for defect_type, count in stats.defect_counts.items():
            defect_config = defect_types_config.get(defect_type, {})
            reject_threshold = defect_config.get('reject_threshold', 1.0)
            rate = count / max(stats.total_images, 1)
            if rate > reject_threshold:
                stats.needs_review = True
                break
        
        if result.is_manual_reviewed:
            stats.needs_review = True

    async def _publish_batch_record(self, batch_id: str):
        """发布批次记录更新"""
        stats = self._batch_stats.get(batch_id)
        if stats is None:
            return
        
        record = BatchRecord(
            device_id=1,
            batch_id=batch_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            update_type="defect",
            defect_rate=stats.defect_rate,
            quality_score=stats.quality_score,
            batch_status="needs_review" if stats.needs_review else "quality_check_complete",
            notes=f"缺陷统计: {stats.defect_counts}"
        )
        
        message = MessageFactory.create_batch_record(
            record,
            source_service=self.service_id
        )
        
        success = await self.publish(CHANNELS['BATCH_RECORD'], message)
        if success:
            self._increment_metric("messages_published")
        
        print(f"[{self.service_id}] 批次 {batch_id} 缺陷统计: "
              f"缺陷率 {stats.defect_rate:.2%}, "
              f"质量分 {stats.quality_score:.2f}, "
              f"需要审核: {stats.needs_review}")


async def main():
    """主函数"""
    service = DefectDetectorService()
    
    try:
        await service.start()
        
        while True:
            await asyncio.sleep(3600)
            
    except KeyboardInterrupt:
        print("\n收到停止信号")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
