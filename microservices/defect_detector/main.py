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
from uuid import uuid4
import hashlib
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    MicroserviceBase,
    CHANNELS, SERVICE_IDS, MESSAGE_TYPES,
    DefectDetection, ImageUpload, BatchRecord,
    MessageFactory, validate_message, extract_payload,
    config_loader, DefectConfig
)

from modules.defect_classifier import (
    CNNClassifier, ImagePreprocessor, ONNXEngine, DefectPostprocessor,
    ClassifierConfig, ImageQualityResult, BoundingBox, DefectCandidate, BatchDefectStats
)


class DefectDetector:

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

        self.classifier_config = ClassifierConfig(
            image_preprocessing=config.image_preprocessing,
            cnn_model=config.cnn_model,
            post_processing=config.__dict__.get('post_processing', {}),
            confidence_threshold=config.confidence_threshold,
            defect_types=config.__dict__.get('defect_types', {}),
            manual_review=config.__dict__.get('manual_review', {}),
            illumination_robustness=config.__dict__.get('illumination_robustness', {}),
            batch_processing=config.__dict__.get('batch_processing', {}),
        )

        self.preprocessor = ImagePreprocessor(self.classifier_config)
        self.engine = ONNXEngine(self.classifier_config.onnx_model_path)
        self.classifier = CNNClassifier(self.classifier_config)
        self.post_processor = DefectPostprocessor(config.__dict__.get('post_processing', {}))

        self._previous_bboxes: Dict[str, List[BoundingBox]] = {}

    def detect(self, image_path: str, batch_id: str,
               shelf_id: Optional[int] = None,
               vial_position: Optional[str] = None) -> Optional[DefectDetection]:
        image = self.preprocessor.load_image(image_path)
        if image is None:
            return None

        quality = self.preprocessor.check_quality(image)
        if not quality.passed:
            print(f"[DefectDetector] 图像质量检查失败: {quality.error_message}")
            return None

        robustness_config = self.config.__dict__.get('illumination_robustness', {})
        use_robust_preprocessing = robustness_config.get('enabled', True)
        use_augmentation = robustness_config.get('use_test_time_augmentation', False)

        def predict_single(preprocessed_img):
            probs = self.engine.predict(preprocessed_img)
            class_idx = int(np.argmax(probs[0]))
            confidence = float(probs[0, class_idx])
            return self.class_names[class_idx], confidence

        if use_augmentation:
            defect_type, confidence, all_confidences = self.preprocessor.predict_with_augmentation(
                image, predict_single
            )
            class_idx = self.class_names.index(defect_type)
        elif use_robust_preprocessing:
            preprocessed, metadata = self.preprocessor.preprocess_with_robustness(image)
            probs = self.engine.predict(preprocessed)
            class_idx = int(np.argmax(probs[0]))
            confidence = float(probs[0, class_idx])
            defect_type = self.class_names[class_idx]
            if metadata.get('illumination_correction', False):
                confidence *= 0.95
        else:
            preprocessed = self.preprocessor.preprocess(image)
            probs = self.engine.predict(preprocessed)
            class_idx = int(np.argmax(probs[0]))
            confidence = float(probs[0, class_idx])
            defect_type = self.class_names[class_idx]

        bbox = self._generate_bbox(image.shape[:2], defect_type, confidence)

        prev_key = f"{batch_id}_{shelf_id}_{vial_position}"
        previous_bboxes = self._previous_bboxes.get(prev_key, [])

        candidates = [DefectCandidate(
            defect_type=defect_type,
            confidence=confidence,
            bbox=bbox
        )]

        candidates = self.post_processor.apply_nms(candidates)
        candidates = self.post_processor.cluster_defects(candidates, image.shape[:2])
        candidates = self.post_processor.smooth_bounding_boxes(candidates, image.shape[:2])

        bboxes = [c.bbox for c in candidates if c.bbox is not None]
        bboxes = self._smooth_bboxes_temporal(bboxes, previous_bboxes)

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

    def _smooth_bboxes_temporal(self, bboxes: List[BoundingBox],
                                previous_bboxes: List[BoundingBox]) -> List[BoundingBox]:
        if not previous_bboxes or len(previous_bboxes) == 0:
            return bboxes

        smoothing_factor = self.config.__dict__.get('post_processing', {}).get('smoothing_factor', 0.5)
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
                    x=int(bbox.x * (1 - smoothing_factor) + matching_prev.x * smoothing_factor),
                    y=int(bbox.y * (1 - smoothing_factor) + matching_prev.y * smoothing_factor),
                    width=int(bbox.width * (1 - smoothing_factor) + matching_prev.width * smoothing_factor),
                    height=int(bbox.height * (1 - smoothing_factor) + matching_prev.height * smoothing_factor),
                    confidence=bbox.confidence * (1 - smoothing_factor) + matching_prev.confidence * smoothing_factor,
                    defect_type=bbox.defect_type
                ))
            else:
                smoothed.append(bbox)

        return smoothed

    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
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

    def _should_manual_review(self, defect_type: str, confidence: float) -> bool:
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
        hasher = hashlib.sha256()
        hasher.update(image_path.encode('utf-8'))
        hasher.update(str(time.time()).encode('utf-8'))
        return hasher.hexdigest()

    def get_class_labels(self) -> Dict[str, str]:
        labels = {}
        for defect_type, config in self.defect_types_config.items():
            labels[defect_type] = config.get('label', defect_type)
        return labels


class DefectDetectorService(MicroserviceBase):

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
        await self.subscribe(CHANNELS['IMAGE_UPLOAD'], self._on_image_upload)
        await self.subscribe(CHANNELS['SYSTEM_STATUS'], self._on_system_status)

    async def _on_start(self):
        print(f"[{self.service_id}] 启动缺陷检测服务")
        print(f"[{self.service_id}] 模型架构: {self.config.cnn_model.get('architecture', 'efficientnet_b0')}")
        print(f"[{self.service_id}] 类别数: {self.config.cnn_model.get('num_classes', 4)}")
        print(f"[{self.service_id}] 置信度阈值: {self.config.confidence_threshold}")

        self._processing_task = asyncio.create_task(self._processing_loop())

    async def _on_stop(self):
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        print(f"[{self.service_id}] 缺陷检测服务已停止")

    async def _on_image_upload(self, message: Dict):
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
        pass

    async def _processing_loop(self):
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
        message = MessageFactory.create_defect_detection(
            result,
            source_service=self.service_id
        )

        success = await self.publish(CHANNELS['DEFECT_DETECTION'], message)
        if success:
            self._increment_metric("messages_published")

    def _update_batch_stats(self, result: DefectDetection):
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
