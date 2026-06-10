"""
制品外观缺陷检测测试
覆盖：CNN分类准确率、缺陷与冻干曲线的关联性、批次追溯的完整性
场景：正常、边界、异常
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "microservices"))

import pytest
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from defect_detector.main import (
    ImagePreprocessor,
    SEBlock,
    SimplifiedEfficientNet,
    DefectPostProcessor,
    DefectDetector,
    BoundingBox,
    DefectCandidate,
    BatchDefectStats,
    ImageQualityResult,
)
from shared import DefectConfig


class TestImagePreprocessor:
    """图像预处理器测试"""

    @pytest.fixture
    def config(self):
        return DefectConfig(
            enabled=True,
            cnn_model={
                'input_size': [224, 224],
                'num_classes': 4,
                'use_attention': True,
            },
            image_preprocessing={
                'resize': [224, 224],
                'normalization': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                },
                'quality_check': {
                    'enabled': True,
                    'min_resolution': [100, 100],
                    'min_brightness': 5,
                    'max_brightness': 250,
                    'blurriness_threshold': 100,
                },
            },
            defect_types={
                'classes': ['normal', 'collapse', 'atrophy', 'cracking'],
            },
            confidence_threshold=0.8,
        )

    @pytest.fixture
    def preprocessor(self, config):
        return ImagePreprocessor(config)

    def test_load_image_normal(self, preprocessor):
        """正常场景：图像加载（模拟）"""
        img = preprocessor._simulate_load_image("test_image_001")
        
        assert img is not None
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8
        assert np.all(img >= 0) and np.all(img <= 255)

    def test_quality_check_normal(self, preprocessor):
        """正常场景：正常图像质量检查通过"""
        np.random.seed(42)
        img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img[50:150, 50:150] = 255
        
        result = preprocessor.check_quality(img)
        
        assert result.passed == True
        assert 5 <= result.brightness <= 250
        assert result.resolution == (224, 224)

    def test_quality_check_low_resolution(self, preprocessor):
        """异常场景：低分辨率图像拒绝"""
        img = np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8)
        
        result = preprocessor.check_quality(img)
        
        assert result.passed == False
        assert '分辨率过低' in result.error_message

    def test_quality_check_too_bright(self, preprocessor):
        """边界场景：过亮图像拒绝"""
        img = np.ones((224, 224, 3), dtype=np.uint8) * 254
        
        result = preprocessor.check_quality(img)
        
        assert result.passed == False
        assert '亮度异常' in result.error_message

    def test_quality_check_too_dark(self, preprocessor):
        """边界场景：过暗图像拒绝"""
        img = np.ones((224, 224, 3), dtype=np.uint8) * 3
        
        result = preprocessor.check_quality(img)
        
        assert result.passed == False
        assert '亮度异常' in result.error_message

    def test_quality_check_blurry(self, preprocessor):
        """边界场景：模糊图像拒绝"""
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        result = preprocessor.check_quality(img)
        
        assert result.passed == False
        assert '图像模糊' in result.error_message

    def test_blurriness_calculation(self, preprocessor):
        """正常场景：模糊度计算准确性"""
        sharp_img = np.zeros((224, 224, 3), dtype=np.uint8)
        sharp_img[::2, :] = 255
        
        blurry_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        sharp_blur = preprocessor._calculate_blurriness(sharp_img)
        blurry_blur = preprocessor._calculate_blurriness(blurry_img)
        
        assert sharp_blur > blurry_blur
        assert sharp_blur > 100

    def test_preprocess_normal(self, preprocessor):
        """正常场景：图像预处理流程"""
        img = preprocessor._simulate_load_image("test_001")
        
        preprocessed = preprocessor.preprocess(img)
        
        assert preprocessed.shape == (1, 3, 224, 224)
        assert preprocessed.dtype == np.float32
        
        mean_val = np.mean(preprocessed)
        assert -3 < mean_val < 3

    def test_preprocess_different_sizes(self, preprocessor):
        """边界场景：不同尺寸图像的resize"""
        for size in [(100, 100), (300, 300), (500, 200)]:
            img = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
            
            preprocessed = preprocessor.preprocess(img)
            
            assert preprocessed.shape == (1, 3, 224, 224)

    def test_quality_check_disabled(self, config):
        """边界场景：质量检查禁用时全部通过"""
        config.image_preprocessing['quality_check']['enabled'] = False
        preprocessor = ImagePreprocessor(config)
        
        img = np.ones((50, 50, 3), dtype=np.uint8) * 3
        result = preprocessor.check_quality(img)
        
        assert result.passed == True


class TestSEBlock:
    """SE注意力模块测试"""

    def test_se_block_forward(self):
        """正常场景：SE模块前向传播"""
        se_block = SEBlock(channels=128, reduction=16)
        
        x = np.random.randn(2, 128, 28, 28).astype(np.float32)
        
        output = se_block.forward(x)
        
        assert output.shape == x.shape
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_se_block_attention_weights(self):
        """正常场景：注意力权重在0-1之间"""
        se_block = SEBlock(channels=64, reduction=8)
        
        x = np.random.randn(1, 64, 14, 14).astype(np.float32)
        
        batch_size, channels, height, width = x.shape
        squeeze = np.mean(x, axis=(2, 3))
        excitation = squeeze @ se_block.fc1
        excitation = np.maximum(0, excitation)
        excitation = excitation @ se_block.fc2
        excitation = 1 / (1 + np.exp(-excitation))
        
        assert np.all(excitation >= 0) and np.all(excitation <= 1)
        assert excitation.shape == (1, 64)


class TestSimplifiedEfficientNet:
    """简化版EfficientNet测试"""

    @pytest.fixture
    def config(self):
        return DefectConfig(
            enabled=True,
            cnn_model={
                'input_size': [224, 224],
                'num_classes': 4,
                'use_attention': True,
                'dropout_rate': 0.3,
            },
            image_preprocessing={
                'resize': [224, 224],
                'normalization': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                },
                'quality_check': {'enabled': False},
            },
            defect_types={
                'classes': ['normal', 'collapse', 'atrophy', 'cracking'],
            },
            confidence_threshold=0.8,
        )

    @pytest.fixture
    def model(self, config):
        return SimplifiedEfficientNet(config)

    def test_model_initialization(self, model):
        """正常场景：模型初始化"""
        assert model.num_classes == 4
        assert len(model.class_names) == 4
        assert model.class_names == ['normal', 'collapse', 'atrophy', 'cracking']

    def test_forward_pass_normal(self, model):
        """正常场景：模型前向传播"""
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        output = model.forward(x, training=False)
        
        assert output.shape == (1, 4)
        assert np.all(output >= 0) and np.all(output <= 1)
        assert abs(np.sum(output) - 1.0) < 0.001

    def test_predict_normal(self, model):
        """正常场景：预测输出"""
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        class_idx, confidence, logits = model.predict(x)
        
        assert 0 <= class_idx < 4
        assert 0 <= confidence <= 1
        assert logits.shape == (4,)
        assert abs(np.sum(logits) - 1.0) < 0.001

    def test_forward_pass_batch(self, model):
        """边界场景：批量前向传播"""
        x = np.random.randn(4, 3, 224, 224).astype(np.float32)
        
        output = model.forward(x, training=False)
        
        assert output.shape == (4, 4)

    def test_training_mode_dropout(self, model):
        """边界场景：训练模式下的dropout"""
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        outputs_train = []
        for _ in range(5):
            outputs_train.append(model.forward(x, training=True))
        
        outputs_eval = []
        for _ in range(5):
            outputs_eval.append(model.forward(x, training=False))
        
        train_var = np.var([o.flatten() for o in outputs_train])
        eval_var = np.var([o.flatten() for o in outputs_eval])
        
        assert train_var >= eval_var

    def test_conv2d_correctness(self, model):
        """正常场景：卷积操作正确性"""
        x = np.ones((1, 3, 224, 224), dtype=np.float32)
        
        output = model._conv2d(x, model.conv1, stride=2, padding=1)
        
        assert output.shape == (1, 32, 112, 112)

    def test_batch_norm(self, model):
        """正常场景：批归一化"""
        x = np.random.randn(2, 32, 112, 112).astype(np.float32)
        
        output = model._batch_norm(x, model.bn1_gamma, model.bn1_beta)
        
        assert output.shape == x.shape
        assert abs(np.mean(output)) < 0.1

    def test_classification_accuracy(self, model):
        """正常场景：分类准确率测试（基于模拟特征）"""
        np.random.seed(42)
        
        test_cases = []
        for class_idx in range(4):
            for _ in range(20):
                x = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.1
                x[:, :, class_idx*20:class_idx*20+50, :] += 2.0
                test_cases.append((x, class_idx))
        
        correct = 0
        for x, true_label in test_cases:
            pred_idx, _, _ = model.predict(x)
            if pred_idx == true_label:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy > 0.5

    def test_attention_enabled_vs_disabled(self, config):
        """边界场景：注意力模块对性能的影响"""
        config.cnn_model['use_attention'] = True
        model_with_attention = SimplifiedEfficientNet(config)
        
        config.cnn_model['use_attention'] = False
        model_without_attention = SimplifiedEfficientNet(config)
        
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        out_with = model_with_attention.forward(x)
        out_without = model_without_attention.forward(x)
        
        assert out_with.shape == out_without.shape
        assert not np.allclose(out_with, out_without, atol=0.1)


class TestDefectPostProcessor:
    """缺陷后处理器测试"""

    @pytest.fixture
    def config(self):
        cfg = DefectConfig(
            enabled=True,
            cnn_model={'input_size': [224, 224], 'num_classes': 4},
            image_preprocessing={'resize': [224, 224], 'quality_check': {'enabled': False}},
            defect_types={'classes': ['normal', 'collapse', 'atrophy', 'cracking']},
            confidence_threshold=0.8,
        )
        cfg.__dict__['post_processing'] = {
            'bbox_smoothing': True,
            'smoothing_factor': 0.5,
            'nms_enabled': True,
            'nms_iou_threshold': 0.5,
            'clustering_enabled': True,
            'clustering_distance': 50,
        }
        return cfg

    @pytest.fixture
    def postprocessor(self, config):
        return DefectPostProcessor(config)

    def test_nms_normal(self, postprocessor):
        """正常场景：NMS去除重叠边界框"""
        bboxes = [
            BoundingBox(x=50, y=50, width=100, height=100, confidence=0.95, defect_type='collapse'),
            BoundingBox(x=55, y=55, width=100, height=100, confidence=0.90, defect_type='collapse'),
            BoundingBox(x=200, y=50, width=100, height=100, confidence=0.85, defect_type='atrophy'),
        ]
        
        result = postprocessor.nms(bboxes)
        
        assert len(result) == 2
        assert result[0].confidence == 0.95
        assert result[1].defect_type == 'atrophy'

    def test_nms_disabled(self, postprocessor):
        """边界场景：NMS禁用时保留所有框"""
        postprocessor.nms_enabled = False
        
        bboxes = [
            BoundingBox(x=50, y=50, width=100, height=100, confidence=0.95, defect_type='collapse'),
            BoundingBox(x=55, y=55, width=100, height=100, confidence=0.90, defect_type='collapse'),
        ]
        
        result = postprocessor.nms(bboxes)
        
        assert len(result) == 2

    def test_iou_calculation(self, postprocessor):
        """正常场景：IoU计算准确性"""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100, confidence=0.9, defect_type='collapse')
        bbox2 = BoundingBox(x=50, y=50, width=100, height=100, confidence=0.9, defect_type='collapse')
        bbox3 = BoundingBox(x=200, y=200, width=100, height=100, confidence=0.9, defect_type='collapse')
        
        iou_overlap = postprocessor._calculate_iou(bbox1, bbox2)
        iou_no_overlap = postprocessor._calculate_iou(bbox1, bbox3)
        iou_identical = postprocessor._calculate_iou(bbox1, bbox1)
        
        assert abs(iou_overlap - 0.1428) < 0.01
        assert iou_no_overlap == 0.0
        assert iou_identical == 1.0

    def test_bbox_smoothing(self, postprocessor):
        """正常场景：边界框平滑"""
        prev_bboxes = [
            BoundingBox(x=50, y=50, width=100, height=100, confidence=0.9, defect_type='collapse'),
        ]
        curr_bboxes = [
            BoundingBox(x=52, y=48, width=102, height=98, confidence=0.92, defect_type='collapse'),
        ]
        
        smoothed = postprocessor.smooth_bboxes(curr_bboxes, prev_bboxes)
        
        assert len(smoothed) == 1
        assert smoothed[0].x == 51
        assert smoothed[0].y == 49

    def test_bbox_smoothing_no_prev(self, postprocessor):
        """边界场景：无前帧时不进行平滑"""
        curr_bboxes = [
            BoundingBox(x=50, y=50, width=100, height=100, confidence=0.9, defect_type='collapse'),
        ]
        
        smoothed = postprocessor.smooth_bboxes(curr_bboxes, None)
        
        assert smoothed == curr_bboxes

    def test_clustering(self, postprocessor):
        """正常场景：相似缺陷聚类"""
        candidates = [
            DefectCandidate(defect_type='collapse', confidence=0.9,
                           bbox=BoundingBox(x=50, y=50, width=30, height=30, confidence=0.9, defect_type='collapse')),
            DefectCandidate(defect_type='collapse', confidence=0.85,
                           bbox=BoundingBox(x=60, y=60, width=30, height=30, confidence=0.85, defect_type='collapse')),
            DefectCandidate(defect_type='atrophy', confidence=0.88,
                           bbox=BoundingBox(x=200, y=50, width=30, height=30, confidence=0.88, defect_type='atrophy')),
        ]
        
        result = postprocessor.cluster_defects(candidates)
        
        assert len(result) == 2
        assert result[0].defect_type == 'collapse'
        assert result[1].defect_type == 'atrophy'


class TestDefectDetectorIntegration:
    """缺陷检测器集成测试"""

    @pytest.fixture
    def config(self):
        cfg = DefectConfig(
            enabled=True,
            cnn_model={
                'input_size': [224, 224],
                'num_classes': 4,
                'use_attention': True,
                'dropout_rate': 0.3,
            },
            image_preprocessing={
                'resize': [224, 224],
                'normalization': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                },
                'quality_check': {
                    'enabled': True,
                    'min_resolution': [100, 100],
                    'min_brightness': 5,
                    'max_brightness': 250,
                    'blurriness_threshold': 50,
                },
            },
            defect_types={
                'classes': ['normal', 'collapse', 'atrophy', 'cracking'],
            },
            confidence_threshold=0.6,
            auto_review_threshold=0.85,
        )
        cfg.__dict__['post_processing'] = {
            'bbox_smoothing': True,
            'smoothing_factor': 0.5,
            'nms_enabled': True,
            'nms_iou_threshold': 0.5,
            'clustering_enabled': True,
            'clustering_distance': 50,
        }
        return cfg

    @pytest.fixture
    def detector(self, config):
        return DefectDetector(config)

    def test_detect_image_normal(self, detector):
        """正常场景：完整的图像检测流程"""
        np.random.seed(42)
        img = detector.preprocessor._simulate_load_image("test_batch_001_img_001")
        
        result = detector.detect_image(img, batch_id="BATCH-001", image_id="IMG-001")
        
        assert result is not None
        assert result.batch_id == "BATCH-001"
        assert result.image_id == "IMG-001"
        assert result.defect_type in ['normal', 'collapse', 'atrophy', 'cracking']
        assert 0 <= result.confidence <= 1

    def test_detect_with_known_defect_patterns(self, detector):
        """正常场景：已知缺陷模式的分类准确率"""
        class_names = ['normal', 'collapse', 'atrophy', 'cracking']
        results = []
        
        for class_idx, defect_type in enumerate(class_names):
            for i in range(10):
                img = detector.preprocessor._simulate_load_image(f"test_{defect_type}_{i}")
                
                if class_idx > 0:
                    img[50+class_idx*20:100+class_idx*20, 50+class_idx*20:100+class_idx*20] = 0
                
                result = detector.detect_image(img, batch_id=f"TEST-{defect_type}", image_id=f"IMG-{i}")
                results.append((result.defect_type, defect_type))
        
        correct = sum(1 for pred, true in results if pred == true)
        accuracy = correct / len(results)
        
        assert accuracy > 0.4

    def test_low_quality_image_rejection(self, detector):
        """异常场景：低质量图像被拒绝"""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 3
        
        result = detector.detect_image(img, batch_id="BATCH-001", image_id="IMG-001")
        
        assert result is None or result.quality_passed == False

    def test_confidence_threshold(self, detector):
        """边界场景：低置信度结果触发人工审核"""
        detector.confidence_threshold = 0.9
        
        img = detector.preprocessor._simulate_load_image("test_low_conf_001")
        
        result = detector.detect_image(img, batch_id="BATCH-001", image_id="IMG-001")
        
        assert result is not None
        if result.confidence < 0.85:
            assert result.needs_review == True

    def test_batch_processing(self, detector):
        """正常场景：批量图像处理"""
        batch_id = "BATCH-TEST-001"
        images = []
        
        for i in range(20):
            img = detector.preprocessor._simulate_load_image(f"batch_img_{i:03d}")
            images.append(img)
        
        results = []
        for i, img in enumerate(images):
            result = detector.detect_image(img, batch_id=batch_id, image_id=f"IMG-{i:03d}")
            if result is not None:
                results.append(result)
        
        assert len(results) > 0
        
        stats = detector.get_batch_stats(batch_id)
        assert stats is not None
        assert stats.batch_id == batch_id
        assert stats.total_images == 20
        assert 0 <= stats.defect_rate <= 1
        assert 0 <= stats.quality_score <= 1

    def test_defect_and_drying_curve_correlation(self, detector):
        """正常场景：缺陷与冻干曲线的关联性分析"""
        test_data = []
        
        for temp in [-30, -20, -10, 0, 10, 20]:
            batch_id = f"BATCH-TEMP-{temp}"
            
            for i in range(10):
                img = detector.preprocessor._simulate_load_image(f"{batch_id}_img_{i}")
                
                if temp > 0:
                    img[80:140, 80:140] = np.clip(img[80:140, 80:140] - 100, 0, 255)
                
                result = detector.detect_image(img, batch_id=batch_id, image_id=f"IMG-{i}")
                
                if result is not None:
                    test_data.append({
                        'temp': temp,
                        'defect_type': result.defect_type,
                        'confidence': result.confidence,
                        'is_defect': result.defect_type != 'normal'
                    })
        
        high_temp_defects = sum(1 for d in test_data if d['temp'] > 0 and d['is_defect'])
        low_temp_defects = sum(1 for d in test_data if d['temp'] <= 0 and d['is_defect'])
        
        high_temp_rate = high_temp_defects / max(1, sum(1 for d in test_data if d['temp'] > 0))
        low_temp_rate = low_temp_defects / max(1, sum(1 for d in test_data if d['temp'] <= 0))
        
        assert high_temp_rate >= low_temp_rate * 0.5

    def test_batch_traceability(self, detector):
        """正常场景：批次追溯完整性"""
        batch_id = "BATCH-TRACE-001"
        formula_id = "FORMULA-001"
        profile_id = 1
        device_id = 3
        
        for i in range(5):
            img = detector.preprocessor._simulate_load_image(f"{batch_id}_img_{i}")
            result = detector.detect_image(
                img, 
                batch_id=batch_id, 
                image_id=f"IMG-{i}",
                formula_id=formula_id,
                profile_id=profile_id,
                device_id=device_id
            )
        
        stats = detector.get_batch_stats(batch_id)
        
        assert stats is not None
        assert stats.batch_id == batch_id
        assert stats.total_images == 5
        assert len(stats.defect_counts) > 0
        
        records = detector.get_batch_records(batch_id)
        assert len(records) == 5
        
        for record in records:
            assert record.batch_id == batch_id
            assert record.formula_id == formula_id
            assert record.device_id == device_id
            assert record.defect_type in ['normal', 'collapse', 'atrophy', 'cracking']

    def test_multiple_batch_traceability(self, detector):
        """边界场景：多批次追溯不混淆"""
        for batch_idx in range(3):
            batch_id = f"BATCH-MULTI-{batch_idx:03d}"
            
            for i in range(5):
                img = detector.preprocessor._simulate_load_image(f"{batch_id}_img_{i}")
                detector.detect_image(img, batch_id=batch_id, image_id=f"IMG-{i}")
        
        for batch_idx in range(3):
            batch_id = f"BATCH-MULTI-{batch_idx:03d}"
            stats = detector.get_batch_stats(batch_id)
            
            assert stats is not None
            assert stats.batch_id == batch_id
            assert stats.total_images == 5

    def test_review_workflow(self, detector):
        """正常场景：人工审核工作流"""
        img = detector.preprocessor._simulate_load_image("review_test_001")
        
        result = detector.detect_image(img, batch_id="BATCH-REVIEW", image_id="IMG-001")
        assert result is not None
        
        initial_defect = result.defect_type
        
        detector.review_defect(result.defect_id, "cracking", reviewer_id="QA-001")
        
        records = detector.get_batch_records("BATCH-REVIEW")
        reviewed = [r for r in records if r.defect_id == result.defect_id][0]
        
        assert reviewed.reviewed == True
        assert reviewed.final_defect_type == "cracking"
        assert reviewed.reviewer_id == "QA-001"
