import sys
sys.path.insert(0, 'd:/SOLO-2/AI_solo_coder_task_B_038')

import unittest
import numpy as np
from modules.defect_classifier import (
    ClassifierConfig,
    ImagePreprocessor,
    ONNXEngine,
    CNNClassifier,
    DefectPostprocessor,
    DefectCandidate,
    BoundingBox,
    ImageQualityResult,
)


class TestClassifierConfig(unittest.TestCase):

    def test_construction(self):
        config = ClassifierConfig(
            confidence_threshold=0.7,
            onnx_model_path=None,
            use_onnx=False,
        )
        self.assertAlmostEqual(config.confidence_threshold, 0.7)
        self.assertFalse(config.use_onnx)
        self.assertIsNone(config.onnx_model_path)

    def test_default_construction(self):
        config = ClassifierConfig()
        self.assertIsInstance(config.image_preprocessing, dict)
        self.assertIsInstance(config.cnn_model, dict)
        self.assertIsInstance(config.post_processing, dict)
        self.assertIsInstance(config.defect_types, dict)
        self.assertAlmostEqual(config.confidence_threshold, 0.7)

    def test_construction_with_all_fields(self):
        config = ClassifierConfig(
            image_preprocessing={'resize': [224, 224]},
            cnn_model={'num_classes': 4},
            post_processing={'nms_threshold': 0.5},
            confidence_threshold=0.8,
            defect_types={'normal': 'normal', 'crack': 'crack'},
            onnx_model_path='/tmp/model.onnx',
            use_onnx=True,
        )
        self.assertEqual(config.image_preprocessing['resize'], [224, 224])
        self.assertAlmostEqual(config.confidence_threshold, 0.8)
        self.assertEqual(config.onnx_model_path, '/tmp/model.onnx')


class TestImagePreprocessor(unittest.TestCase):

    def setUp(self):
        self.config = ClassifierConfig()
        self.preprocessor = ImagePreprocessor(self.config)

    def test_check_quality_passes_good_image(self):
        image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        result = self.preprocessor.check_quality(image)
        self.assertIsInstance(result, ImageQualityResult)
        self.assertTrue(result.passed)
        self.assertGreater(result.brightness, 0)

    def test_check_quality_fails_low_resolution(self):
        image = np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8)
        config = ClassifierConfig(
            image_preprocessing={'quality_check': {'enabled': True, 'min_resolution': [100, 100]}}
        )
        preprocessor = ImagePreprocessor(config)
        result = preprocessor.check_quality(image)
        self.assertFalse(result.passed)
        self.assertIsNotNone(result.error_message)

    def test_check_quality_disabled(self):
        config = ClassifierConfig(
            image_preprocessing={'quality_check': {'enabled': False}}
        )
        preprocessor = ImagePreprocessor(config)
        image = np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8)
        result = preprocessor.check_quality(image)
        self.assertTrue(result.passed)

    def test_normalize_illumination(self):
        image = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        normalized = self.preprocessor.normalize_illumination(image)
        self.assertEqual(normalized.shape, image.shape)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 255))

    def test_normalize_illumination_dark_image(self):
        image = np.random.randint(0, 30, (64, 64, 3), dtype=np.uint8)
        normalized = self.preprocessor.normalize_illumination(image)
        self.assertEqual(normalized.shape, image.shape)
        avg_brightness = float(np.mean(normalized))
        self.assertGreater(avg_brightness, float(np.mean(image)))

    def test_augment_image_disabled(self):
        config = ClassifierConfig(
            image_preprocessing={'augmentation': {'enabled': False}}
        )
        preprocessor = ImagePreprocessor(config)
        image = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        augmented = preprocessor.augment_image(image)
        self.assertEqual(len(augmented), 1)
        np.testing.assert_array_equal(augmented[0], image)

    def test_augment_image_enabled(self):
        config = ClassifierConfig(
            image_preprocessing={
                'augmentation': {
                    'enabled': True,
                    'num_variants': 3,
                    'brightness_range': [-0.1, 0.1],
                    'contrast_range': [-0.05, 0.05],
                }
            }
        )
        preprocessor = ImagePreprocessor(config)
        image = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
        augmented = preprocessor.augment_image(image)
        self.assertEqual(len(augmented), 4)
        self.assertEqual(augmented[0].shape, image.shape)

    def test_domain_adaptation_preprocess(self):
        image = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        adapted = self.preprocessor.domain_adaptation_preprocess(image)
        self.assertEqual(adapted.shape, image.shape)
        self.assertEqual(adapted.dtype, np.uint8)
        self.assertTrue(np.all(adapted >= 0))
        self.assertTrue(np.all(adapted <= 255))


class TestONNXEngine(unittest.TestCase):

    def test_initialization_no_model(self):
        engine = ONNXEngine(model_path=None, use_gpu=False)
        self.assertFalse(engine.is_loaded)

    def test_initialization_nonexistent_model(self):
        engine = ONNXEngine(model_path='/nonexistent/model.onnx', use_gpu=False)
        self.assertFalse(engine.is_loaded)

    def test_predict_numpy_fallback(self):
        engine = ONNXEngine(model_path=None, use_gpu=False)
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        probs = engine.predict(x)
        self.assertEqual(probs.shape[0], 1)
        self.assertEqual(probs.shape[1], 4)
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, [1.0], atol=1e-5)

    def test_predict_batch_numpy(self):
        engine = ONNXEngine(model_path=None, use_gpu=False)
        images = [
            np.random.randn(1, 3, 224, 224).astype(np.float32),
            np.random.randn(1, 3, 224, 224).astype(np.float32),
        ]
        probs = engine.predict_batch(images)
        self.assertEqual(probs.shape[0], 2)
        self.assertEqual(probs.shape[1], 4)

    def test_is_available_property(self):
        engine = ONNXEngine(model_path=None, use_gpu=False)
        self.assertIsInstance(engine.is_available, bool)


class TestCNNClassifier(unittest.TestCase):

    def setUp(self):
        self.config = ClassifierConfig(
            confidence_threshold=0.5,
            use_onnx=False,
            onnx_model_path=None,
        )
        self.classifier = CNNClassifier(self.config, use_gpu=False)

    def test_predict_classes_with_dummy_input(self):
        input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
        probs = self.classifier.engine.predict(input_tensor)
        candidates = self.classifier._probs_to_candidates(probs[0])
        self.assertIsInstance(candidates, list)
        for candidate in candidates:
            self.assertIsInstance(candidate, DefectCandidate)
            self.assertIsInstance(candidate.defect_type, str)
            self.assertIsInstance(candidate.confidence, float)

    def test_predict_with_augmentation(self):
        image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        preprocessed = self.classifier.preprocessor.preprocess(image)
        input_tensor = preprocessed
        probs = self.classifier.engine.predict(input_tensor)
        candidates = self.classifier._probs_to_candidates(probs[0])
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)

    def test_classify_batch(self):
        images = [
            np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8),
            np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8),
        ]
        all_candidates = []
        for image in images:
            preprocessed = self.classifier.preprocessor.preprocess(image)
            probs = self.classifier.engine.predict(preprocessed)
            candidates = self.classifier._probs_to_candidates(probs[0])
            all_candidates.append(candidates)
        self.assertEqual(len(all_candidates), 2)
        self.assertGreater(len(all_candidates[0]), 0)

    def test_defect_types_defined(self):
        self.assertIn('normal', self.classifier.DEFECT_TYPES)
        self.assertIn('collapse', self.classifier.DEFECT_TYPES)
        self.assertIn('shrinkage', self.classifier.DEFECT_TYPES)
        self.assertIn('cracking', self.classifier.DEFECT_TYPES)


class TestDefectPostprocessor(unittest.TestCase):

    def setUp(self):
        self.postprocessor = DefectPostprocessor()

    def test_apply_nms_empty(self):
        result = self.postprocessor.apply_nms([])
        self.assertEqual(result, [])

    def test_apply_nms_single(self):
        candidate = DefectCandidate(
            defect_type='cracking',
            confidence=0.9,
            bbox=BoundingBox(x=10, y=10, width=50, height=50, confidence=0.9, defect_type='cracking'),
        )
        result = self.postprocessor.apply_nms([candidate])
        self.assertEqual(len(result), 1)

    def test_apply_nms_overlapping(self):
        candidates = [
            DefectCandidate(
                defect_type='cracking',
                confidence=0.9,
                bbox=BoundingBox(x=10, y=10, width=50, height=50, confidence=0.9, defect_type='cracking'),
            ),
            DefectCandidate(
                defect_type='cracking',
                confidence=0.8,
                bbox=BoundingBox(x=12, y=12, width=50, height=50, confidence=0.8, defect_type='cracking'),
            ),
        ]
        result = self.postprocessor.apply_nms(candidates)
        self.assertLessEqual(len(result), 2)
        if len(result) == 1:
            self.assertAlmostEqual(result[0].confidence, 0.9)

    def test_cluster_defects_close_defects(self):
        candidates = [
            DefectCandidate(
                defect_type='cracking',
                confidence=0.9,
                bbox=BoundingBox(x=10, y=10, width=20, height=20, confidence=0.9, defect_type='cracking'),
            ),
            DefectCandidate(
                defect_type='cracking',
                confidence=0.8,
                bbox=BoundingBox(x=25, y=25, width=20, height=20, confidence=0.8, defect_type='cracking'),
            ),
        ]
        clustered = self.postprocessor.cluster_defects(candidates, (224, 224))
        self.assertIsInstance(clustered, list)
        self.assertLessEqual(len(clustered), 2)

    def test_cluster_defects_different_types(self):
        candidates = [
            DefectCandidate(
                defect_type='cracking',
                confidence=0.9,
                bbox=BoundingBox(x=10, y=10, width=20, height=20, confidence=0.9, defect_type='cracking'),
            ),
            DefectCandidate(
                defect_type='collapse',
                confidence=0.8,
                bbox=BoundingBox(x=15, y=15, width=20, height=20, confidence=0.8, defect_type='collapse'),
            ),
        ]
        clustered = self.postprocessor.cluster_defects(candidates, (224, 224))
        self.assertEqual(len(clustered), 2)

    def test_cluster_defects_single(self):
        candidates = [
            DefectCandidate(
                defect_type='cracking',
                confidence=0.9,
                bbox=BoundingBox(x=10, y=10, width=20, height=20, confidence=0.9, defect_type='cracking'),
            ),
        ]
        clustered = self.postprocessor.cluster_defects(candidates, (224, 224))
        self.assertEqual(len(clustered), 1)

    def test_smooth_bounding_boxes(self):
        candidates = [
            DefectCandidate(
                defect_type='cracking',
                confidence=0.9,
                bbox=BoundingBox(x=10, y=10, width=50, height=50, confidence=0.9, defect_type='cracking'),
            ),
        ]
        smoothed = self.postprocessor.smooth_bounding_boxes(candidates, (224, 224))
        self.assertEqual(len(smoothed), 1)
        bbox = smoothed[0].bbox
        self.assertLessEqual(bbox.x, 10)
        self.assertLessEqual(bbox.y, 10)
        self.assertGreaterEqual(bbox.width, 50)

    def test_smooth_bounding_boxes_no_bbox(self):
        candidates = [
            DefectCandidate(defect_type='cracking', confidence=0.9, bbox=None),
        ]
        smoothed = self.postprocessor.smooth_bounding_boxes(candidates, (224, 224))
        self.assertEqual(len(smoothed), 1)
        self.assertIsNone(smoothed[0].bbox)

    def test_process_pipeline(self):
        candidates = [
            DefectCandidate(
                defect_type='cracking',
                confidence=0.9,
                bbox=BoundingBox(x=10, y=10, width=50, height=50, confidence=0.9, defect_type='cracking'),
            ),
            DefectCandidate(
                defect_type='shrinkage',
                confidence=0.7,
                bbox=BoundingBox(x=100, y=100, width=30, height=30, confidence=0.7, defect_type='shrinkage'),
            ),
        ]
        result = self.postprocessor.process(candidates, (224, 224))
        self.assertIsInstance(result, list)


if __name__ == '__main__':
    unittest.main()
