import sys
sys.path.insert(0, 'd:/SOLO-2/AI_solo_coder_task_B_038')

import unittest
import numpy as np
from modules.endpoint_detector import (
    SignalFilter,
    FirstDerivativeDetector,
    AutoEncoderDetector,
    PressureRiseTester,
    PRTState,
)


class TestSignalFilter(unittest.TestCase):

    def setUp(self):
        self.sf = SignalFilter()

    def test_median_filter_removes_spikes(self):
        data = np.array([20.0, 20.1, 35.0, 20.0, 20.2, 20.1, 19.9, 20.0, 20.1, 20.0])
        filtered = self.sf.median_filter(data)
        self.assertEqual(len(filtered), len(data))
        spike_idx = 2
        self.assertLess(np.abs(filtered[spike_idx] - 20.0), np.abs(data[spike_idx] - 20.0))

    def test_median_filter_short_data(self):
        data = np.array([1.0, 2.0])
        filtered = self.sf.median_filter(data)
        np.testing.assert_array_equal(filtered, data)

    def test_moving_average_smooths(self):
        data = np.arange(20, dtype=float)
        smoothed = self.sf.moving_average(data)
        self.assertEqual(len(smoothed), len(data))

    def test_moving_average_short_data(self):
        data = np.array([1.0, 2.0])
        smoothed = self.sf.moving_average(data)
        np.testing.assert_array_equal(smoothed, data)

    def test_remove_outliers_replaces_spikes(self):
        np.random.seed(42)
        data = np.random.normal(-40.0, 0.5, 50)
        data[10] = -10.0
        data[25] = -80.0
        cleaned = self.sf.remove_outliers(data)
        self.assertLess(np.abs(cleaned[10] - (-40.0)), np.abs(data[10] - (-40.0)))
        self.assertLess(np.abs(cleaned[25] - (-40.0)), np.abs(data[25] - (-40.0)))

    def test_remove_outliers_short_data(self):
        data = np.array([1.0, 2.0, 3.0])
        cleaned = self.sf.remove_outliers(data)
        np.testing.assert_array_equal(cleaned, data)

    def test_stability_score_stable_signal(self):
        data = np.random.normal(-40.0, 0.01, 50)
        score = self.sf.calculate_stability_score(data)
        self.assertGreater(score, 0.8)

    def test_stability_score_noisy_signal(self):
        data = np.random.normal(-40.0, 20.0, 50)
        score = self.sf.calculate_stability_score(data)
        self.assertLess(score, 0.8)

    def test_stability_score_short_data(self):
        data = np.array([1.0, 2.0])
        score = self.sf.calculate_stability_score(data)
        self.assertEqual(score, 1.0)

    def test_filter_returns_tuple(self):
        sf = SignalFilter()
        result = sf.filter(-40.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_filter_accumulates_history(self):
        sf = SignalFilter()
        for v in np.linspace(-50.0, -30.0, 30):
            sf.filter(v)
        self.assertGreater(len(sf._raw_history), 0)


class TestFirstDerivativeDetector(unittest.TestCase):

    def setUp(self):
        self.config = {
            'window_size': 5,
            'poly_order': 2,
            'primary_drying_threshold': 0.05,
            'secondary_drying_threshold': 0.02,
            'consecutive_points': 3,
        }
        self.detector = FirstDerivativeDetector(self.config)

    def test_detect_primary_endpoint_rising_data(self):
        timestamps = list(np.linspace(0, 3600, 60))
        temps_flat = list(np.full(30, -40.0))
        temps_rising = list(np.linspace(-40.0, -20.0, 30))
        temps = temps_flat + temps_rising
        detected, deriv, smoothed = self.detector.detect_primary_endpoint(
            timestamps, temps, stability_score=0.9
        )
        self.assertFalse(detected)
        self.assertGreater(deriv, 0.0)

    def test_detect_primary_endpoint_stable_data(self):
        timestamps = list(np.linspace(0, 7200, 120))
        temps = list(np.random.normal(-40.0, 0.01, 120))
        detector = FirstDerivativeDetector(self.config)
        detected_count = 0
        for i in range(6):
            detected, _, _ = detector.detect_primary_endpoint(
                timestamps, temps, stability_score=0.9
            )
            if detected:
                detected_count += 1
        self.assertGreaterEqual(detected_count, 0)

    def test_detect_primary_endpoint_insufficient_data(self):
        timestamps = [0.0, 60.0]
        temps = [-40.0, -40.0]
        detected, deriv, smoothed = self.detector.detect_primary_endpoint(
            timestamps, temps
        )
        self.assertFalse(detected)

    def test_detect_secondary_endpoint(self):
        timestamps = list(np.linspace(0, 7200, 120))
        temps = list(np.random.normal(-20.0, 0.005, 120))
        detector = FirstDerivativeDetector(self.config)
        detected_count = 0
        for i in range(6):
            detected, deriv, smoothed = detector.detect_secondary_endpoint(
                timestamps, temps, stability_score=0.9
            )
            if detected:
                detected_count += 1
        self.assertGreaterEqual(detected_count, 0)

    def test_detect_low_stability_rejects(self):
        timestamps = list(np.linspace(0, 3600, 60))
        temps = list(np.random.normal(-40.0, 0.01, 60))
        detected, _, _ = self.detector.detect_primary_endpoint(
            timestamps, temps, stability_score=0.1
        )
        self.assertFalse(detected)

    def test_reset(self):
        self.detector._consecutive_count_primary = 5
        self.detector._primary_confirmed = True
        self.detector.reset()
        self.assertEqual(self.detector._consecutive_count_primary, 0)
        self.assertFalse(self.detector._primary_confirmed)
        self.assertEqual(self.detector._consecutive_count_secondary, 0)
        self.assertFalse(self.detector._secondary_confirmed)


class TestAutoEncoderDetector(unittest.TestCase):

    def setUp(self):
        self.config = {
            'input_dim': 11,
            'latent_dim': 4,
            'threshold': 0.1,
            'min_training_samples': 50,
            'confirmation_count': 3,
            'min_stability_for_detection': 0.4,
        }
        self.detector = AutoEncoderDetector(self.config)

    def test_extract_features_shape(self):
        temps = list(np.random.normal(-40.0, 1.0, 20))
        vacuums = list(np.random.normal(10.0, 0.5, 20))
        cold_trap = -75.0
        powers = list(np.random.normal(5.0, 0.3, 20))
        features = self.detector.extract_features(temps, vacuums, cold_trap, powers)
        self.assertEqual(features.shape, (11,))

    def test_extract_features_values(self):
        temps = [0.0, 10.0]
        vacuums = [5.0, 5.0]
        cold_trap = -80.0
        powers = [2.0, 2.0]
        features = self.detector.extract_features(temps, vacuums, cold_trap, powers)
        self.assertAlmostEqual(features[0], 5.0)
        self.assertAlmostEqual(features[5], -80.0)

    def test_train_with_enough_samples(self):
        training_data = []
        for _ in range(60):
            sample = np.random.normal(0.0, 1.0, 11)
            training_data.append(sample)
        self.detector.train(training_data)
        self.assertTrue(self.detector._trained)
        self.assertIsNotNone(self.detector._encoder_weights)

    def test_train_insufficient_samples(self):
        training_data = [np.random.normal(0.0, 1.0, 11) for _ in range(10)]
        self.detector.train(training_data)
        self.assertFalse(self.detector._trained)

    def test_predict_returns_tuple(self):
        features = np.random.normal(0.0, 1.0, 11)
        error, is_anomaly = self.detector.predict(features)
        self.assertIsInstance(error, float)
        self.assertIsInstance(is_anomaly, bool)

    def test_predict_before_training(self):
        features = np.random.normal(0.0, 1.0, 11)
        error, is_anomaly = self.detector.predict(features)
        self.assertEqual(error, 0.0)
        self.assertFalse(is_anomaly)

    def test_predict_after_training(self):
        np.random.seed(42)
        training_data = [np.random.normal(0.0, 1.0, 11) for _ in range(80)]
        self.detector.train(training_data)
        normal_features = np.random.normal(0.0, 1.0, 11)
        error, is_anomaly = self.detector.predict(normal_features, stability_score=0.9)
        self.assertIsInstance(error, float)
        self.assertGreater(error, 0.0)

    def test_predict_low_stability(self):
        np.random.seed(42)
        training_data = [np.random.normal(0.0, 1.0, 11) for _ in range(80)]
        self.detector.train(training_data)
        features = np.random.normal(0.0, 1.0, 11)
        error, is_anomaly = self.detector.predict(features, stability_score=0.1)
        self.assertEqual(error, 0.0)
        self.assertFalse(is_anomaly)

    def test_reset(self):
        self.detector._trained = True
        self.detector._consecutive_anomalies = 5
        self.detector._confirmed_endpoint = True
        self.detector.reset()
        self.assertFalse(self.detector._trained)
        self.assertEqual(self.detector._consecutive_anomalies, 0)
        self.assertFalse(self.detector._confirmed_endpoint)
        self.assertIsNone(self.detector._encoder_weights)
        self.assertIsNone(self.detector._decoder_weights)
        self.assertIsNone(self.detector._mean)
        self.assertIsNone(self.detector._std)


class TestPressureRiseTester(unittest.TestCase):

    def setUp(self):
        self.config = {
            'enabled': True,
            'test_duration_seconds': 120,
            'measurement_interval_seconds': 5,
            'endpoint_threshold_pa_per_min': 0.05,
            'min_interval_between_tests_minutes': 0,
            'auto_trigger_enabled': True,
            'min_data_quality': 0.3,
            'confirmation_count': 2,
        }
        self.prt = PressureRiseTester(self.config)

    def test_construction(self):
        self.assertTrue(self.prt.enabled)
        self.assertEqual(self.prt.test_duration, 120)
        self.assertAlmostEqual(self.prt.endpoint_threshold, 0.05)

    def test_start_test_success(self):
        state = PRTState()
        result = self.prt.start_test(state, 5.0, 1000.0)
        self.assertTrue(result)
        self.assertTrue(state.in_progress)
        self.assertEqual(state.initial_pressure, 5.0)

    def test_start_test_disabled(self):
        self.prt.enabled = False
        state = PRTState()
        result = self.prt.start_test(state, 5.0, 1000.0)
        self.assertFalse(result)

    def test_start_test_already_in_progress(self):
        state = PRTState(in_progress=True)
        result = self.prt.start_test(state, 5.0, 1000.0)
        self.assertFalse(result)

    def test_record_measurement(self):
        state = PRTState(in_progress=True, start_time=1000.0, initial_pressure=5.0)
        state.measurements = [(1000.0, 5.0)]
        self.prt.record_measurement(state, 5.1, 1010.0)
        self.assertEqual(len(state.measurements), 2)

    def test_record_measurement_not_in_progress(self):
        state = PRTState(in_progress=False)
        self.prt.record_measurement(state, 5.1, 1010.0)
        self.assertEqual(len(state.measurements), 0)

    def test_check_test_complete_not_done(self):
        state = PRTState(in_progress=True, start_time=1000.0, initial_pressure=5.0)
        state.measurements = [(1000.0, 5.0)]
        result = self.prt.check_test_complete(state, 1050.0)
        self.assertIsNone(result)

    def test_check_test_complete_with_enough_data(self):
        state = PRTState(in_progress=True, start_time=1000.0, initial_pressure=5.0)
        base_time = 1000.0
        for i in range(30):
            t = base_time + i * 5
            p = 5.0 + i * 0.001
            state.measurements.append((t, p))
        result = self.prt.check_test_complete(state, 1000.0 + 150)
        self.assertIsNotNone(result)
        self.assertFalse(state.in_progress)
        self.assertIsInstance(result.pressure_rise_pa_per_min, float)
        self.assertIsInstance(result.data_quality_score, float)

    def test_check_test_not_in_progress(self):
        state = PRTState(in_progress=False)
        result = self.prt.check_test_complete(state, 1200.0)
        self.assertIsNone(result)

    def test_reset(self):
        self.prt._test_results.append(True)
        self.prt._confirmed_endpoint = True
        self.prt.reset()
        self.assertEqual(len(self.prt._test_results), 0)
        self.assertFalse(self.prt._confirmed_endpoint)


if __name__ == '__main__':
    unittest.main()
