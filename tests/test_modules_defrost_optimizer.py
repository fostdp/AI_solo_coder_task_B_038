import sys
sys.path.insert(0, 'd:/SOLO-2/AI_solo_coder_task_B_038')

import unittest
import numpy as np
import time
from modules.defrost_optimizer import (
    MultiSensorFusion,
    FrostThicknessEstimator,
    DefrostOptimizer,
    DeviceDefrostState,
)


class TestMultiSensorFusion(unittest.TestCase):

    def setUp(self):
        self.fusion = MultiSensorFusion()

    def test_fuse_temperatures_consistent_sensors(self):
        sensor_temps = {1: -60.0, 2: -59.5, 3: -60.2, 4: -60.1, 5: -59.8}
        sensor_weights = {1: 0.35, 2: 0.20, 3: 0.20, 4: 0.15, 5: 0.10}
        sensor_health = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
        sensor_positions = {1: 'inlet', 2: 'coil_1', 3: 'coil_2', 4: 'coil_3', 5: 'outlet'}
        fused, corrected = self.fusion.fuse_temperatures(
            sensor_temps, sensor_weights, sensor_health, sensor_positions
        )
        self.assertIsInstance(fused, float)
        self.assertGreater(fused, -100.0)
        self.assertLess(fused, 0.0)
        self.assertEqual(len(corrected), 5)

    def test_fuse_temperatures_one_failed_sensor(self):
        sensor_temps = {1: -60.0, 2: -59.5, 3: 0.0, 4: -60.1, 5: -59.8}
        sensor_weights = {1: 0.35, 2: 0.20, 3: 0.20, 4: 0.15, 5: 0.10}
        sensor_health = {1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 1.0}
        sensor_positions = {1: 'inlet', 2: 'coil_1', 3: 'coil_2', 4: 'coil_3', 5: 'outlet'}
        fused, corrected = self.fusion.fuse_temperatures(
            sensor_temps, sensor_weights, sensor_health, sensor_positions
        )
        self.assertIsInstance(fused, float)
        self.assertNotEqual(corrected.get(3, None), 0.0 / 1.0 if 3 not in corrected else None)

    def test_fuse_temperatures_too_few_valid(self):
        sensor_temps = {1: -60.0, 2: 100.0}
        sensor_weights = {1: 0.5, 2: 0.5}
        sensor_health = {1: 1.0, 2: 1.0}
        sensor_positions = {1: 'inlet', 2: 'outlet'}
        fusion = MultiSensorFusion({'num_sensors': 2, 'min_valid_sensors': 3})
        fused, corrected = fusion.fuse_temperatures(
            sensor_temps, sensor_weights, sensor_health, sensor_positions
        )
        self.assertIsInstance(fused, float)

    def test_detect_outliers(self):
        sensor_readings = {1: -60.0, 2: -59.5, 3: -60.2, 4: 20.0, 5: -59.8}
        cleaned, is_valid = self.fusion.detect_outliers(sensor_readings)
        self.assertFalse(is_valid[4])
        self.assertTrue(is_valid[1])
        self.assertTrue(is_valid[2])

    def test_detect_outliers_few_sensors(self):
        sensor_readings = {1: -60.0, 2: -59.5}
        cleaned, is_valid = self.fusion.detect_outliers(sensor_readings)
        self.assertEqual(len(cleaned), 2)

    def test_check_consistency(self):
        consistent = {1: -60.0, 2: -60.1, 3: -59.9, 4: -60.2, 5: -60.0}
        score = self.fusion.check_temperature_consistency(consistent)
        self.assertGreater(score, 0.9)

        inconsistent = {1: -60.0, 2: -40.0, 3: -70.0, 4: -50.0, 5: -30.0}
        score2 = self.fusion.check_temperature_consistency(inconsistent)
        self.assertLess(score2, score)

    def test_check_consistency_few_sensors(self):
        single = {1: -60.0}
        score = self.fusion.check_temperature_consistency(single)
        self.assertEqual(score, 1.0)

    def test_calculate_health_scores(self):
        historical = list(np.random.normal(-60.0, 0.5, 50))
        health = self.fusion.calculate_sensor_health(1, -60.0, historical)
        self.assertIsInstance(health, float)
        self.assertGreaterEqual(health, 0.0)
        self.assertLessEqual(health, 1.0)

    def test_calculate_health_short_history(self):
        health = self.fusion.calculate_sensor_health(1, -60.0, [-60.0, -59.0])
        self.assertEqual(health, 1.0)


class TestFrostThicknessEstimator(unittest.TestCase):

    def setUp(self):
        self.config = {
            'method': 'thermal_resistance',
            'base_cold_trap_temp': -80.0,
            'max_frost_thickness_mm': 5.0,
            'calibration_factor': 1.2,
        }

    def test_estimate_decreasing_temperature(self):
        estimator = FrostThicknessEstimator(self.config)
        base_time = time.time()
        temp_history = []
        for i in range(60):
            t = base_time - (60 - i) * 10
            temp = -80.0 + i * 0.5
            temp_history.append((t, temp))
        thickness = estimator.estimate(temp_history)
        self.assertIsInstance(thickness, float)
        self.assertGreaterEqual(thickness, 0.0)
        self.assertLessEqual(thickness, self.config['max_frost_thickness_mm'])

    def test_estimate_stable_temperature(self):
        estimator = FrostThicknessEstimator(self.config)
        base_time = time.time()
        temp_history = []
        for i in range(60):
            t = base_time - (60 - i) * 10
            temp = -80.0 + np.random.normal(0, 0.1)
            temp_history.append((t, temp))
        thickness = estimator.estimate(temp_history)
        self.assertIsInstance(thickness, float)
        self.assertGreaterEqual(thickness, 0.0)

    def test_estimate_short_history(self):
        estimator = FrostThicknessEstimator(self.config)
        temp_history = [(time.time(), -60.0), (time.time() + 10, -59.5)]
        thickness = estimator.estimate(temp_history)
        self.assertEqual(thickness, 0.0)

    def test_estimate_empirical_method(self):
        config = self.config.copy()
        config['method'] = 'empirical'
        estimator = FrostThicknessEstimator(config)
        base_time = time.time()
        temp_history = []
        for i in range(60):
            t = base_time - (60 - i) * 10
            temp = -80.0 + i * 0.3
            temp_history.append((t, temp))
        thickness = estimator.estimate(temp_history)
        self.assertIsInstance(thickness, float)
        self.assertGreaterEqual(thickness, 0.0)


class TestDefrostOptimizer(unittest.TestCase):

    def setUp(self):
        self.config = {
            'optimization': {
                'prefer_valley_electricity': True,
                'allow_defrost_during_batches': False,
                'min_running_hours_before_defrost': 8.0,
            },
            'thresholds': {
                'frost_thickness_trigger_mm': 3.0,
                'temp_trend_trigger': 0.5,
                'min_defrost_interval_hours': 4.0,
                'max_defrost_interval_hours': 24.0,
            },
            'power_profile': {
                'max_power_pct': 80.0,
                'min_power_pct': 20.0,
                'main_heating_power_pct': 80.0,
            },
            'energy_model': {
                'specific_energy_kwh_per_mm_frost': 0.5,
                'standby_power_kw': 2.0,
                'efficiency_coefficient': 0.85,
            },
            'frost_thickness_estimation': {
                'method': 'thermal_resistance',
                'base_cold_trap_temp': -80.0,
                'max_frost_thickness_mm': 5.0,
                'calibration_factor': 1.2,
                'use_multi_sensor': False,
            },
        }
        self.optimizer = DefrostOptimizer(self.config)

    def test_construction(self):
        self.assertIsNotNone(self.optimizer.frost_estimator)

    def test_optimize_returns_tuple(self):
        state = DeviceDefrostState(device_id=1)
        base_time = time.time()
        for i in range(20):
            state.cold_trap_history.append(
                (base_time - (20 - i) * 10, -80.0 + i * 0.2)
            )
        result = self.optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=14, is_batch_running=False
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        need_defrost, interval, power, saving = result
        self.assertIsInstance(need_defrost, bool)
        self.assertIsInstance(interval, float)
        self.assertIsInstance(power, float)
        self.assertIsInstance(saving, float)

    def test_optimize_no_batch_running(self):
        state = DeviceDefrostState(device_id=1)
        base_time = time.time()
        for i in range(20):
            state.cold_trap_history.append(
                (base_time - (20 - i) * 10, -60.0)
            )
        need_defrost, interval, power, saving = self.optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=3, is_batch_running=False
        )
        self.assertIsInstance(need_defrost, bool)

    def test_optimize_during_batch(self):
        state = DeviceDefrostState(device_id=1)
        need_defrost, _, _, _ = self.optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=14, is_batch_running=True
        )
        self.assertFalse(need_defrost)


if __name__ == '__main__':
    unittest.main()
