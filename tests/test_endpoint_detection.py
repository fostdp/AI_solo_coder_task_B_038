"""
冻干过程终点判定测试
覆盖：压力升法终点判定准确率、自编码器异常检测灵敏度、冻干周期缩短效果
场景：正常、边界、异常
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "microservices"))

import pytest
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from collections import deque
from unittest.mock import patch

from endpoint_detector.main import (
    FirstDerivativeDetector,
    AutoencoderDetector,
    PressureRiseTestManager,
    DeviceState,
)


class TestFirstDerivativeDetector:
    """一阶导数法测试"""

    @pytest.fixture
    def detector(self):
        config = {
            'window_size': 10,
            'poly_order': 2,
            'primary_drying_threshold': 0.05,
            'secondary_drying_threshold': 0.02,
            'consecutive_points': 3,
        }
        return FirstDerivativeDetector(config)

    def test_primary_drying_endpoint_normal(self, detector):
        """正常场景：一次干燥终点检测 - 温度上升速率下降"""
        times = []
        temps = []
        t0 = datetime.now(timezone.utc).timestamp()
        
        for i in range(100):
            times.append(t0 + i * 10)
            if i < 50:
                temps.append(-40 + i * 0.5)
            else:
                temps.append(-15 + (i - 50) * 0.004)
        
        for i in range(len(times)):
            is_end, deriv, smoothed = detector.detect_primary_endpoint(
                times[:i+1], temps[:i+1]
            )
        
        assert is_end == True
        assert deriv < 0.05
        assert -15 < smoothed < -10

    def test_secondary_drying_endpoint_normal(self, detector):
        """正常场景：二次干燥终点检测 - 温度变化率极小"""
        times = []
        temps = []
        t0 = datetime.now(timezone.utc).timestamp()
        
        for i in range(100):
            times.append(t0 + i * 10)
            if i < 50:
                temps.append(20 + i * 0.1)
            else:
                temps.append(25 + (i - 50) * 0.002)
        
        for i in range(len(times)):
            is_end, deriv, smoothed = detector.detect_secondary_endpoint(
                times[:i+1], temps[:i+1]
            )
        
        assert is_end == True
        assert deriv < 0.02

    def test_primary_drying_boundary_threshold(self, detector):
        """边界场景：阈值边界处的检测稳定性"""
        times = []
        temps = []
        t0 = datetime.now(timezone.utc).timestamp()
        
        for i in range(60):
            times.append(t0 + i * 10)
            if i < 30:
                temps.append(-40 + i * 0.5)
            else:
                temps.append(-25 + (i - 30) * 0.049)
        
        for i in range(len(times)):
            is_end, deriv, smoothed = detector.detect_primary_endpoint(
                times[:i+1], temps[:i+1]
            )
        
        assert is_end == False

    def test_secondary_drying_boundary_oscillating(self, detector):
        """边界场景：温度振荡时的检测"""
        times = []
        temps = []
        t0 = datetime.now(timezone.utc).timestamp()
        
        np.random.seed(42)
        for i in range(100):
            times.append(t0 + i * 10)
            base_temp = 25
            noise = np.random.uniform(-0.001, 0.001)
            temps.append(base_temp + noise)
        
        for i in range(len(times)):
            is_end, deriv, smoothed = detector.detect_secondary_endpoint(
                times[:i+1], temps[:i+1]
            )
        
        assert is_end == True

    def test_insufficient_data(self, detector):
        """异常场景：数据不足时的处理"""
        times = [0, 10]
        temps = [-40, -39]
        
        is_end, deriv, smoothed = detector.detect_primary_endpoint(times, temps)
        assert is_end == False
        assert deriv == 0.0

    def test_savitzky_golay_smoothing(self, detector):
        """正常场景：SG平滑滤波效果验证"""
        np.random.seed(42)
        raw = np.linspace(-40, 25, 100)
        noisy = raw + np.random.normal(0, 0.5, 100)
        
        smoothed = detector.savitzky_golay(noisy, 10, 2)
        
        noise_before = np.std(noisy - raw)
        noise_after = np.std(smoothed - raw)
        
        assert noise_after < noise_before * 0.5

    def test_consecutive_points_required(self, detector):
        """边界场景：连续点确认机制"""
        times = []
        temps = []
        t0 = datetime.now(timezone.utc).timestamp()
        
        for i in range(40):
            times.append(t0 + i * 10)
            temps.append(25 + (i - 30) * 0.002 if i >= 30 else 20 + i * 0.1)
        
        is_end, deriv, smoothed = detector.detect_secondary_endpoint(times, temps)
        assert is_end == False

        for i in range(40, 55):
            times.append(t0 + i * 10)
            temps.append(25 + (i - 30) * 0.002)
            is_end, deriv, smoothed = detector.detect_secondary_endpoint(times, temps)
        
        assert is_end == True


class TestAutoencoderDetector:
    """自编码器异常检测测试"""

    @pytest.fixture
    def detector(self):
        config = {
            'input_dim': 11,
            'latent_dim': 4,
            'hidden_layers': [32, 16],
            'threshold': 0.1,
            'min_training_samples': 50,
        }
        return AutoencoderDetector(config)

    def test_normal_pattern_training(self, detector):
        """正常场景：正常模式下的训练与检测"""
        np.random.seed(42)
        training_data = []
        
        for i in range(100):
            temps = np.random.normal(-30, 2, 8)
            vacuums = np.random.normal(10, 1, 2)
            powers = np.random.normal(50, 5, 8)
            cold_trap = np.random.normal(-80, 1)
            features = detector.extract_features(temps, vacuums, cold_trap, powers)
            training_data.append(features)
        
        detector.train(training_data)
        assert detector._trained == True
        assert detector._encoder_weights is not None
        assert detector._decoder_weights is not None

    def test_normal_reconstruction_error(self, detector):
        """正常场景：正常样本的重构误差应低于阈值"""
        np.random.seed(42)
        training_data = []
        
        for i in range(100):
            temps = np.random.normal(-30, 2, 8)
            vacuums = np.random.normal(10, 1, 2)
            powers = np.random.normal(50, 5, 8)
            cold_trap = np.random.normal(-80, 1)
            features = detector.extract_features(temps, vacuums, cold_trap, powers)
            training_data.append(features)
        
        detector.train(training_data)
        
        test_temps = np.random.normal(-30, 2, 8)
        test_vacs = np.random.normal(10, 1, 2)
        test_powers = np.random.normal(50, 5, 8)
        test_cold = np.random.normal(-80, 1)
        test_features = detector.extract_features(test_temps, test_vacs, test_cold, test_powers)
        
        error, is_anomaly = detector.predict(test_features)
        assert is_anomaly == False
        assert error < detector.threshold

    def test_anomaly_detection_sensitivity(self, detector):
        """边界场景：异常样本检测灵敏度"""
        np.random.seed(42)
        training_data = []
        
        for i in range(100):
            temps = np.random.normal(-30, 2, 8)
            vacuums = np.random.normal(10, 1, 2)
            powers = np.random.normal(50, 5, 8)
            cold_trap = np.random.normal(-80, 1)
            features = detector.extract_features(temps, vacuums, cold_trap, powers)
            training_data.append(features)
        
        detector.train(training_data)
        
        anomaly_temps = np.random.normal(0, 5, 8)
        anomaly_vacs = np.random.normal(100, 10, 2)
        anomaly_powers = np.random.normal(100, 20, 8)
        anomaly_cold = np.random.normal(-20, 5)
        anomaly_features = detector.extract_features(anomaly_temps, anomaly_vacs, anomaly_cold, anomaly_powers)
        
        error, is_anomaly = detector.predict(anomaly_features)
        assert is_anomaly == True
        assert error > detector.threshold

    def test_insufficient_training_data(self, detector):
        """异常场景：训练数据不足"""
        np.random.seed(42)
        training_data = []
        
        for i in range(30):
            temps = np.random.normal(-30, 2, 8)
            vacuums = np.random.normal(10, 1, 2)
            powers = np.random.normal(50, 5, 8)
            cold_trap = np.random.normal(-80, 1)
            features = detector.extract_features(temps, vacuums, cold_trap, powers)
            training_data.append(features)
        
        detector.train(training_data)
        assert detector._trained == False

    def test_predict_before_training(self, detector):
        """异常场景：未训练时预测"""
        features = np.random.rand(11)
        error, is_anomaly = detector.predict(features)
        assert error == 0.0
        assert is_anomaly == False

    def test_feature_extraction_normal(self, detector):
        """正常场景：特征提取正确性"""
        temps = np.array([-30.0, -30.1, -29.9, -30.2, -30.0, -29.8, -30.1, -29.9])
        vacuums = np.array([10.0, 10.1])
        powers = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        cold_trap = -80.0
        
        features = detector.extract_features(temps, vacuums, cold_trap, powers)
        
        assert len(features) == 11
        assert abs(features[0] - (-30.0)) < 0.1
        assert abs(features[5] - (-80.0)) < 0.1


class TestPressureRiseTestManager:
    """压力升测试管理器测试"""

    @pytest.fixture
    def prt_manager(self):
        config = {
            'enabled': True,
            'test_duration_seconds': 60,
            'measurement_interval_seconds': 5,
            'endpoint_threshold_pa_per_min': 0.05,
            'min_interval_between_tests_minutes': 5,
            'auto_trigger_enabled': True,
        }
        return PressureRiseTestManager(config)

    def test_primary_drying_prt_normal(self, prt_manager):
        """正常场景：一次干燥PRT - 冰升华导致压力快速上升"""
        state = DeviceState(device_id=1, batch_id="TEST-001", current_phase="primary_drying")
        
        start_time = time.time()
        with patch('endpoint_detector.main.time.time') as mock_time:
            mock_time.return_value = start_time
            prt_manager.start_test(state, 1.0)
            
            for i in range(25):
                mock_time.return_value = start_time + i * prt_manager.measurement_interval
                pressure = 1.0 + i * 0.05
                prt_manager.record_measurement(state, pressure)
        
        state.prt_start_time -= prt_manager.test_duration + 1
        
        result = prt_manager.check_test_complete(state)
        
        assert result is not None
        assert result.is_endpoint_detected == False
        assert result.pressure_rise_pa_per_min > prt_manager.endpoint_threshold

    def test_secondary_drying_prt_normal(self, prt_manager):
        """正常场景：二次干燥PRT - 压力上升缓慢"""
        state = DeviceState(device_id=1, batch_id="TEST-001", current_phase="secondary_drying")
        
        start_time = time.time()
        with patch('endpoint_detector.main.time.time') as mock_time:
            mock_time.return_value = start_time
            prt_manager.start_test(state, 0.05)
            
            for i in range(25):
                mock_time.return_value = start_time + i * prt_manager.measurement_interval
                pressure = 0.05 + i * 0.002
                prt_manager.record_measurement(state, pressure)
        
        state.prt_start_time -= prt_manager.test_duration + 1
        
        result = prt_manager.check_test_complete(state)
        
        assert result is not None
        assert result.is_endpoint_detected == True
        assert result.pressure_rise_pa_per_min < prt_manager.endpoint_threshold

    def test_prt_boundary_threshold(self, prt_manager):
        """边界场景：阈值附近的PRT结果"""
        state = DeviceState(device_id=1, batch_id="TEST-001", current_phase="secondary_drying")
        
        start_time = time.time()
        with patch('endpoint_detector.main.time.time') as mock_time:
            mock_time.return_value = start_time
            prt_manager.start_test(state, 0.05)
            
            for i in range(25):
                mock_time.return_value = start_time + i * prt_manager.measurement_interval
                pressure = 0.05 + (i * prt_manager.measurement_interval / 60.0) * (prt_manager.endpoint_threshold * 0.99)
                prt_manager.record_measurement(state, pressure)
        
        state.prt_start_time -= prt_manager.test_duration + 1
        
        result = prt_manager.check_test_complete(state)
        
        assert result is not None
        assert result.is_endpoint_detected == True

    def test_prt_in_progress(self, prt_manager):
        """异常场景：测试进行中（未到结束时间）"""
        state = DeviceState(device_id=1, batch_id="TEST-001")
        
        prt_manager.start_test(state, 1.0)
        
        for i in range(5):
            pressure = 1.0 + i * 0.05
            prt_manager.record_measurement(state, pressure)
        
        result = prt_manager.check_test_complete(state)
        
        assert result is None

    def test_prt_too_frequent(self, prt_manager):
        """异常场景：测试间隔过短 - 应自动忽略"""
        state = DeviceState(device_id=1, batch_id="TEST-001")
        state.last_prt_time = time.time() - 60
        
        initial_time = state.last_prt_time
        
        prt_manager.start_test(state, 1.0)
        
        assert state.last_prt_time == initial_time
        assert state.prt_in_progress == False

    def test_linear_regression_accuracy(self, prt_manager):
        """正常场景：线性回归准确率（通过测试完整流程验证）"""
        np.random.seed(42)
        state = DeviceState(device_id=1, batch_id="TEST-001")
        
        true_rate = 0.03
        prt_manager.endpoint_threshold = 0.05
        
        start_time = time.time()
        with patch('endpoint_detector.main.time.time') as mock_time:
            mock_time.return_value = start_time
            prt_manager.start_test(state, 1.0)
            
            for i in range(25):
                mock_time.return_value = start_time + i * prt_manager.measurement_interval
                pressure = 1.0 + (i * prt_manager.measurement_interval / 60.0) * true_rate + np.random.normal(0, 0.001)
                prt_manager.record_measurement(state, pressure)
        
        state.prt_start_time -= prt_manager.test_duration + 1
        
        result = prt_manager.check_test_complete(state)
        
        assert result is not None
        assert abs(result.pressure_rise_pa_per_min - true_rate) < 0.01
        assert result.is_endpoint_detected == True


class TestCycleTimeReduction:
    """冻干周期缩短效果测试"""

    def test_combined_decision_reduces_cycle(self):
        """集成测试：组合决策相比传统方法的周期缩短效果"""
        config_fd = {
            'window_size': 10,
            'poly_order': 2,
            'primary_drying_threshold': 0.05,
            'secondary_drying_threshold': 0.02,
            'consecutive_points': 5,
        }
        detector = FirstDerivativeDetector(config_fd)
        
        times = []
        temps = []
        t0 = datetime.now(timezone.utc).timestamp()
        
        for i in range(300):
            times.append(t0 + i * 10)
            if i < 100:
                temps.append(-40 + i * 0.5)
            elif i < 150:
                temps.append(-15 + (i - 100) * 0.8)
            else:
                temps.append(25 + (i - 150) * 0.002)
        
        traditional_endpoint = 240
        auto_endpoint = None
        
        for i in range(len(times)):
            is_end, deriv, smoothed = detector.detect_secondary_endpoint(
                times[:i+1], temps[:i+1]
            )
            if is_end and auto_endpoint is None:
                auto_endpoint = i
        
        assert auto_endpoint is not None
        assert auto_endpoint < traditional_endpoint
        
        reduction_pct = (traditional_endpoint - auto_endpoint) / traditional_endpoint * 100
        assert reduction_pct > 10
