"""
冷阱除霜优化测试
覆盖：结霜厚度估算精度、除霜能耗降低效果、除霜周期的合理性
场景：正常、边界、异常
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "microservices"))

import pytest
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import asdict

from defrost_optimizer.main import (
    FrostThicknessEstimator,
    DefrostOptimizer,
    DeviceDefrostState,
)
from shared import DefrostConfig


class TestFrostThicknessEstimator:
    """结霜厚度估算器测试"""

    @pytest.fixture
    def estimator(self):
        config = {
            'method': 'thermal_resistance',
            'base_cold_trap_temp': -80.0,
            'max_frost_thickness_mm': 5.0,
            'calibration_factor': 1.2,
            'temp_window_minutes': 60,
        }
        return FrostThicknessEstimator(config)

    def test_thickness_estimation_normal(self, estimator):
        """正常场景：稳定结霜过程的厚度估算"""
        temp_history = []
        t0 = time.time()
        
        for i in range(720):
            t = t0 - (720 - i) * 10
            temp = -80.0 + i * 0.005
            temp_history.append((t, temp))
        
        thickness = estimator.estimate(temp_history)
        
        assert 0.0 < thickness <= 5.0
        assert abs(thickness - 4.3) < 1.0

    def test_thickness_estimation_no_frost(self, estimator):
        """正常场景：无霜状态的估算"""
        temp_history = []
        t0 = time.time()
        
        for i in range(100):
            t = t0 - (100 - i) * 10
            temp = -80.0 + np.random.normal(0, 0.1)
            temp_history.append((t, temp))
        
        thickness = estimator.estimate(temp_history)
        
        assert thickness < 0.5

    def test_thickness_estimation_heavy_frost(self, estimator):
        """边界场景：严重结霜的估算"""
        temp_history = []
        t0 = time.time()
        
        for i in range(720):
            t = t0 - (720 - i) * 10
            temp = -60.0 + np.random.normal(0, 0.5)
            temp_history.append((t, temp))
        
        thickness = estimator.estimate(temp_history)
        
        assert thickness <= 5.0
        assert thickness > 2.0

    def test_thickness_estimation_boundary_max(self, estimator):
        """边界场景：最大厚度限制"""
        temp_history = []
        t0 = time.time()
        
        for i in range(720):
            t = t0 - (720 - i) * 10
            temp = -50.0
            temp_history.append((t, temp))
        
        thickness = estimator.estimate(temp_history)
        
        assert thickness == 5.0

    def test_insufficient_data(self, estimator):
        """异常场景：数据不足"""
        temp_history = [(time.time() - i * 10, -80.0) for i in range(5)]
        
        thickness = estimator.estimate(temp_history)
        
        assert thickness == 0.0

    def test_thermal_resistance_vs_empirical(self, estimator):
        """正常场景：两种估算方法的一致性"""
        temp_history = []
        t0 = time.time()
        
        for i in range(300):
            t = t0 - (300 - i) * 10
            temp = -80.0 + i * 0.01
            temp_history.append((t, temp))
        
        tr_estimator = FrostThicknessEstimator({
            'method': 'thermal_resistance',
            'base_cold_trap_temp': -80.0,
            'max_frost_thickness_mm': 5.0,
            'calibration_factor': 1.2,
            'temp_window_minutes': 60,
        })
        
        emp_estimator = FrostThicknessEstimator({
            'method': 'empirical',
            'base_cold_trap_temp': -80.0,
            'max_frost_thickness_mm': 5.0,
            'calibration_factor': 1.2,
            'temp_window_minutes': 60,
        })
        
        tr_thickness = tr_estimator.estimate(temp_history)
        emp_thickness = emp_estimator.estimate(temp_history)
        
        assert abs(tr_thickness - emp_thickness) < 1.5

    def test_temperature_trend_calculation(self, estimator):
        """正常场景：温度趋势计算精度"""
        t0 = time.time()
        times = np.array([t0 + i * 10 for i in range(60)])
        values = np.array([-80.0 + i * 0.01 for i in range(60)])
        
        trend = estimator._calculate_trend(times, values)
        
        expected_trend = 0.01 * 360  # ℃/hour
        assert abs(trend - expected_trend) < 0.5


class TestDefrostOptimizer:
    """除霜优化器测试"""

    @pytest.fixture
    def config(self):
        return DefrostConfig(
            enabled=True,
            check_interval_seconds=300,
            frost_thickness_estimation={
                'method': 'thermal_resistance',
                'base_cold_trap_temp': -80.0,
                'max_frost_thickness_mm': 5.0,
                'calibration_factor': 1.2,
                'temp_window_minutes': 60,
            },
            optimization={
                'min_running_hours_before_defrost': 8.0,
                'allow_defrost_during_batches': False,
                'prefer_valley_electricity': True,
            },
            thresholds={
                'frost_thickness_trigger_mm': 3.0,
                'temp_trend_trigger': 0.5,
                'min_defrost_interval_hours': 4.0,
                'max_defrost_interval_hours': 24.0,
            },
            power_profile={
                'max_power_pct': 80.0,
                'min_power_pct': 20.0,
                'main_heating_power_pct': 80.0,
                'adaptive_power_enabled': True,
                'power_adjustment_factor': 0.1,
            },
        )

    @pytest.fixture
    def optimizer(self, config):
        config_dict = asdict(config)
        config_dict['energy_model'] = {
            'specific_energy_kwh_per_mm_frost': 0.5,
            'standby_power_kw': 2.0,
            'efficiency_coefficient': 0.85,
        }
        return DefrostOptimizer(config_dict)

    @pytest.fixture
    def state(self):
        state = DeviceDefrostState(device_id=1)
        state.last_defrost_time = time.time() - 12 * 3600
        return state

    def test_optimization_normal_defrost_needed(self, optimizer, state):
        """正常场景：结霜严重需要除霜"""
        t0 = time.time()
        for i in range(720):
            t = t0 - (720 - i) * 10
            temp = -60.0 + np.random.normal(0, 0.3)
            state.cold_trap_history.append((t, temp))
        
        need_defrost, interval, power, saving = optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=10, is_batch_running=False
        )
        
        assert need_defrost == True
        assert 4.0 <= interval <= 24.0
        assert 20.0 <= power <= 80.0
        assert saving >= 0.0

    def test_optimization_no_defrost_needed(self, optimizer, state):
        """正常场景：结霜轻微不需要除霜"""
        t0 = time.time()
        for i in range(100):
            t = t0 - (100 - i) * 10
            temp = -79.0 + np.random.normal(0, 0.1)
            state.cold_trap_history.append((t, temp))
        
        need_defrost, interval, power, saving = optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=10, is_batch_running=False
        )
        
        assert need_defrost == False
        assert interval > 12.0

    def test_energy_saving_calculation(self, optimizer, state):
        """正常场景：能耗降低效果验证"""
        t0 = time.time()
        for i in range(720):
            t = t0 - (720 - i) * 10
            temp = -60.0
            state.cold_trap_history.append((t, temp))
        
        need_defrost, interval, power, saving = optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=10, is_batch_running=False
        )
        
        frost_thickness = state.estimated_frost_thickness
        assert frost_thickness > 2.0
        assert saving > 0.0
        
        traditional_energy = frost_thickness * 0.5 * 0.8
        optimized_energy = frost_thickness * 0.5 * (power / 100.0)
        assert optimized_energy < traditional_energy

    def test_valley_electricity_preference(self, optimizer, state):
        """边界场景：谷电时段偏好验证"""
        t0 = time.time()
        for i in range(500):
            t = t0 - (500 - i) * 10
            temp = -70.0
            state.cold_trap_history.append((t, temp))
        
        _, peak_interval, _, _ = optimizer.optimize(
            state, electricity_price=1.2, hour_of_day=10, is_batch_running=False
        )
        
        _, valley_interval, _, _ = optimizer.optimize(
            state, electricity_price=0.4, hour_of_day=2, is_batch_running=False
        )
        
        assert valley_interval < peak_interval

    def test_batch_running_block(self, optimizer, state):
        """边界场景：批次运行中禁止除霜"""
        t0 = time.time()
        for i in range(720):
            t = t0 - (720 - i) * 10
            temp = -60.0
            state.cold_trap_history.append((t, temp))
        
        need_defrost_batch, _, _, _ = optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=10, is_batch_running=True
        )
        
        need_defrost_idle, _, _, _ = optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=10, is_batch_running=False
        )
        
        assert need_defrost_batch == False
        assert need_defrost_idle == True

    def test_min_interval_enforcement(self, optimizer, state):
        """异常场景：最短除霜间隔强制执行"""
        t0 = time.time()
        for i in range(720):
            t = t0 - (720 - i) * 10
            temp = -65.0
            state.cold_trap_history.append((t, temp))
        
        state.last_defrost_time = time.time() - 2 * 3600
        
        need_defrost, _, _, _ = optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=10, is_batch_running=False
        )
        
        assert need_defrost == False

    def test_max_interval_enforcement(self, optimizer, state):
        """异常场景：最长除霜间隔强制执行"""
        t0 = time.time()
        for i in range(100):
            t = t0 - (100 - i) * 10
            temp = -78.0
            state.cold_trap_history.append((t, temp))
        
        state.last_defrost_time = time.time() - 25 * 3600
        
        need_defrost, _, _, _ = optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=10, is_batch_running=False
        )
        
        assert need_defrost == True

    def test_adaptive_power_with_electricity_price(self, optimizer, state):
        """边界场景：自适应功率与电价的关系"""
        t0 = time.time()
        for i in range(500):
            t = t0 - (500 - i) * 10
            temp = -70.0
            state.cold_trap_history.append((t, temp))
        
        _, _, peak_power, _ = optimizer.optimize(
            state, electricity_price=1.2, hour_of_day=10, is_batch_running=False
        )
        
        _, _, valley_power, _ = optimizer.optimize(
            state, electricity_price=0.4, hour_of_day=2, is_batch_running=False
        )
        
        assert valley_power > peak_power

    def test_temperature_trend_trigger(self, optimizer, state):
        """边界场景：温度趋势快速上升触发除霜"""
        t0 = time.time()
        for i in range(200):
            t = t0 - (200 - i) * 10
            temp = -80.0 + i * 0.05
            state.cold_trap_history.append((t, temp))
        
        need_defrost, _, _, _ = optimizer.optimize(
            state, electricity_price=0.8, hour_of_day=10, is_batch_running=False
        )
        
        assert need_defrost == True


class TestDefrostCycleRationality:
    """除霜周期合理性测试"""

    @pytest.fixture
    def config(self):
        return DefrostConfig(
            enabled=True,
            check_interval_seconds=300,
            frost_thickness_estimation={
                'method': 'thermal_resistance',
                'base_cold_trap_temp': -80.0,
                'max_frost_thickness_mm': 5.0,
                'calibration_factor': 1.2,
                'temp_window_minutes': 60,
            },
            optimization={
                'min_running_hours_before_defrost': 8.0,
                'allow_defrost_during_batches': False,
                'prefer_valley_electricity': True,
            },
            thresholds={
                'frost_thickness_trigger_mm': 3.0,
                'temp_trend_trigger': 0.5,
                'min_defrost_interval_hours': 4.0,
                'max_defrost_interval_hours': 24.0,
            },
            power_profile={
                'max_power_pct': 80.0,
                'min_power_pct': 20.0,
                'main_heating_power_pct': 80.0,
                'adaptive_power_enabled': True,
                'power_adjustment_factor': 0.1,
            },
        )

    @pytest.fixture
    def optimizer(self, config):
        config_dict = asdict(config)
        config_dict['energy_model'] = {
            'specific_energy_kwh_per_mm_frost': 0.5,
            'standby_power_kw': 2.0,
            'efficiency_coefficient': 0.85,
        }
        return DefrostOptimizer(config_dict)

    def test_cycle_optimization_over_time(self, optimizer):
        """集成测试：模拟24小时的除霜周期优化"""
        state = DeviceDefrostState(device_id=1)
        state.last_defrost_time = time.time() - 8 * 3600
        
        defrost_events = []
        t0 = time.time()
        
        for hour in range(24):
            for minute in range(0, 60, 10):
                current_time = t0 + hour * 3600 + minute * 60
                
                frost_rate = 0.5
                elapsed_hours = (current_time - state.last_defrost_time) / 3600
                expected_temp = -80.0 + elapsed_hours * frost_rate * 2.0
                
                for _ in range(6):
                    t = current_time - _ * 10
                    state.cold_trap_history.append((t, expected_temp + np.random.normal(0, 0.2)))
                
                hour_of_day = hour
                price = 1.2 if 8 <= hour < 22 else 0.4
                
                need_defrost, interval, power, saving = optimizer.optimize(
                    state, electricity_price=price, hour_of_day=hour_of_day, is_batch_running=False
                )
                
                if need_defrost:
                    defrost_events.append({
                        'hour': hour,
                        'minute': minute,
                        'price': price,
                        'frost_thickness': state.estimated_frost_thickness,
                        'power': power,
                        'saving': saving,
                    })
                    state.last_defrost_time = current_time
                    state.cold_trap_history.clear()
        
        assert len(defrost_events) >= 1
        
        for event in defrost_events:
            assert 4.0 <= event['power'] <= 80.0
            assert event['frost_thickness'] >= 2.0
        
        valley_events = [e for e in defrost_events if e['price'] < 0.6]
        peak_events = [e for e in defrost_events if e['price'] >= 0.6]
        
        assert len(valley_events) >= len(peak_events)

    def test_power_profile_phases(self, optimizer):
        """正常场景：四阶段除霜功率曲线合理性"""
        state = DeviceDefrostState(device_id=1)
        state.defrost_in_progress = True
        state.defrost_phase = 'preheat'
        state.defrost_start_time = time.time()
        
        t0 = time.time()
        for i in range(100):
            t = t0 - (100 - i) * 10
            temp = -65.0
            state.cold_trap_history.append((t, temp))
        
        phases = ['preheat', 'main_heating', 'soak', 'cooldown']
        expected_powers = {
            'preheat': 30.0,
            'main_heating': 80.0,
            'soak': 50.0,
            'cooldown': 0.0,
        }
        
        for phase in phases:
            state.defrost_phase = phase
            need_defrost, interval, power, saving = optimizer.optimize(
                state, electricity_price=0.8, hour_of_day=10, is_batch_running=False
            )
            
            if phase == 'preheat':
                assert power <= expected_powers['main_heating']
            elif phase == 'main_heating':
                assert power >= expected_powers['preheat']
            elif phase == 'cooldown':
                assert power == 0 or power <= 20.0
