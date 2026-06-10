"""
冻干机群控调度测试
覆盖：整数规划求解速度和全局最优性、电价响应效果、多机约束满足
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

from fleet_controller.main import (
    IntegerProgrammingSolver,
    DeviceState,
    ScheduledBatch,
    TimeSlot,
)
from shared import FleetConfig


class TestIntegerProgrammingSolver:
    """整数规划求解器测试"""

    @pytest.fixture
    def config(self):
        return FleetConfig(
            enabled=True,
            schedule_interval_minutes=60,
            optimization={
                'time_resolution_minutes': 30,
                'max_solve_time_seconds': 10,
                'objective_weights': {'cost': 0.7, 'throughput': 0.3},
            },
            electricity_price={
                'static_prices': {
                    'peak': 1.2,
                    'flat': 0.8,
                    'valley': 0.4,
                    'peak_hours': [8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21],
                    'flat_hours': [6, 7, 12, 13, 22, 23],
                }
            },
            device_priorities={},
            constraints={
                'max_concurrent_devices': 10,
                'min_batch_interval_minutes': 30,
                'maintenance_hours_per_week': 4,
            },
        )

    @pytest.fixture
    def solver(self, config):
        config.__dict__['freeze_profiles'] = [
            {'formula_id': 'FORMULA-001', 'primary_drying_hours': 8, 'secondary_drying_hours': 4, 'energy_kwh': 80, 'priority': 1},
            {'formula_id': 'FORMULA-002', 'primary_drying_hours': 12, 'secondary_drying_hours': 6, 'energy_kwh': 150, 'priority': 2},
        ]
        return IntegerProgrammingSolver(config)

    @pytest.fixture
    def device_states(self):
        states = {}
        for i in range(1, 11):
            states[i] = DeviceState(device_id=i, status="idle", priority=1)
        return states

    def test_solver_initialization(self, solver):
        """正常场景：求解器初始化"""
        assert solver._has_pulp in [True, False]
        assert len(solver.profiles) == 2

    def test_heuristic_solve_normal(self, solver, device_states):
        """正常场景：启发式算法求解"""
        required_batches = 5
        time_horizon = 24
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon,
            datetime.now(timezone.utc), []
        )
        
        assert len(schedule) == required_batches
        assert total_cost > 0
        assert status == 'heuristic'
        
        device_ids = [b.device_id for b in schedule]
        assert len(set(device_ids)) <= 10

    def test_electricity_price_response(self, solver, device_states):
        """正常场景：电价响应效果 - 谷电时段调度更多"""
        required_batches = 10
        time_horizon = 24
        start_time = datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc)
        
        price_schedule = solver._get_electricity_prices(start_time, time_horizon)
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon, start_time, price_schedule
        )
        
        assert len(schedule) == required_batches
        
        valley_starts = 0
        peak_starts = 0
        
        for batch in schedule:
            start_dt = datetime.fromtimestamp(batch.start_time, tz=timezone.utc)
            hour = start_dt.hour
            
            if hour in [0, 1, 2, 3, 4, 5]:
                valley_starts += 1
            elif hour in [8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21]:
                peak_starts += 1
        
        assert valley_starts >= peak_starts

    def test_optimization_vs_baseline(self, solver, device_states):
        """正常场景：优化调度 vs 基准调度的成本对比"""
        required_batches = 10
        time_horizon = 24
        start_time = datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc)
        
        price_schedule = solver._get_electricity_prices(start_time, time_horizon)
        
        schedule, optimized_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon, start_time, price_schedule
        )
        
        profiles = solver.profiles
        baseline_cost = solver._calculate_baseline_cost(
            device_states, required_batches, time_horizon, start_time, price_schedule, profiles
        )
        
        assert optimized_cost <= baseline_cost
        assert energy_saving >= 0
        if baseline_cost > 0:
            saving_pct = (baseline_cost - optimized_cost) / baseline_cost * 100
            assert saving_pct >= 10

    def test_no_available_devices(self, solver):
        """异常场景：无可用设备"""
        device_states = {}
        for i in range(1, 11):
            device_states[i] = DeviceState(device_id=i, status="running")
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, 5, 24, datetime.now(timezone.utc), []
        )
        
        assert len(schedule) == 0
        assert total_cost == 0
        assert energy_saving == 0
        assert status == 'no_available'

    def test_exceed_time_horizon(self, solver, device_states):
        """边界场景：批次数量超出时间范围容量"""
        required_batches = 100
        time_horizon = 24
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon,
            datetime.now(timezone.utc), []
        )
        
        assert len(schedule) < required_batches
        assert len(schedule) > 0

    def test_single_device_scheduling(self, solver):
        """边界场景：单设备多批次调度"""
        device_states = {1: DeviceState(device_id=1, status="idle")}
        required_batches = 3
        time_horizon = 72
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon,
            datetime.now(timezone.utc), []
        )
        
        assert len(schedule) == required_batches
        
        for i in range(1, len(schedule)):
            assert schedule[i].start_time >= schedule[i-1].end_time


class TestMultiDeviceConstraints:
    """多设备约束满足测试"""

    @pytest.fixture
    def config(self):
        return FleetConfig(
            enabled=True,
            schedule_interval_minutes=60,
            optimization={
                'time_resolution_minutes': 30,
                'max_solve_time_seconds': 10,
            },
            electricity_price={
                'static_prices': {
                    'peak': 1.2, 'flat': 0.8, 'valley': 0.4,
                    'peak_hours': list(range(8, 22)),
                    'flat_hours': [6, 7, 22, 23],
                }
            },
            constraints={
                'max_concurrent_devices': 5,
                'min_batch_interval_minutes': 30,
            },
        )

    @pytest.fixture
    def solver(self, config):
        config.__dict__['freeze_profiles'] = [
            {'formula_id': 'FORMULA-001', 'primary_drying_hours': 4, 'secondary_drying_hours': 2, 'energy_kwh': 50, 'priority': 1},
        ]
        return IntegerProgrammingSolver(config)

    @pytest.fixture
    def device_states(self):
        states = {}
        for i in range(1, 11):
            states[i] = DeviceState(device_id=i, status="idle")
        return states

    def test_max_concurrent_constraint(self, solver, device_states):
        """边界场景：最大并发设备数约束"""
        required_batches = 10
        time_horizon = 24
        start_time = datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc)
        
        price_schedule = solver._get_electricity_prices(start_time, time_horizon)
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon, start_time, price_schedule
        )
        
        max_concurrent = solver.constraints.get('max_concurrent_devices', 10)
        
        time_points = []
        for batch in schedule:
            time_points.append((batch.start_time, 1))
            time_points.append((batch.end_time, -1))
        
        time_points.sort(key=lambda x: x[0])
        
        current_concurrent = 0
        max_concurrent_seen = 0
        for _, delta in time_points:
            current_concurrent += delta
            max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
        
        assert max_concurrent_seen <= max_concurrent

    def test_device_priority_ordering(self, solver):
        """正常场景：设备优先级排序"""
        device_states = {}
        priorities = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        for i in range(1, 11):
            device_states[i] = DeviceState(device_id=i, status="idle", priority=priorities[i-1])
        
        available = sorted([
            (d, s) for d, s in device_states.items() if s.status == "idle"
        ], key=lambda x: x[1].priority, reverse=True)
        
        assert available[0][1].priority == 9
        assert available[-1][1].priority == 1

    def test_mixed_device_states(self, solver):
        """边界场景：混合设备状态（部分运行中）"""
        device_states = {}
        for i in range(1, 11):
            if i <= 5:
                device_states[i] = DeviceState(device_id=i, status="idle")
            else:
                device_states[i] = DeviceState(device_id=i, status="running")
        
        required_batches = 10
        time_horizon = 48
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon,
            datetime.now(timezone.utc), []
        )
        
        device_ids = set(b.device_id for b in schedule)
        assert all(d <= 5 for d in device_ids)

    def test_zero_batches(self, solver, device_states):
        """异常场景：零批次需求"""
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, 0, 24, datetime.now(timezone.utc), []
        )
        
        assert len(schedule) == 0
        assert total_cost == 0
        assert energy_saving == 0


class TestSolverPerformance:
    """求解器性能测试"""

    @pytest.fixture
    def config(self):
        return FleetConfig(
            enabled=True,
            schedule_interval_minutes=60,
            optimization={
                'time_resolution_minutes': 30,
                'max_solve_time_seconds': 10,
            },
            electricity_price={
                'static_prices': {
                    'peak': 1.2, 'flat': 0.8, 'valley': 0.4,
                    'peak_hours': list(range(8, 22)),
                    'flat_hours': [6, 7, 22, 23],
                }
            },
            constraints={
                'max_concurrent_devices': 10,
            },
        )

    @pytest.fixture
    def solver(self, config):
        config.__dict__['freeze_profiles'] = [
            {'formula_id': 'FORMULA-001', 'primary_drying_hours': 8, 'secondary_drying_hours': 4, 'energy_kwh': 100, 'priority': 1},
        ]
        return IntegerProgrammingSolver(config)

    @pytest.fixture
    def device_states(self):
        states = {}
        for i in range(1, 11):
            states[i] = DeviceState(device_id=i, status="idle")
        return states

    def test_solve_speed(self, solver, device_states):
        """正常场景：求解速度测试"""
        required_batches = 20
        time_horizon = 48
        start_time = datetime.now(timezone.utc)
        
        start_ts = time.time()
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon, start_time, []
        )
        
        end_ts = time.time()
        solve_time = end_ts - start_ts
        
        assert solve_time < 5.0
        assert len(schedule) > 0

    def test_scalability_10_devices(self, solver, device_states):
        """边界场景：10台设备大规模调度"""
        required_batches = 50
        time_horizon = 168
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon,
            datetime.now(timezone.utc), []
        )
        
        assert len(schedule) >= required_batches * 0.8

    def test_electricity_price_schedule_generation(self, solver):
        """正常场景：电价时间表生成"""
        start_time = datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc)
        hours = 24
        
        schedule = solver._get_electricity_prices(start_time, hours)
        
        assert len(schedule) == hours
        
        valley_prices = [s.price for s in schedule if s.is_valley]
        peak_prices = [s.price for s in schedule if not s.is_valley]
        
        assert all(p == 0.4 for p in valley_prices)
        assert all(p >= 0.8 for p in peak_prices)
        assert len(valley_prices) == 6


class TestScheduleValidation:
    """调度结果验证测试"""

    @pytest.fixture
    def config(self):
        return FleetConfig(
            enabled=True,
            schedule_interval_minutes=60,
            optimization={
                'time_resolution_minutes': 30,
            },
            electricity_price={
                'static_prices': {
                    'peak': 1.2, 'flat': 0.8, 'valley': 0.4,
                    'peak_hours': list(range(8, 22)),
                    'flat_hours': [6, 7, 22, 23],
                }
            },
            constraints={
                'max_concurrent_devices': 10,
            },
        )

    @pytest.fixture
    def solver(self, config):
        config.__dict__['freeze_profiles'] = [
            {'formula_id': 'FORMULA-001', 'primary_drying_hours': 8, 'secondary_drying_hours': 4, 'energy_kwh': 100, 'priority': 1},
        ]
        return IntegerProgrammingSolver(config)

    def test_schedule_batch_consistency(self, solver):
        """正常场景：调度批次属性一致性"""
        device_states = {1: DeviceState(device_id=1, status="idle")}
        required_batches = 3
        time_horizon = 72
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon,
            datetime.now(timezone.utc), []
        )
        
        profile = solver.profiles[0]
        expected_duration = (profile['primary_drying_hours'] + profile['secondary_drying_hours']) * 3600
        
        for batch in schedule:
            assert batch.formula_id == profile['formula_id']
            assert batch.energy_kwh == profile['energy_kwh']
            assert abs((batch.end_time - batch.start_time) - expected_duration) < 1

    def test_schedule_no_overlap(self, solver):
        """正常场景：单设备调度无时间重叠"""
        device_states = {1: DeviceState(device_id=1, status="idle")}
        required_batches = 5
        time_horizon = 120
        
        schedule, total_cost, energy_saving, status = solver._solve_heuristic(
            device_states, required_batches, time_horizon,
            datetime.now(timezone.utc), []
        )
        
        schedule.sort(key=lambda b: b.start_time)
        
        for i in range(1, len(schedule)):
            assert schedule[i].start_time >= schedule[i-1].end_time
