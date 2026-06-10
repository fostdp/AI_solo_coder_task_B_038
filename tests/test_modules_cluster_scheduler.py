import sys
sys.path.insert(0, 'd:/SOLO-2/AI_solo_coder_task_B_038')

import unittest
import numpy as np
from datetime import datetime, timezone, timedelta
from modules.cluster_scheduler import (
    SolverConfig,
    DeviceState,
    UrgentBatch,
    ScheduledBatch,
    TimeSlot,
    IntegerProgrammingSolver,
    SchedulingWorker,
)


class TestSolverConfig(unittest.TestCase):

    def test_construction_with_dict_fields(self):
        config = SolverConfig(
            optimization={'max_concurrent_devices': 5},
            electricity_price={'valley_hours': [0, 1, 2, 3, 4, 5], 'peak_hours': [8, 9, 10, 11]},
            constraints={'max_concurrent_devices': 5},
            freeze_profiles=[
                {'formula_id': 'FORMULA-001', 'primary_drying_hours': 24, 'secondary_drying_hours': 8, 'energy_kwh': 120},
            ],
        )
        self.assertEqual(config.optimization['max_concurrent_devices'], 5)
        self.assertEqual(len(config.electricity_price['valley_hours']), 6)
        self.assertEqual(len(config.freeze_profiles), 1)

    def test_default_construction(self):
        config = SolverConfig()
        self.assertIsInstance(config.optimization, dict)
        self.assertIsInstance(config.electricity_price, dict)
        self.assertIsInstance(config.constraints, dict)
        self.assertIsInstance(config.freeze_profiles, list)

    def test_electricity_price_config(self):
        config = SolverConfig(
            electricity_price={
                'valley_hours': [0, 1, 2, 3, 4, 5],
                'peak_hours': [8, 9, 10, 11, 16, 17, 18, 19],
                'valley_price': 0.4,
                'peak_price': 1.2,
                'flat_price': 0.8,
            },
        )
        self.assertAlmostEqual(config.electricity_price['valley_price'], 0.4)
        self.assertAlmostEqual(config.electricity_price['peak_price'], 1.2)


class TestIntegerProgrammingSolver(unittest.TestCase):

    def setUp(self):
        self.config = SolverConfig(
            optimization={'time_resolution_minutes': 30},
            electricity_price={
                'valley_hours': [0, 1, 2, 3, 4, 5],
                'peak_hours': [8, 9, 10, 11, 16, 17, 18, 19],
                'valley_price': 0.4,
                'peak_price': 1.2,
                'flat_price': 0.8,
            },
            constraints={'max_concurrent_devices': 10},
            freeze_profiles=[
                {
                    'formula_id': 'FORMULA-001',
                    'primary_drying_hours': 24,
                    'secondary_drying_hours': 8,
                    'energy_kwh': 120,
                    'priority': 1,
                },
            ],
        )
        self.solver = IntegerProgrammingSolver(self.config)

    def test_solve_with_idle_devices(self):
        device_states = {
            1: DeviceState(device_id=1, status='idle'),
            2: DeviceState(device_id=2, status='idle'),
            3: DeviceState(device_id=3, status='idle'),
        }
        start_time = datetime.now(timezone.utc)
        schedule, cost, saving, status = self.solver.solve(
            device_states, required_batches=2, time_horizon_hours=48, start_time=start_time
        )
        self.assertIsInstance(schedule, list)
        self.assertGreater(len(schedule), 0)
        self.assertIsInstance(cost, float)
        self.assertIsInstance(saving, float)
        self.assertIn(status, ['optimal', 'suboptimal', 'heuristic'])
        for batch in schedule:
            self.assertIsInstance(batch, ScheduledBatch)
            self.assertIsInstance(batch.device_id, int)
            self.assertIsInstance(batch.batch_id, str)

    def test_solve_with_no_available_devices(self):
        device_states = {
            1: DeviceState(device_id=1, status='running'),
            2: DeviceState(device_id=2, status='maintenance'),
        }
        schedule, cost, saving, status = self.solver.solve(
            device_states, required_batches=1
        )
        self.assertEqual(len(schedule), 0)
        self.assertEqual(status, 'no_available')

    def test_reschedule_for_urgent_batch(self):
        device_states = {
            1: DeviceState(device_id=1, status='idle'),
            2: DeviceState(device_id=2, status='idle'),
        }
        start_time = datetime.now(timezone.utc)
        schedule, cost, saving, status = self.solver.solve(
            device_states, required_batches=2, start_time=start_time
        )
        urgent = UrgentBatch(
            batch_id='URGENT-001',
            formula_id='FORMULA-001',
            priority=10,
            deadline_hours=36.0,
        )
        new_schedule, urgent_cost, cost_delta, resched_status = self.solver.reschedule_for_urgent_batch(
            schedule, device_states, urgent, time_horizon_hours=48, current_time=start_time
        )
        self.assertIsInstance(new_schedule, list)
        self.assertIn(resched_status, ['rescheduled', 'no_available', 'no_slot'])

    def test_validate_schedule_valid(self):
        start_time = datetime.now(timezone.utc)
        start_ts = start_time.timestamp()
        schedule = [
            ScheduledBatch(
                device_id=1, batch_id='B-001', formula_id='F-001', profile_id=1,
                start_time=start_ts, end_time=start_ts + 32 * 3600,
                energy_kwh=120, priority=1,
            ),
            ScheduledBatch(
                device_id=2, batch_id='B-002', formula_id='F-001', profile_id=1,
                start_time=start_ts + 3600, end_time=start_ts + 33 * 3600,
                energy_kwh=120, priority=1,
            ),
        ]
        is_valid, violations = self.solver.validate_schedule(schedule, 48, start_time)
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(violations, list)

    def test_validate_schedule_conflict(self):
        start_time = datetime.now(timezone.utc)
        start_ts = start_time.timestamp()
        schedule = [
            ScheduledBatch(
                device_id=1, batch_id='B-001', formula_id='F-001', profile_id=1,
                start_time=start_ts, end_time=start_ts + 32 * 3600,
                energy_kwh=120, priority=1,
            ),
            ScheduledBatch(
                device_id=1, batch_id='B-002', formula_id='F-001', profile_id=1,
                start_time=start_ts + 10 * 3600, end_time=start_ts + 42 * 3600,
                energy_kwh=120, priority=1,
            ),
        ]
        is_valid, violations = self.solver.validate_schedule(schedule, 48, start_time)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)


class TestSchedulingWorker(unittest.TestCase):

    def setUp(self):
        self.config = SolverConfig(
            optimization={},
            electricity_price={
                'valley_hours': [0, 1, 2, 3, 4, 5],
                'peak_hours': [8, 9, 10, 11],
                'valley_price': 0.4,
                'peak_price': 1.2,
                'flat_price': 0.8,
            },
            constraints={'max_concurrent_devices': 10},
            freeze_profiles=[
                {
                    'formula_id': 'FORMULA-001',
                    'primary_drying_hours': 24,
                    'secondary_drying_hours': 8,
                    'energy_kwh': 120,
                    'priority': 1,
                },
            ],
        )

    def test_start_stop_lifecycle(self):
        worker = SchedulingWorker(self.config)
        worker.start()
        self.assertTrue(worker.is_alive())
        worker.stop(timeout=5.0)
        self.assertFalse(worker.is_alive())

    def test_submit_solve_task(self):
        worker = SchedulingWorker(self.config)
        worker.start()
        try:
            device_states = {
                1: DeviceState(device_id=1, status='idle'),
            }
            task_id = worker.submit_solve(
                device_states, required_batches=1, time_horizon_hours=48
            )
            self.assertIsInstance(task_id, str)
            self.assertIn(task_id, worker._active_tasks)
            result = worker.get_result(task_id, timeout=10.0)
            self.assertIsNotNone(result)
            self.assertTrue(result['success'])
            self.assertIn('schedule', result['result'])
        finally:
            worker.stop(timeout=5.0)

    def test_submit_validate_task(self):
        worker = SchedulingWorker(self.config)
        worker.start()
        try:
            start_time = datetime.now(timezone.utc)
            start_ts = start_time.timestamp()
            schedule = [
                ScheduledBatch(
                    device_id=1, batch_id='B-001', formula_id='F-001', profile_id=1,
                    start_time=start_ts, end_time=start_ts + 32 * 3600,
                    energy_kwh=120, priority=1,
                ),
            ]
            task_id = worker.submit_validate(schedule, 48, start_time)
            self.assertIsInstance(task_id, str)
            result = worker.get_result(task_id, timeout=10.0)
            self.assertIsNotNone(result)
            self.assertTrue(result['success'])
            self.assertIn('is_valid', result['result'])
        finally:
            worker.stop(timeout=5.0)

    def test_context_manager(self):
        with SchedulingWorker(self.config) as worker:
            self.assertTrue(worker.is_alive())
        self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
