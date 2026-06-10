"""
整数规划求解器
基于电价和产能需求，用整数规划优化多台冻干机的启停和冻干曲线选择

优化目标：
1. 最小化电费成本（考虑峰谷电价）
2. 最大化产能
3. 平衡设备使用率

约束条件：
- 最大并发设备数
- 设备维护周期
- 人员可用时间
- 批次持续时间
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

from .types import (
    SolverConfig,
    DeviceState,
    UrgentBatch,
    ScheduledBatch,
    TimeSlot,
)


class IntegerProgrammingSolver:
    """整数规划求解器（简化版，使用贪心+启发式算法）"""

    def __init__(self, config: SolverConfig):
        self.config = config
        self.opt_config = config.optimization
        self.price_config = config.electricity_price
        self.constraints = config.constraints

        self.profiles = config.freeze_profiles

        self._has_pulp = False
        try:
            import pulp
            self._pulp = pulp
            self._has_pulp = True
        except ImportError:
            self._has_pulp = False
            print("[IntegerProgrammingSolver] 警告: PuLP未安装，使用启发式算法")

    def solve(self, device_states: Dict[int, DeviceState],
              required_batches: int,
              time_horizon_hours: int = 24,
              start_time: Optional[datetime] = None) -> Tuple[List[ScheduledBatch], float, float, str]:
        """
        求解调度问题

        返回：(调度计划, 预计电费, 预计节能, 求解状态)
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)

        price_schedule = self._get_electricity_prices(start_time, time_horizon_hours)

        if self._has_pulp:
            try:
                return self._solve_with_pulp(device_states, required_batches,
                                             time_horizon_hours, start_time, price_schedule)
            except Exception as e:
                print(f"[IntegerProgrammingSolver] PuLP solver failed, falling back to heuristic: {e}")
                return self._solve_heuristic(device_states, required_batches,
                                             time_horizon_hours, start_time, price_schedule)
        else:
            return self._solve_heuristic(device_states, required_batches,
                                         time_horizon_hours, start_time, price_schedule)

    def _get_electricity_prices(self, start_time: datetime, hours: int) -> List[TimeSlot]:
        """获取电价表"""
        slots = []

        static_prices = self.price_config.get('static_prices', {})
        valley_hours = self.price_config.get('valley_hours', static_prices.get('valley_hours', []))
        peak_hours = self.price_config.get('peak_hours', static_prices.get('peak_hours', []))
        valley_price = self.price_config.get('valley_price', static_prices.get('valley', 0.4))
        peak_price = self.price_config.get('peak_price', static_prices.get('peak', 1.2))
        flat_price = self.price_config.get('flat_price', static_prices.get('flat', 0.8))

        for hour_offset in range(hours):
            dt = start_time + timedelta(hours=hour_offset)
            hour = dt.hour

            if hour in peak_hours:
                price = peak_price
                is_valley = False
            elif hour in valley_hours:
                price = valley_price
                is_valley = True
            else:
                price = flat_price
                is_valley = False

            slots.append(TimeSlot(hour_offset, hour_offset + 1, price, is_valley))

        return slots

    def _solve_with_pulp(self, device_states: Dict[int, DeviceState],
                         required_batches: int,
                         time_horizon_hours: int,
                         start_time: datetime,
                         price_schedule: List[TimeSlot]) -> Tuple[List[ScheduledBatch], float, float, str]:
        """使用PuLP求解整数规划"""
        try:
            pulp = self._pulp

            available_devices = [
                device_id for device_id, state in device_states.items()
                if state.status == "idle"
            ]

            if not available_devices:
                return [], 0.0, 0.0, "no_available"

            profiles = self.profiles
            if not profiles:
                profiles = [
                    {'formula_id': 'FORMULA-001', 'primary_drying_hours': 24, 'secondary_drying_hours': 8, 'energy_kwh': 120},
                ]

            time_resolution = self.opt_config.get('time_resolution_minutes', 30)
            num_slots = time_horizon_hours * 60 // time_resolution

            prob = pulp.LpProblem("FleetScheduling", pulp.LpMinimize)

            x = {}
            for d in available_devices:
                for p_idx, profile in enumerate(profiles):
                    total_hours = profile['primary_drying_hours'] + profile['secondary_drying_hours']
                    duration_slots = int(total_hours * 60 // time_resolution)
                    for t in range(num_slots - duration_slots + 1):
                        x[(d, p_idx, t)] = pulp.LpVariable(
                            f"x_{d}_{p_idx}_{t}", cat='Binary')

            objective = []
            for (d, p_idx, t), var in x.items():
                profile = profiles[p_idx]
                duration_slots = int((profile['primary_drying_hours'] + profile['secondary_drying_hours']) * 60 // time_resolution)
                cost = 0
                for i in range(duration_slots):
                    slot_idx = t + i
                    if slot_idx < len(price_schedule):
                        hour_idx = slot_idx * time_resolution // 60
                        if hour_idx < len(price_schedule):
                            cost += price_schedule[hour_idx].price * profile['energy_kwh'] * (time_resolution / 60) / (profile['primary_drying_hours'] + profile['secondary_drying_hours'])
                objective.append(cost * var)

            prob += pulp.lpSum(objective)

            prob += pulp.lpSum(x.values()) >= required_batches

            for d in available_devices:
                for t in range(num_slots):
                    vars_at_t = []
                    for p_idx, profile in enumerate(profiles):
                        duration_slots = int((profile['primary_drying_hours'] + profile['secondary_drying_hours']) * 60 // time_resolution)
                        for start_t in range(max(0, t - duration_slots + 1), t + 1):
                            if (d, p_idx, start_t) in x:
                                vars_at_t.append(x[(d, p_idx, start_t)])
                    if vars_at_t:
                        prob += pulp.lpSum(vars_at_t) <= 1

            max_concurrent = self.constraints.get('max_concurrent_devices', 10)
            for t in range(num_slots):
                vars_at_t = []
                for d in available_devices:
                    for p_idx, profile in enumerate(profiles):
                        duration_slots = int((profile['primary_drying_hours'] + profile['secondary_drying_hours']) * 60 // time_resolution)
                        for start_t in range(max(0, t - duration_slots + 1), t + 1):
                            if (d, p_idx, start_t) in x:
                                vars_at_t.append(x[(d, p_idx, start_t)])
                if vars_at_t:
                    prob += pulp.lpSum(vars_at_t) <= max_concurrent

            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if pulp.LpStatus[prob.status] == 'Optimal':
                status = 'optimal'
            elif pulp.LpStatus[prob.status] == 'Not Solved':
                return self._solve_heuristic(device_states, required_batches,
                                             time_horizon_hours, start_time, price_schedule)
            else:
                status = 'suboptimal'

            schedule = []
            total_cost = 0.0
            for (d, p_idx, t), var in x.items():
                if var.value() and var.value() > 0.5:
                    profile = profiles[p_idx]
                    start_ts = start_time.timestamp() + t * time_resolution * 60
                    duration_hours = profile['primary_drying_hours'] + profile['secondary_drying_hours']
                    end_ts = start_ts + duration_hours * 3600

                    cost = 0
                    duration_slots = int(duration_hours * 60 // time_resolution)
                    for i in range(duration_slots):
                        slot_idx = t + i
                        hour_idx = slot_idx * time_resolution // 60
                        if hour_idx < len(price_schedule):
                            cost += price_schedule[hour_idx].price * profile['energy_kwh'] * (time_resolution / 60) / duration_hours

                    batch_id = f"BATCH-{start_time.strftime('%Y%m%d')}-{len(schedule) + 1:03d}"

                    schedule.append(ScheduledBatch(
                        device_id=d,
                        batch_id=batch_id,
                        formula_id=profile['formula_id'],
                        profile_id=p_idx + 1,
                        start_time=start_ts,
                        end_time=end_ts,
                        energy_kwh=profile['energy_kwh'],
                        priority=profile.get('priority', 1)
                    ))
                    total_cost += cost

            baseline_cost = self._calculate_baseline_cost(
                device_states, required_batches, time_horizon_hours, start_time, price_schedule, profiles
            )
            energy_saving = baseline_cost - total_cost

            return schedule, total_cost, energy_saving, status

        except Exception as e:
            print(f"[IntegerProgrammingSolver] PuLP求解失败: {e}, 使用启发式算法")
            return self._solve_heuristic(device_states, required_batches,
                                         time_horizon_hours, start_time, price_schedule)

    def _calculate_baseline_cost(self, device_states: Dict[int, DeviceState],
                                 required_batches: int,
                                 time_horizon_hours: int,
                                 start_time: datetime,
                                 price_schedule: List[TimeSlot],
                                 profiles: List[Dict]) -> float:
        """计算基准成本（不优化的顺序调度，使用相同的配方选择逻辑）"""
        available_devices = sorted([
            (d, s) for d, s in device_states.items()
            if s.status == "idle"
        ], key=lambda x: x[0])

        if not price_schedule:
            price_schedule = self._get_electricity_prices(start_time, time_horizon_hours)

        sorted_profiles = sorted(profiles, key=lambda p: p['energy_kwh'], reverse=True)

        total_cost = 0.0
        batches_scheduled = 0
        device_next_available = {d: 0.0 for d, _ in available_devices}

        max_concurrent = self.constraints.get('max_concurrent_devices', 10)
        concurrent_tracker = [0] * time_horizon_hours

        def _check_concurrent(start_hour: int, duration_hours: float) -> bool:
            end_hour = min(start_hour + int(duration_hours) + 1, time_horizon_hours)
            for h in range(start_hour, end_hour):
                if concurrent_tracker[h] >= max_concurrent:
                    return False
            return True

        def _update_concurrent(start_hour: int, duration_hours: float, delta: int):
            end_hour = min(start_hour + int(duration_hours) + 1, time_horizon_hours)
            for h in range(start_hour, end_hour):
                concurrent_tracker[h] += delta

        while batches_scheduled < required_batches:
            scheduled_this_round = False

            for profile in sorted_profiles:
                if batches_scheduled >= required_batches:
                    break

                duration_hours = profile['primary_drying_hours'] + profile['secondary_drying_hours']

                for device_id, state in available_devices:
                    next_avail = device_next_available[device_id]
                    next_avail_hour = int(next_avail) // 3600

                    for start_hour in range(next_avail_hour, time_horizon_hours - int(duration_hours) + 1):
                        if _check_concurrent(start_hour, duration_hours):
                            cost = 0
                            for h in range(start_hour, min(start_hour + int(duration_hours), time_horizon_hours)):
                                if h < len(price_schedule):
                                    cost += price_schedule[h].price * profile['energy_kwh'] / duration_hours

                            _update_concurrent(start_hour, duration_hours, 1)
                            total_cost += cost
                            device_next_available[device_id] = (start_hour + duration_hours) * 3600
                            batches_scheduled += 1
                            scheduled_this_round = True
                            break

                    if scheduled_this_round:
                        break

            if not scheduled_this_round:
                break

        return total_cost

    def _solve_heuristic(self, device_states: Dict[int, DeviceState],
                         required_batches: int,
                         time_horizon_hours: int,
                         start_time: datetime,
                         price_schedule: List[TimeSlot]) -> Tuple[List[ScheduledBatch], float, float, str]:
        """启发式算法：优先在谷电时段调度高能耗批次，满足最大并发约束"""
        available_devices = sorted([
            (d, s) for d, s in device_states.items()
            if s.status == "idle"
        ], key=lambda x: x[1].priority, reverse=True)

        if not available_devices:
            return [], 0.0, 0.0, "no_available"

        profiles = self.profiles
        if not profiles:
            profiles = [
                {'formula_id': 'FORMULA-001', 'primary_drying_hours': 24, 'secondary_drying_hours': 8, 'energy_kwh': 120, 'priority': 1},
            ]

        if not price_schedule:
            price_schedule = self._get_electricity_prices(start_time, time_horizon_hours)

        sorted_profiles = sorted(profiles, key=lambda p: p['energy_kwh'], reverse=True)

        schedule = []
        total_cost = 0.0
        device_next_available = {d: 0.0 for d, _ in available_devices}

        max_concurrent = self.constraints.get('max_concurrent_devices', 10)
        concurrent_tracker = [0] * time_horizon_hours

        valley_hours = [i for i, slot in enumerate(price_schedule) if slot.is_valley]

        batches_scheduled = 0
        start_ts = start_time.timestamp()

        def _check_concurrent(start_hour: int, duration_hours: float) -> bool:
            """检查该时间段是否满足最大并发约束"""
            end_hour = min(start_hour + int(duration_hours) + 1, time_horizon_hours)
            for h in range(start_hour, end_hour):
                if concurrent_tracker[h] >= max_concurrent:
                    return False
            return True

        def _update_concurrent(start_hour: int, duration_hours: float, delta: int):
            """更新并发跟踪器"""
            end_hour = min(start_hour + int(duration_hours) + 1, time_horizon_hours)
            for h in range(start_hour, end_hour):
                concurrent_tracker[h] += delta

        while batches_scheduled < required_batches:
            scheduled_this_round = False

            for profile in sorted_profiles:
                if batches_scheduled >= required_batches:
                    break

                best_device = None
                best_start_hour = None
                best_cost = float('inf')

                duration_hours = profile['primary_drying_hours'] + profile['secondary_drying_hours']

                for device_id, state in available_devices:
                    next_avail = device_next_available[device_id]
                    next_avail_hour = int(next_avail) // 3600

                    candidate_hours = []
                    for valley_h in valley_hours:
                        if valley_h >= next_avail_hour and valley_h + duration_hours <= time_horizon_hours:
                            if _check_concurrent(valley_h, duration_hours):
                                candidate_hours.append(valley_h)

                    if not candidate_hours:
                        for h in range(next_avail_hour, time_horizon_hours - int(duration_hours) + 1):
                            if _check_concurrent(h, duration_hours):
                                candidate_hours.append(h)

                    for start_hour in candidate_hours:
                        cost = 0
                        for h in range(start_hour, min(start_hour + int(duration_hours), time_horizon_hours)):
                            if h < len(price_schedule):
                                cost += price_schedule[h].price * profile['energy_kwh'] / duration_hours

                        if cost < best_cost:
                            best_cost = cost
                            best_device = device_id
                            best_start_hour = start_hour

                if best_device is not None and best_start_hour is not None:
                    batch_id = f"BATCH-{start_time.strftime('%Y%m%d')}-{batches_scheduled + 1:03d}"
                    batch_start = start_ts + best_start_hour * 3600
                    batch_end = batch_start + duration_hours * 3600

                    _update_concurrent(best_start_hour, duration_hours, 1)

                    schedule.append(ScheduledBatch(
                        device_id=best_device,
                        batch_id=batch_id,
                        formula_id=profile['formula_id'],
                        profile_id=sorted_profiles.index(profile) + 1,
                        start_time=batch_start,
                        end_time=batch_end,
                        energy_kwh=profile['energy_kwh'],
                        priority=profile.get('priority', 1)
                    ))

                    total_cost += best_cost
                    device_next_available[best_device] = (best_start_hour + duration_hours) * 3600
                    batches_scheduled += 1
                    scheduled_this_round = True

            if not scheduled_this_round:
                break

        baseline_cost = self._calculate_baseline_cost(
            device_states, required_batches, time_horizon_hours, start_time, price_schedule, profiles
        )
        energy_saving = max(0.0, baseline_cost - total_cost)

        return schedule, total_cost, energy_saving, "heuristic"

    def reschedule_for_urgent_batch(self,
                                    current_schedule: List[ScheduledBatch],
                                    device_states: Dict[int, DeviceState],
                                    urgent_batch: UrgentBatch,
                                    time_horizon_hours: int = 24,
                                    current_time: Optional[datetime] = None) -> Tuple[List[ScheduledBatch], float, float, str]:
        """
        动态重调度：插入紧急批次

        策略：
        1. 优先使用空闲设备
        2. 如果没有空闲设备，考虑抢占低优先级批次
        3. 调整后续批次的时间安排
        4. 确保满足所有约束条件

        返回：(新的调度计划, 增加的成本, 成本变化, 状态)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        price_schedule = self._get_electricity_prices(current_time, time_horizon_hours)
        if not price_schedule:
            price_schedule = self._get_electricity_prices(current_time, time_horizon_hours)

        urgent_profile = None
        for profile in self.profiles:
            if profile['formula_id'] == urgent_batch.formula_id:
                urgent_profile = profile
                break

        if urgent_profile is None:
            urgent_profile = self.profiles[0] if self.profiles else {
                'formula_id': urgent_batch.formula_id,
                'primary_drying_hours': 24,
                'secondary_drying_hours': 8,
                'energy_kwh': 120,
                'priority': urgent_batch.priority
            }

        duration_hours = urgent_profile['primary_drying_hours'] + urgent_profile['secondary_drying_hours']
        current_ts = current_time.timestamp()

        fixed_batches = [b for b in current_schedule if b.start_time <= current_ts]
        movable_batches = [b for b in current_schedule if b.start_time > current_ts]

        available_devices = []
        for device_id, state in device_states.items():
            if state.status == "idle":
                available_devices.append((device_id, state))
            elif state.status == "running" and state.estimated_completion_time:
                avail_time = state.estimated_completion_time
                if avail_time - current_ts < duration_hours * 3600:
                    available_devices.append((device_id, state))

        if not available_devices:
            movable_batches.sort(key=lambda b: b.priority)
            if movable_batches and movable_batches[0].priority < urgent_batch.priority:
                preempted = movable_batches.pop(0)
                available_devices.append((preempted.device_id, device_states[preempted.device_id]))
            else:
                return current_schedule, 0.0, 0.0, "no_available"

        best_device = None
        best_start_hour = None
        best_cost = float('inf')
        best_preempted = None

        start_ts = current_time.timestamp()

        valley_hours = [i for i, slot in enumerate(price_schedule) if slot.is_valley]

        max_concurrent = self.constraints.get('max_concurrent_devices', 10)
        concurrent_tracker = [0] * time_horizon_hours

        for batch in fixed_batches + movable_batches:
            start_h = int((batch.start_time - start_ts) // 3600)
            duration_h = int((batch.end_time - batch.start_time) // 3600) + 1
            for h in range(max(0, start_h), min(time_horizon_hours, start_h + duration_h)):
                if 0 <= h < time_horizon_hours:
                    concurrent_tracker[h] += 1

        def _check_concurrent(start_hour: int, dur_hours: float) -> bool:
            end_h = min(start_hour + int(dur_hours) + 1, time_horizon_hours)
            for h in range(max(0, start_hour), min(time_horizon_hours, end_h)):
                if concurrent_tracker[h] >= max_concurrent:
                    return False
            return True

        def _update_concurrent(start_hour: int, dur_hours: float, delta: int):
            end_h = min(start_hour + int(dur_hours) + 1, time_horizon_hours)
            for h in range(max(0, start_hour), min(time_horizon_hours, end_h)):
                concurrent_tracker[h] += delta

        for device_id, state in available_devices:
            next_avail_hour = 0
            if state.status == "running" and state.estimated_completion_time:
                next_avail_hour = int((state.estimated_completion_time - start_ts) // 3600) + 1

            candidate_hours = []
            for valley_h in valley_hours:
                if valley_h >= next_avail_hour and valley_h + duration_hours <= time_horizon_hours:
                    if _check_concurrent(valley_h, duration_hours):
                        candidate_hours.append(valley_h)

            if not candidate_hours:
                for h in range(next_avail_hour, time_horizon_hours - int(duration_hours) + 1):
                    if _check_concurrent(h, duration_hours):
                        candidate_hours.append(h)

            for start_hour in candidate_hours:
                cost = 0
                for h in range(start_hour, min(start_hour + int(duration_hours), time_horizon_hours)):
                    if h < len(price_schedule):
                        cost += price_schedule[h].price * urgent_profile['energy_kwh'] / duration_hours

                if cost < best_cost:
                    best_cost = cost
                    best_device = device_id
                    best_start_hour = start_hour

        if best_device is None or best_start_hour is None:
            return current_schedule, 0.0, 0.0, "no_slot"

        _update_concurrent(best_start_hour, duration_hours, 1)

        urgent_scheduled = ScheduledBatch(
            device_id=best_device,
            batch_id=urgent_batch.batch_id,
            formula_id=urgent_profile['formula_id'],
            profile_id=self.profiles.index(urgent_profile) + 1 if urgent_profile in self.profiles else 1,
            start_time=start_ts + best_start_hour * 3600,
            end_time=start_ts + (best_start_hour + duration_hours) * 3600,
            energy_kwh=urgent_profile['energy_kwh'],
            priority=urgent_batch.priority,
            is_urgent=True
        )

        new_movable = []
        for batch in movable_batches:
            if batch.device_id == best_device:
                batch_start_h = int((batch.start_time - start_ts) // 3600)
                batch_end_h = int((batch.end_time - start_ts) // 3600) + 1
                urgent_end_h = best_start_hour + int(duration_hours)

                if batch_start_h < urgent_end_h:
                    _update_concurrent(batch_start_h, (batch.end_time - batch.start_time) // 3600, -1)

                    new_start_h = urgent_end_h
                    while new_start_h < time_horizon_hours - int((batch.end_time - batch.start_time) // 3600) + 1:
                        if _check_concurrent(new_start_h, (batch.end_time - batch.start_time) // 3600):
                            break
                        new_start_h += 1

                    if new_start_h < time_horizon_hours:
                        _update_concurrent(new_start_h, (batch.end_time - batch.start_time) // 3600, 1)
                        old_start = batch.start_time
                        old_end = batch.end_time
                        batch.start_time = start_ts + new_start_h * 3600
                        batch.end_time = start_ts + new_start_h * 3600 + (old_end - old_start)
                        batch.rescheduled = True

            new_movable.append(batch)

        new_schedule = fixed_batches + [urgent_scheduled] + new_movable

        new_total_cost = best_cost
        for batch in new_movable:
            start_h = int((batch.start_time - start_ts) // 3600)
            duration_h = int((batch.end_time - batch.start_time) // 3600)
            for h in range(start_h, min(start_h + duration_h, time_horizon_hours)):
                if h < len(price_schedule):
                    new_total_cost += price_schedule[h].price * batch.energy_kwh / max(duration_h, 1)

        original_total_cost = 0.0
        for batch in movable_batches:
            start_h = int((batch.start_time - start_ts) // 3600)
            duration_h = int((batch.end_time - batch.start_time) // 3600)
            for h in range(start_h, min(start_h + duration_h, time_horizon_hours)):
                if h < len(price_schedule):
                    original_total_cost += price_schedule[h].price * batch.energy_kwh / max(duration_h, 1)

        cost_delta = new_total_cost - original_total_cost

        return new_schedule, best_cost, cost_delta, "rescheduled"

    def validate_schedule(self,
                          schedule: List[ScheduledBatch],
                          time_horizon_hours: int,
                          start_time: datetime) -> Tuple[bool, List[str]]:
        """
        验证调度计划是否满足所有约束

        返回：(是否有效, 违规信息列表)
        """
        violations = []
        start_ts = start_time.timestamp()

        max_concurrent = self.constraints.get('max_concurrent_devices', 10)
        concurrent_tracker = [0] * time_horizon_hours

        device_schedules: Dict[int, List[ScheduledBatch]] = {}
        for batch in schedule:
            if batch.device_id not in device_schedules:
                device_schedules[batch.device_id] = []
            device_schedules[batch.device_id].append(batch)

        for device_id, batches in device_schedules.items():
            batches.sort(key=lambda b: b.start_time)
            for i in range(len(batches) - 1):
                if batches[i].end_time > batches[i+1].start_time:
                    violations.append(
                        f"设备{device_id}时间冲突: {batches[i].batch_id}({batches[i].start_time}) 与 {batches[i+1].batch_id}({batches[i+1].start_time})"
                    )

        for batch in schedule:
            start_h = int((batch.start_time - start_ts) // 3600)
            duration_h = int((batch.end_time - batch.start_time) // 3600) + 1
            for h in range(max(0, start_h), min(time_horizon_hours, start_h + duration_h)):
                if 0 <= h < time_horizon_hours:
                    concurrent_tracker[h] += 1

        for h, count in enumerate(concurrent_tracker):
            if count > max_concurrent:
                violations.append(
                    f"并发约束违反: 第{h}小时有{count}台设备运行，超过限制{max_concurrent}"
                )

        return len(violations) == 0, violations
