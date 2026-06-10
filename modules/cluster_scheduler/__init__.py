"""
群控调度算法模块
提供基于整数规划的冻干机群控调度核心算法

主要组件：
- 数据类：SolverConfig, DeviceState, UrgentBatch, ScheduledBatch, TimeSlot
- 求解器：IntegerProgrammingSolver（支持PuLP和启发式算法）
- 工作进程：SchedulingWorker（异步调度计算）

使用示例：
    from modules.cluster_scheduler import (
        SolverConfig, DeviceState, IntegerProgrammingSolver, SchedulingWorker
    )

    # 同步使用
    config = SolverConfig(...)
    solver = IntegerProgrammingSolver(config)
    schedule, cost, saving, status = solver.solve(device_states, 10)

    # 异步使用
    with SchedulingWorker(config) as worker:
        task_id = worker.submit_solve(device_states, 10)
        result = worker.get_result(task_id, timeout=5.0)
"""

from .types import (
    SolverConfig,
    DeviceState,
    UrgentBatch,
    ScheduledBatch,
    TimeSlot,
)

from .solver import IntegerProgrammingSolver
from .worker import SchedulingWorker

__all__ = [
    'SolverConfig',
    'DeviceState',
    'UrgentBatch',
    'ScheduledBatch',
    'TimeSlot',
    'IntegerProgrammingSolver',
    'SchedulingWorker',
]

__version__ = '1.0.0'
