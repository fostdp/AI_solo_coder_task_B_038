"""
数据类型定义
包含群控调度算法所需的数据类
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class SolverConfig:
    """
    求解器配置
    纯算法相关的配置，不包含微服务特定内容
    """
    optimization: Dict[str, Any] = field(default_factory=dict)
    electricity_price: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    freeze_profiles: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DeviceState:
    """
    设备状态
    包含设备当前运行状态和历史统计信息
    """
    device_id: int
    status: str = "idle"  # idle, running, paused, maintenance, defrosting
    current_batch_id: Optional[str] = None
    current_formula_id: Optional[str] = None
    current_profile_id: Optional[int] = None
    current_phase: Optional[str] = None
    phase_start_time: Optional[float] = None
    estimated_completion_time: Optional[float] = None
    batches_completed: int = 0
    total_run_hours: float = 0.0
    last_maintenance_time: Optional[float] = None
    priority: int = 1


@dataclass
class UrgentBatch:
    """
    紧急插单批次
    包含紧急批次的所有属性和约束
    """
    batch_id: str
    formula_id: str
    priority: int
    deadline_hours: float  # 要求在多少小时内完成
    requested_start_time: Optional[float] = None  # 请求的最早开始时间
    energy_kwh: float = 0.0
    min_start_delay: float = 0.0


@dataclass
class ScheduledBatch:
    """
    调度的批次
    包含批次的完整调度信息
    """
    device_id: int
    batch_id: str
    formula_id: str
    profile_id: int
    start_time: float
    end_time: float
    energy_kwh: float
    priority: int = 0
    is_urgent: bool = False
    original_schedule: Optional['ScheduledBatch'] = None  # 被替换的原调度（用于回滚）
    rescheduled: bool = False  # 是否被重新调度过


@dataclass
class TimeSlot:
    """
    时间段
    包含时间段的起止时间和电价信息
    """
    start_hour: int
    end_hour: int
    price: float
    is_valley: bool
