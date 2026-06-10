from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass, field


@dataclass
class DeviceDefrostState:
    """设备除霜状态"""
    device_id: int
    batch_id: Optional[str] = None
    
    # 多传感器冷阱温度历史（每个传感器独立历史）
    # sensor_id -> deque of (timestamp, temperature)
    multi_sensor_history: Dict[int, Deque[Tuple[float, float]]] = field(
        default_factory=lambda: {
            i: deque(maxlen=720) for i in range(1, 6)  # 支持5个传感器
        }
    )
    
    # 兼容单传感器接口（主传感器历史）
    cold_trap_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=720)  # 2小时（10秒间隔）
    )
    
    # 传感器元数据
    sensor_positions: Dict[int, str] = field(default_factory=lambda: {
        1: "inlet",      # 入口（结霜最严重）
        2: "coil_1",     # 盘管1
        3: "coil_2",     # 盘管2
        4: "coil_3",     # 盘管3
        5: "outlet"      # 出口（结霜最轻）
    })
    
    sensor_weights: Dict[int, float] = field(default_factory=lambda: {
        1: 0.35,  # 入口权重最高
        2: 0.20,
        3: 0.20,
        4: 0.15,
        5: 0.10   # 出口权重最低
    })
    
    sensor_health: Dict[int, float] = field(default_factory=lambda: {
        i: 1.0 for i in range(1, 6)  # 0-1，1表示传感器正常
    })
    
    # 融合后的温度历史
    fused_temperature_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=720)
    )
    
    # 结霜厚度分布估算（每个传感器位置）
    frost_thickness_distribution: Dict[int, float] = field(default_factory=lambda: {
        i: 0.0 for i in range(1, 6)
    })
    
    # 结霜估算
    estimated_frost_thickness: float = 0.0
    last_defrost_time: Optional[float] = None
    cumulative_running_hours: float = 0.0
    
    # 除霜状态
    defrost_in_progress: bool = False
    defrost_phase: Optional[str] = None  # preheat, main_heating, soak, cooldown
    defrost_start_time: Optional[float] = None
    defrost_power_pct: float = 0.0
    target_defrost_temp: float = 10.0  # 除霜目标温度
    current_cold_trap_temp: float = -80.0
    
    # 调度状态
    next_scheduled_defrost: Optional[float] = None
    recommended_interval_hours: float = 12.0
