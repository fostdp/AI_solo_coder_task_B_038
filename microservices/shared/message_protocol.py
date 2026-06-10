"""
消息协议定义
微服务间通信的消息格式规范
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from uuid import UUID, uuid4


@dataclass
class MessageHeader:
    """消息头"""
    message_id: str
    message_type: str
    source_service: str
    target_service: Optional[str]
    timestamp: str
    version: str = "1.0"


@dataclass
class TelemetryData:
    """遥测数据消息"""
    device_id: int
    shelf_id: int
    timestamp: str
    temperatures: List[float]
    vacuum_levels: List[float]
    cold_trap_temp: float
    heating_powers: List[float]
    batch_id: Optional[str] = None
    cycle_id: Optional[int] = None
    data_quality: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ControlCommand:
    """控制命令消息"""
    device_id: int
    shelf_id: int
    timestamp: str
    auto_mode: bool
    power_adjustments: List[float]
    target_temp: Optional[float] = None
    batch_id: Optional[str] = None
    command_id: str = ""

    def __post_init__(self):
        if not self.command_id:
            self.command_id = str(uuid4())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ControlStatus:
    """控制状态消息"""
    device_id: int
    shelf_id: int
    timestamp: str
    auto_mode: bool
    current_powers: List[float]
    temperature_diff: float
    avg_temperature: float
    adjustments: List[float]
    batch_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PredictionResult:
    """预测结果消息"""
    device_id: int
    timestamp: str
    moisture_content: float
    moisture_confidence: float
    reconstitution_time: float
    reconstitution_confidence: float
    drying_rate: float
    is_qualified: bool
    moisture_threshold: float
    reconstitution_threshold: float
    formula_id: Optional[str] = None
    batch_id: Optional[str] = None
    drift_detected: bool = False
    adaptation_level: float = 0.0
    model_version: str = "2.0"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AlarmEvent:
    """告警事件消息"""
    alarm_id: str
    device_id: int
    shelf_id: Optional[int]
    timestamp: str
    alarm_type: str
    severity: str
    message: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AlarmAck:
    """告警确认消息"""
    alarm_id: str
    acknowledged_by: str
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConfigUpdate:
    """配置更新消息"""
    config_type: str  # control, prediction, alarm
    config_data: Dict
    source_service: str
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ServiceStatus:
    """服务状态消息"""
    service_id: str
    service_type: str
    status: str  # running, error, degraded
    timestamp: str
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ========== 新增消息类型 ==========

@dataclass
class EndpointDetection:
    """干燥终点判定结果"""
    device_id: int
    batch_id: str
    cycle_phase: str  # primary_drying, secondary_drying
    detection_method: str  # first_derivative, autoencoder, pressure_rise_test, combined
    endpoint_timestamp: str
    detection_confidence: float
    pressure_rise_delta: Optional[float] = None
    temp_inflection_point: Optional[float] = None
    temp_first_derivative: Optional[float] = None
    autoencoder_recon_error: Optional[float] = None
    cycle_duration_hours: Optional[float] = None
    estimated_energy_saving: Optional[float] = None
    is_accepted: bool = True
    shelf_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PressureRiseTest:
    """压力升测试结果"""
    device_id: int
    batch_id: Optional[str]
    test_start_time: str
    test_end_time: Optional[str]
    initial_pressure_pa: float
    final_pressure_pa: float
    pressure_rise_pa_per_min: float
    test_duration_seconds: int
    is_endpoint_detected: bool = False
    detection_confidence: Optional[float] = None
    test_status: str = "completed"  # requested, in_progress, completed

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DefrostOptimization:
    """冷阱除霜优化建议"""
    device_id: int
    batch_id: Optional[str]
    timestamp: str
    estimated_frost_thickness_mm: float
    cold_trap_temp_avg: float
    cold_trap_temp_trend: float
    recommended_defrost_interval_hours: float
    recommended_heating_power_pct: float
    estimated_energy_saving: Optional[float] = None
    defrost_status: str = "pending"  # pending, in_progress, completed, skipped

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DefrostCommand:
    """除霜执行命令"""
    device_id: int
    timestamp: str
    command: str  # start, stop, cancel
    heating_power_pct: float
    max_duration_minutes: Optional[int] = None
    batch_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DefrostStatus:
    """除霜状态更新"""
    device_id: int
    timestamp: str
    status: str  # idle, defrosting, defrost_complete, error
    progress_pct: Optional[float] = None
    current_temp: Optional[float] = None
    target_temp: Optional[float] = None
    energy_consumed_kwh: Optional[float] = None
    batch_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FleetSchedule:
    """群控调度计划"""
    schedule_id: str
    schedule_date: str
    total_required_batches: int
    estimated_energy_cost: float
    optimized_energy_saving: float
    solver_status: str  # optimal, suboptimal, timeout
    details: List[Dict]  # FleetScheduleDetail列表
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FleetCommand:
    """群控启停命令"""
    device_id: int
    command: str  # start_batch, stop_batch, pause, resume
    timestamp: str
    batch_id: Optional[str] = None
    formula_id: Optional[str] = None
    freeze_profile_id: Optional[int] = None
    priority: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FleetStatus:
    """群控状态更新"""
    device_id: int
    timestamp: str
    batch_id: Optional[str]
    batch_status: str  # idle, running, paused, completed
    current_phase: Optional[str] = None  # freezing, primary_drying, secondary_drying, defrosting
    phase_progress_pct: Optional[float] = None
    estimated_completion_time: Optional[str] = None
    current_power_kw: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DefectDetection:
    """制品缺陷检测结果"""
    device_id: int
    batch_id: str
    timestamp: str
    image_path: str
    image_hash: str
    defect_type: str  # collapse, atrophy, cracking, normal
    defect_severity: str  # low, medium, high
    confidence: float
    bbox_x: Optional[int] = None
    bbox_y: Optional[int] = None
    bbox_width: Optional[int] = None
    bbox_height: Optional[int] = None
    shelf_id: Optional[int] = None
    vial_position: Optional[str] = None
    is_manual_reviewed: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ImageUpload:
    """图像上传通知"""
    device_id: int
    batch_id: str
    timestamp: str
    image_path: str
    image_hash: str
    shelf_id: Optional[int] = None
    vial_position: Optional[str] = None
    file_size_bytes: Optional[int] = None
    content_type: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BatchRecord:
    """批次记录更新"""
    device_id: int
    batch_id: str
    timestamp: str
    update_type: str  # start, primary_endpoint, secondary_endpoint, complete, quality, defect
    freeze_profile_id: Optional[int] = None
    formula_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    primary_drying_endpoint: Optional[str] = None
    secondary_drying_endpoint: Optional[str] = None
    avg_moisture_content: Optional[float] = None
    avg_reconstitution_time: Optional[float] = None
    defect_rate: Optional[float] = None
    quality_score: Optional[float] = None
    batch_status: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class MessageFactory:
    """消息工厂"""

    @staticmethod
    def create_telemetry(data: TelemetryData, source_service: str) -> Dict:
        """创建遥测消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="telemetry",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": data.to_dict()
        }

    @staticmethod
    def create_control_command(cmd: ControlCommand, source_service: str) -> Dict:
        """创建控制命令消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="control_command",
            source_service=source_service,
            target_service="profinet-driver",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": cmd.to_dict()
        }

    @staticmethod
    def create_prediction(result: PredictionResult, source_service: str) -> Dict:
        """创建预测结果消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="prediction",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": result.to_dict()
        }

    @staticmethod
    def create_alarm(alarm: AlarmEvent, source_service: str) -> Dict:
        """创建告警消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="alarm",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": alarm.to_dict()
        }

    @staticmethod
    def create_config_update(config_type: str, config_data: Dict, source_service: str) -> Dict:
        """创建配置更新消息"""
        config = ConfigUpdate(
            config_type=config_type,
            config_data=config_data,
            source_service=source_service,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="config",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": config.to_dict()
        }

    @staticmethod
    def create_service_status(service_id: str, service_type: str, status: str,
                               metrics: Dict = None, error_message: str = None) -> Dict:
        """创建服务状态消息"""
        service_status = ServiceStatus(
            service_id=service_id,
            service_type=service_type,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics=metrics or {},
            error_message=error_message
        )
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="status",
            source_service=service_id,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": service_status.to_dict()
        }

    @staticmethod
    def create_endpoint_detection(result: EndpointDetection, source_service: str) -> Dict:
        """创建终点判定消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="endpoint",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": result.to_dict()
        }

    @staticmethod
    def create_pressure_rise_test(result: PressureRiseTest, source_service: str) -> Dict:
        """创建压力升测试消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="pressure_rise_test",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": result.to_dict()
        }

    @staticmethod
    def create_defrost_optimization(result: DefrostOptimization, source_service: str) -> Dict:
        """创建除霜优化消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="defrost_optimization",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": result.to_dict()
        }

    @staticmethod
    def create_defrost_command(cmd: DefrostCommand, source_service: str) -> Dict:
        """创建除霜命令消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="defrost_command",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": cmd.to_dict()
        }

    @staticmethod
    def create_defrost_status(status: DefrostStatus, source_service: str) -> Dict:
        """创建除霜状态消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="defrost_status",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": status.to_dict()
        }

    @staticmethod
    def create_fleet_schedule(schedule: FleetSchedule, source_service: str) -> Dict:
        """创建群控调度消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="fleet_schedule",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": schedule.to_dict()
        }

    @staticmethod
    def create_fleet_command(cmd: FleetCommand, source_service: str) -> Dict:
        """创建群控命令消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="fleet_command",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": cmd.to_dict()
        }

    @staticmethod
    def create_fleet_status(status: FleetStatus, source_service: str) -> Dict:
        """创建群控状态消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="fleet_status",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": status.to_dict()
        }

    @staticmethod
    def create_defect_detection(result: DefectDetection, source_service: str) -> Dict:
        """创建缺陷检测消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="defect_detection",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": result.to_dict()
        }

    @staticmethod
    def create_image_upload(upload: ImageUpload, source_service: str) -> Dict:
        """创建图像上传消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="image_upload",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": upload.to_dict()
        }

    @staticmethod
    def create_batch_record(record: BatchRecord, source_service: str) -> Dict:
        """创建批次记录消息"""
        header = MessageHeader(
            message_id=str(uuid4()),
            message_type="batch_record",
            source_service=source_service,
            target_service=None,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        return {
            "header": asdict(header),
            "payload": record.to_dict()
        }


def serialize_message(message: Dict) -> str:
    """序列化消息为JSON字符串"""
    return json.dumps(message, ensure_ascii=False)


def deserialize_message(message_str: str) -> Dict:
    """反序列化JSON字符串为消息"""
    return json.loads(message_str)


def validate_message(message: Dict, expected_type: str) -> bool:
    """验证消息类型"""
    try:
        return message["header"]["message_type"] == expected_type
    except (KeyError, TypeError):
        return False


def extract_payload(message: Dict) -> Dict:
    """提取消息载荷"""
    return message.get("payload", {})
