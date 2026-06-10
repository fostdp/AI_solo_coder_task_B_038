"""
共享模块
包含Redis客户端、消息协议、配置加载器
"""

from .redis_channels import CHANNELS, MESSAGE_TYPES, SERVICE_IDS
from .redis_client import RedisClientBase, MicroserviceBase, RedisConfig
from .message_protocol import (
    MessageFactory, MessageHeader,
    TelemetryData, ControlCommand, ControlStatus,
    PredictionResult, AlarmEvent, AlarmAck,
    ConfigUpdate, ServiceStatus,
    EndpointDetection, PressureRiseTest,
    DefrostOptimization, DefrostCommand, DefrostStatus,
    FleetSchedule, FleetCommand, FleetStatus,
    DefectDetection, ImageUpload, BatchRecord,
    serialize_message, deserialize_message,
    validate_message, extract_payload
)
from .config_loader import (
    ConfigLoader, ControlConfig, ModelConfig, AlarmConfig,
    EndpointConfig, DefrostConfig, FleetConfig, DefectConfig,
    config_loader
)

__all__ = [
    'CHANNELS', 'MESSAGE_TYPES', 'SERVICE_IDS',
    'RedisClientBase', 'MicroserviceBase', 'RedisConfig',
    'MessageFactory', 'MessageHeader',
    'TelemetryData', 'ControlCommand', 'ControlStatus',
    'PredictionResult', 'AlarmEvent', 'AlarmAck',
    'ConfigUpdate', 'ServiceStatus',
    'EndpointDetection', 'PressureRiseTest',
    'DefrostOptimization', 'DefrostCommand', 'DefrostStatus',
    'FleetSchedule', 'FleetCommand', 'FleetStatus',
    'DefectDetection', 'ImageUpload', 'BatchRecord',
    'serialize_message', 'deserialize_message',
    'validate_message', 'extract_payload',
    'ConfigLoader', 'ControlConfig', 'ModelConfig', 'AlarmConfig',
    'EndpointConfig', 'DefrostConfig', 'FleetConfig', 'DefectConfig',
    'config_loader'
]
