"""
配置加载器
从YAML文件加载控制参数、模型参数、告警阈值
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


@dataclass
class ControlConfig:
    fuzzy_control: Dict[str, Any] = field(default_factory=dict)
    ilc_control: Dict[str, Any] = field(default_factory=dict)
    power_allocation: Dict[str, Any] = field(default_factory=dict)
    temperature_limits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    pls_model: Dict[str, Any] = field(default_factory=dict)
    transfer_learning: Dict[str, Any] = field(default_factory=dict)
    adaptive_update: Dict[str, Any] = field(default_factory=dict)
    concept_drift: Dict[str, Any] = field(default_factory=dict)
    formulas: list = field(default_factory=list)


@dataclass
class AlarmConfig:
    global_config: Dict[str, Any] = field(default_factory=dict)
    temperature: Dict[str, Any] = field(default_factory=dict)
    vacuum: Dict[str, Any] = field(default_factory=dict)
    cold_trap: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    severity_levels: Dict[str, Any] = field(default_factory=dict)
    mqtt_publisher: Dict[str, Any] = field(default_factory=dict)
    notification_channels: list = field(default_factory=list)
    auto_suppression: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointConfig:
    """干燥终点判定配置"""
    enabled: bool = True
    detection_interval_seconds: int = 60
    first_derivative: Dict[str, Any] = field(default_factory=dict)
    autoencoder: Dict[str, Any] = field(default_factory=dict)
    pressure_rise_test: Dict[str, Any] = field(default_factory=dict)
    combined_decision: Dict[str, Any] = field(default_factory=dict)
    phase_thresholds: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefrostConfig:
    """冷阱除霜优化配置"""
    enabled: bool = True
    check_interval_seconds: int = 300
    frost_thickness_estimation: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    power_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FleetConfig:
    """群控调度配置"""
    enabled: bool = True
    schedule_interval_minutes: int = 60
    optimization: Dict[str, Any] = field(default_factory=dict)
    electricity_price: Dict[str, Any] = field(default_factory=dict)
    device_priorities: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefectConfig:
    """缺陷检测配置"""
    enabled: bool = True
    cnn_model: Dict[str, Any] = field(default_factory=dict)
    image_preprocessing: Dict[str, Any] = field(default_factory=dict)
    defect_types: Dict[str, Any] = field(default_factory=dict)
    confidence_threshold: float = 0.8
    auto_review: bool = False


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self._control_config: ControlConfig = None
        self._model_config: ModelConfig = None
        self._alarm_config: AlarmConfig = None
        self._endpoint_config: EndpointConfig = None
        self._defrost_config: DefrostConfig = None
        self._fleet_config: FleetConfig = None
        self._defect_config: DefectConfig = None

    def load_control_config(self) -> ControlConfig:
        """加载控制参数"""
        if self._control_config is None:
            file_path = self.config_dir / "control_params.yaml"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self._control_config = ControlConfig(
                fuzzy_control=data.get('fuzzy_control', {}),
                ilc_control=data.get('ilc_control', {}),
                power_allocation=data.get('power_allocation', {}),
                temperature_limits=data.get('temperature_limits', {})
            )
        return self._control_config

    def load_model_config(self) -> ModelConfig:
        """加载模型参数"""
        if self._model_config is None:
            file_path = self.config_dir / "model_params.yaml"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self._model_config = ModelConfig(
                pls_model=data.get('pls_model', {}),
                transfer_learning=data.get('transfer_learning', {}),
                adaptive_update=data.get('adaptive_update', {}),
                concept_drift=data.get('concept_drift', {}),
                formulas=data.get('formulas', [])
            )
        return self._model_config

    def load_alarm_config(self) -> AlarmConfig:
        """加载告警阈值"""
        if self._alarm_config is None:
            file_path = self.config_dir / "alarm_thresholds.yaml"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self._alarm_config = AlarmConfig(
                global_config=data.get('global', {}),
                temperature=data.get('temperature', {}),
                vacuum=data.get('vacuum', {}),
                cold_trap=data.get('cold_trap', {}),
                quality=data.get('quality', {}),
                severity_levels=data.get('severity_levels', {}),
                mqtt_publisher=data.get('mqtt_publisher', {}),
                notification_channels=data.get('notification_channels', []),
                auto_suppression=data.get('auto_suppression', {})
            )
        return self._alarm_config

    def load_endpoint_config(self) -> EndpointConfig:
        """加载终点判定配置"""
        if self._endpoint_config is None:
            file_path = self.config_dir / "endpoint_params.yaml"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                self._endpoint_config = EndpointConfig(
                    enabled=data.get('enabled', True),
                    detection_interval_seconds=data.get('detection_interval_seconds', 60),
                    first_derivative=data.get('first_derivative', {}),
                    autoencoder=data.get('autoencoder', {}),
                    pressure_rise_test=data.get('pressure_rise_test', {}),
                    combined_decision=data.get('combined_decision', {}),
                    phase_thresholds=data.get('phase_thresholds', {})
                )
            else:
                self._endpoint_config = EndpointConfig()
        return self._endpoint_config

    def load_defrost_config(self) -> DefrostConfig:
        """加载除霜优化配置"""
        if self._defrost_config is None:
            file_path = self.config_dir / "defrost_params.yaml"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                self._defrost_config = DefrostConfig(
                    enabled=data.get('enabled', True),
                    check_interval_seconds=data.get('check_interval_seconds', 300),
                    frost_thickness_estimation=data.get('frost_thickness_estimation', {}),
                    optimization=data.get('optimization', {}),
                    thresholds=data.get('thresholds', {}),
                    power_profile=data.get('power_profile', {})
                )
            else:
                self._defrost_config = DefrostConfig()
        return self._defrost_config

    def load_fleet_config(self) -> FleetConfig:
        """加载群控调度配置"""
        if self._fleet_config is None:
            file_path = self.config_dir / "fleet_params.yaml"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                self._fleet_config = FleetConfig(
                    enabled=data.get('enabled', True),
                    schedule_interval_minutes=data.get('schedule_interval_minutes', 60),
                    optimization=data.get('optimization', {}),
                    electricity_price=data.get('electricity_price', {}),
                    device_priorities=data.get('device_priorities', {}),
                    constraints=data.get('constraints', {})
                )
            else:
                self._fleet_config = FleetConfig()
        return self._fleet_config

    def load_defect_config(self) -> DefectConfig:
        """加载缺陷检测配置"""
        if self._defect_config is None:
            file_path = self.config_dir / "defect_params.yaml"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                self._defect_config = DefectConfig(
                    enabled=data.get('enabled', True),
                    cnn_model=data.get('cnn_model', {}),
                    image_preprocessing=data.get('image_preprocessing', {}),
                    defect_types=data.get('defect_types', {}),
                    confidence_threshold=data.get('confidence_threshold', 0.8),
                    auto_review=data.get('auto_review', False)
                )
            else:
                self._defect_config = DefectConfig()
        return self._defect_config

    def reload_all(self):
        """重新加载所有配置"""
        self._control_config = None
        self._model_config = None
        self._alarm_config = None
        self._endpoint_config = None
        self._defrost_config = None
        self._fleet_config = None
        self._defect_config = None
        return (
            self.load_control_config(), self.load_model_config(), self.load_alarm_config(),
            self.load_endpoint_config(), self.load_defrost_config(),
            self.load_fleet_config(), self.load_defect_config()
        )

    def get(self, config_type: str, key_path: str, default: Any = None) -> Any:
        """获取配置值"""
        parts = key_path.split('.')
        config = None

        if config_type == 'control':
            config = self.load_control_config()
        elif config_type == 'model':
            config = self.load_model_config()
        elif config_type == 'alarm':
            config = self.load_alarm_config()
        elif config_type == 'endpoint':
            config = self.load_endpoint_config()
        elif config_type == 'defrost':
            config = self.load_defrost_config()
        elif config_type == 'fleet':
            config = self.load_fleet_config()
        elif config_type == 'defect':
            config = self.load_defect_config()
        else:
            return default

        value = config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        return value


# 全局配置加载器实例
config_loader = ConfigLoader()
