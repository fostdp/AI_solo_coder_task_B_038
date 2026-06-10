"""
Redis Pub/Sub 通道定义
微服务间通信协议
"""

# Redis通道定义
CHANNELS = {
    # 遥测数据通道 - profinet_driver -> 所有订阅者
    'TELEMETRY_RAW': 'telemetry:raw',           # 原始遥测数据
    'TELEMETRY_PROCESSED': 'telemetry:processed',  # 处理后的遥测数据

    # 控制命令通道 - temp_controller -> profinet_driver
    'CONTROL_COMMAND': 'control:command',       # 加热功率调整命令
    'CONTROL_STATUS': 'control:status',         # 控制状态更新

    # 预测结果通道 - quality_predictor -> alarm_publisher / api_gateway
    'PREDICTION_RESULT': 'prediction:result',   # 质量预测结果
    'PREDICTION_CONFIG': 'prediction:config',   # 预测配置更新

    # 告警通道 - alarm_publisher -> api_gateway / mqtt
    'ALARM_EVENT': 'alarm:event',               # 告警事件
    'ALARM_ACK': 'alarm:acknowledge',           # 告警确认
    'ALARM_CONFIG': 'alarm:config',             # 告警配置更新

    # 系统配置通道
    'CONFIG_UPDATE': 'config:update',           # 配置更新通知
    'SYSTEM_STATUS': 'system:status',           # 系统健康状态

    # 数据库写入通道
    'DB_WRITE': 'db:write',                     # 批量写入请求

    # ========== 新增功能通道 ==========
    # 终点判定通道 - endpoint_detector -> 所有订阅者
    'ENDPOINT_DETECTION': 'endpoint:detection',  # 干燥终点判定结果
    'PRESSURE_RISE_TEST': 'endpoint:prt',        # 压力升测试命令/结果

    # 冷阱除霜优化通道 - defrost_optimizer -> 相关服务
    'DEFROST_OPTIMIZATION': 'defrost:optimization',  # 除霜优化建议
    'DEFROST_COMMAND': 'defrost:command',            # 除霜执行命令
    'DEFROST_STATUS': 'defrost:status',              # 除霜状态更新

    # 群控调度通道 - fleet_controller -> 相关服务
    'FLEET_SCHEDULE': 'fleet:schedule',       # 群控调度计划
    'FLEET_COMMAND': 'fleet:command',         # 群控启停命令
    'FLEET_STATUS': 'fleet:status',           # 群控状态更新

    # 缺陷检测通道 - defect_detector -> 相关服务
    'DEFECT_DETECTION': 'defect:detection',   # 缺陷检测结果
    'IMAGE_UPLOAD': 'defect:image_upload',    # 图像上传通知
}

# 消息类型定义
MESSAGE_TYPES = {
    'TELEMETRY': 'telemetry',
    'CONTROL_COMMAND': 'control_command',
    'CONTROL_STATUS': 'control_status',
    'PREDICTION': 'prediction',
    'ALARM': 'alarm',
    'ALARM_ACK': 'alarm_ack',
    'CONFIG': 'config',
    'STATUS': 'status',
    'DB_BATCH': 'db_batch',
    # 新增消息类型
    'ENDPOINT': 'endpoint',
    'PRESSURE_RISE_TEST': 'pressure_rise_test',
    'DEFROST_OPTIMIZATION': 'defrost_optimization',
    'DEFROST_COMMAND': 'defrost_command',
    'DEFROST_STATUS': 'defrost_status',
    'FLEET_SCHEDULE': 'fleet_schedule',
    'FLEET_COMMAND': 'fleet_command',
    'FLEET_STATUS': 'fleet_status',
    'DEFECT_DETECTION': 'defect_detection',
    'IMAGE_UPLOAD': 'image_upload',
    'BATCH_RECORD': 'batch_record',
}

# 服务ID
SERVICE_IDS = {
    'PROFINET_DRIVER': 'profinet-driver',
    'TEMP_CONTROLLER': 'temp-controller',
    'QUALITY_PREDICTOR': 'quality-predictor',
    'ALARM_PUBLISHER': 'alarm-publisher',
    'API_GATEWAY': 'api-gateway',
    'DB_WRITER': 'db-writer',
    # 新增服务ID
    'ENDPOINT_DETECTOR': 'endpoint-detector',
    'DEFROST_OPTIMIZER': 'defrost-optimizer',
    'FLEET_CONTROLLER': 'fleet-controller',
    'DEFECT_DETECTOR': 'defect-detector',
}


def get_telemetry_channel(device_id: int) -> str:
    """获取特定设备的遥测通道"""
    return f"{CHANNELS['TELEMETRY_RAW']}:{device_id}"


def get_control_channel(device_id: int) -> str:
    """获取特定设备的控制通道"""
    return f"{CHANNELS['CONTROL_COMMAND']}:{device_id}"


def get_alarm_channel(device_id: int) -> str:
    """获取特定设备的告警通道"""
    return f"{CHANNELS['ALARM_EVENT']}:{device_id}"


def get_prediction_channel(device_id: int) -> str:
    """获取特定设备的预测通道"""
    return f"{CHANNELS['PREDICTION_RESULT']}:{device_id}"
