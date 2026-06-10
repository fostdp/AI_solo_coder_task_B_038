"""
数据库写入微服务
订阅Redis频道，批量写入TimescaleDB
支持批量缓存、优雅降级、自动重连
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import text, exc
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    MicroserviceBase, RedisConfig,
    CHANNELS, SERVICE_IDS,
    TelemetryData, ControlCommand, PredictionResult, AlarmEvent,
    EndpointDetection, PressureRiseTest,
    DefrostOptimization, DefrostCommand, DefrostStatus,
    FleetSchedule, FleetCommand, FleetStatus,
    DefectDetection, ImageUpload, BatchRecord,
    validate_message, extract_payload
)


class DataType(str, Enum):
    """数据类型枚举"""
    TELEMETRY = "telemetry"
    CONTROL = "control"
    PREDICTION = "prediction"
    ALARM = "alarm"
    ENDPOINT = "endpoint"
    PRESSURE_RISE_TEST = "pressure_rise_test"
    DEFROST_OPTIMIZATION = "defrost_optimization"
    DEFROST_COMMAND = "defrost_command"
    DEFROST_STATUS = "defrost_status"
    FLEET_SCHEDULE = "fleet_schedule"
    FLEET_COMMAND = "fleet_command"
    FLEET_STATUS = "fleet_status"
    DEFECT_DETECTION = "defect_detection"
    IMAGE_UPLOAD = "image_upload"
    BATCH_RECORD = "batch_record"


@dataclass
class WriteItem:
    """写入队列项"""
    data_type: DataType
    data: Dict[str, Any]
    received_at: float


@dataclass
class DBConfig:
    """数据库配置"""
    url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/freeze_dryer"
    pool_size: int = 20
    max_overflow: int = 30
    pool_recycle: int = 3600
    connect_timeout: int = 10
    max_reconnect_attempts: int = 10


class DBWriterService(MicroserviceBase):
    """数据库写入微服务"""

    def __init__(self, redis_config: RedisConfig = None, db_config: DBConfig = None):
        super().__init__(
            service_id=SERVICE_IDS['DB_WRITER'],
            service_type="database_writer",
            redis_config=redis_config
        )

        self._db_config: DBConfig = db_config or DBConfig(
            url=os.environ.get(
                'DATABASE_URL',
                'postgresql+asyncpg://postgres:postgres@localhost:5432/freeze_dryer'
            )
        )

        self._engine = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._db_connected: bool = False
        self._db_reconnect_attempts: int = 0

        self._write_queue: asyncio.Queue[WriteItem] = asyncio.Queue(maxsize=10000)
        self._batch_size: int = 50
        self._flush_interval: float = 10.0

        self._fallback_dir: Path = Path(__file__).parent / "fallback_data"
        self._fallback_dir.mkdir(exist_ok=True)

        self._writer_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        self._metrics["queue_size"] = 0
        self._metrics["total_written"] = 0
        self._metrics["total_fallback"] = 0
        self._metrics["db_errors"] = 0

    async def _connect_db(self) -> bool:
        """连接数据库"""
        if not HAS_SQLALCHEMY:
            print(f"[{self.service_id}] 警告: SQLAlchemy或asyncpg未安装")
            return False

        try:
            if self._engine:
                await self._engine.dispose()

            self._engine = create_async_engine(
                self._db_config.url,
                echo=False,
                pool_size=self._db_config.pool_size,
                max_overflow=self._db_config.max_overflow,
                pool_recycle=self._db_config.pool_recycle,
                connect_args={"timeout": self._db_config.connect_timeout}
            )

            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            self._db_connected = True
            self._db_reconnect_attempts = 0
            print(f"[{self.service_id}] 数据库连接成功")

            await self._load_fallback_data()
            return True

        except Exception as e:
            print(f"[{self.service_id}] 数据库连接失败: {e}")
            self._db_connected = False
            return False

    async def _disconnect_db(self) -> None:
        """断开数据库连接"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._db_connected = False
            print(f"[{self.service_id}] 数据库已断开")

    async def _reconnect_db(self) -> bool:
        """重连数据库"""
        if self._db_reconnect_attempts >= self._db_config.max_reconnect_attempts:
            print(f"[{self.service_id}] 达到最大重连次数，放弃重连")
            return False

        self._db_reconnect_attempts += 1
        wait_time = min(2 ** self._db_reconnect_attempts, 30)
        print(f"[{self.service_id}] 第{self._db_reconnect_attempts}次数据库重连，等待{wait_time}s...")

        await asyncio.sleep(wait_time)
        return await self._connect_db()

    async def _subscribe_channels(self) -> None:
        """订阅Redis频道"""
        await self.subscribe(CHANNELS['TELEMETRY_RAW'], self._handle_telemetry)
        await self.subscribe(CHANNELS['CONTROL_COMMAND'], self._handle_control)
        await self.subscribe(CHANNELS['PREDICTION_RESULT'], self._handle_prediction)
        await self.subscribe(CHANNELS['ALARM_EVENT'], self._handle_alarm)
        await self.subscribe(CHANNELS['ENDPOINT_DETECTION'], self._handle_endpoint)
        await self.subscribe(CHANNELS['PRESSURE_RISE_TEST'], self._handle_pressure_rise_test)
        await self.subscribe(CHANNELS['DEFROST_OPTIMIZATION'], self._handle_defrost_optimization)
        await self.subscribe(CHANNELS['DEFROST_COMMAND'], self._handle_defrost_command)
        await self.subscribe(CHANNELS['DEFROST_STATUS'], self._handle_defrost_status)
        await self.subscribe(CHANNELS['FLEET_SCHEDULE'], self._handle_fleet_schedule)
        await self.subscribe(CHANNELS['FLEET_COMMAND'], self._handle_fleet_command)
        await self.subscribe(CHANNELS['FLEET_STATUS'], self._handle_fleet_status)
        await self.subscribe(CHANNELS['DEFECT_DETECTION'], self._handle_defect_detection)
        await self.subscribe(CHANNELS['IMAGE_UPLOAD'], self._handle_image_upload)
        await self.subscribe(CHANNELS['BATCH_RECORD'], self._handle_batch_record)

    async def _on_start(self) -> None:
        """服务启动时执行"""
        print(f"[{self.service_id}] 启动数据库写入服务...")

        await self._connect_db()

        self._writer_task = asyncio.create_task(self._writer_loop())
        self._sub_tasks.append(self._writer_task)

        self._reconnect_task = asyncio.create_task(self._reconnect_loop())
        self._sub_tasks.append(self._reconnect_task)

        print(f"[{self.service_id}] 数据库写入服务已启动")
        print(f"[{self.service_id}] 批量大小: {self._batch_size}")
        print(f"[{self.service_id}] 刷新间隔: {self._flush_interval}s")
        print(f"[{self.service_id}] 队列容量: {self._write_queue.maxsize}")

    async def _on_stop(self) -> None:
        """服务停止时执行"""
        print(f"[{self.service_id}] 停止数据库写入服务...")

        await self._flush_queue()
        await self._disconnect_db()

        print(f"[{self.service_id}] 数据库写入服务已停止")

    async def _handle_telemetry(self, message: Dict) -> None:
        """处理遥测数据"""
        if not validate_message(message, 'telemetry'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.TELEMETRY,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_control(self, message: Dict) -> None:
        """处理控制命令"""
        if not validate_message(message, 'control_command'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.CONTROL,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_prediction(self, message: Dict) -> None:
        """处理预测结果"""
        if not validate_message(message, 'prediction'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.PREDICTION,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_alarm(self, message: Dict) -> None:
        """处理告警事件"""
        if not validate_message(message, 'alarm'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.ALARM,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_endpoint(self, message: Dict) -> None:
        """处理干燥终点判定"""
        if not validate_message(message, 'endpoint'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.ENDPOINT,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_pressure_rise_test(self, message: Dict) -> None:
        """处理压力升测试"""
        if not validate_message(message, 'pressure_rise_test'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.PRESSURE_RISE_TEST,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_defrost_optimization(self, message: Dict) -> None:
        """处理除霜优化"""
        if not validate_message(message, 'defrost_optimization'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.DEFROST_OPTIMIZATION,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_defrost_command(self, message: Dict) -> None:
        """处理除霜命令"""
        if not validate_message(message, 'defrost_command'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.DEFROST_COMMAND,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_defrost_status(self, message: Dict) -> None:
        """处理除霜状态"""
        if not validate_message(message, 'defrost_status'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.DEFROST_STATUS,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_fleet_schedule(self, message: Dict) -> None:
        """处理群控调度计划"""
        if not validate_message(message, 'fleet_schedule'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.FLEET_SCHEDULE,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_fleet_command(self, message: Dict) -> None:
        """处理群控命令"""
        if not validate_message(message, 'fleet_command'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.FLEET_COMMAND,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_fleet_status(self, message: Dict) -> None:
        """处理群控状态"""
        if not validate_message(message, 'fleet_status'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.FLEET_STATUS,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_defect_detection(self, message: Dict) -> None:
        """处理缺陷检测结果"""
        if not validate_message(message, 'defect_detection'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.DEFECT_DETECTION,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_image_upload(self, message: Dict) -> None:
        """处理图像上传"""
        if not validate_message(message, 'image_upload'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.IMAGE_UPLOAD,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _handle_batch_record(self, message: Dict) -> None:
        """处理批次记录"""
        if not validate_message(message, 'batch_record'):
            return

        payload = extract_payload(message)
        self._increment_metric("messages_received")

        item = WriteItem(
            data_type=DataType.BATCH_RECORD,
            data=payload,
            received_at=asyncio.get_event_loop().time()
        )

        await self._enqueue_item(item)

    async def _enqueue_item(self, item: WriteItem) -> None:
        """将数据项加入队列"""
        try:
            if self._write_queue.full():
                print(f"[{self.service_id}] 队列已满，写入降级文件")
                await self._write_fallback(item)
            else:
                self._write_queue.put_nowait(item)

            self._metrics["queue_size"] = self._write_queue.qsize()

        except Exception as e:
            print(f"[{self.service_id}] 入队失败，写入降级文件: {e}")
            await self._write_fallback(item)

    async def _writer_loop(self) -> None:
        """写入循环"""
        while self._running:
            batch: List[WriteItem] = []
            try:
                item = await asyncio.wait_for(
                    self._write_queue.get(),
                    timeout=self._flush_interval
                )
                batch.append(item)

                while len(batch) < self._batch_size:
                    try:
                        item = self._write_queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break

                if self._db_connected:
                    success = await self._write_batch(batch)
                    if not success:
                        await self._write_fallback_batch(batch)
                else:
                    await self._write_fallback_batch(batch)

                self._metrics["queue_size"] = self._write_queue.qsize()

            except asyncio.TimeoutError:
                if batch:
                    if self._db_connected:
                        success = await self._write_batch(batch)
                        if not success:
                            await self._write_fallback_batch(batch)
                    else:
                        await self._write_fallback_batch(batch)

            except Exception as e:
                print(f"[{self.service_id}] 写入循环异常: {e}")
                self._increment_metric("errors")
                if batch:
                    await self._write_fallback_batch(batch)
                await asyncio.sleep(1)

    async def _write_batch(self, batch: List[WriteItem]) -> bool:
        """批量写入数据库"""
        if not self._session_factory or not self._db_connected:
            return False

        try:
            async with self._session_factory() as session:
                telemetry_data: List[Tuple] = []
                control_data: List[Tuple] = []
                prediction_data: List[Tuple] = []
                alarm_data: List[Tuple] = []
                endpoint_data: List[Tuple] = []
                prt_data: List[Tuple] = []
                defrost_opt_data: List[Tuple] = []
                defrost_cmd_data: List[Tuple] = []
                defrost_status_data: List[Tuple] = []
                fleet_schedule_data: List[Tuple] = []
                fleet_cmd_data: List[Tuple] = []
                fleet_status_data: List[Tuple] = []
                defect_data: List[Tuple] = []
                image_upload_data: List[Tuple] = []
                batch_record_data: List[Tuple] = []

                for item in batch:
                    try:
                        if item.data_type == DataType.TELEMETRY:
                            telemetry_data.append(self._prepare_telemetry(item.data))
                        elif item.data_type == DataType.CONTROL:
                            control_data.append(self._prepare_control(item.data))
                        elif item.data_type == DataType.PREDICTION:
                            prediction_data.append(self._prepare_prediction(item.data))
                        elif item.data_type == DataType.ALARM:
                            alarm_data.append(self._prepare_alarm(item.data))
                        elif item.data_type == DataType.ENDPOINT:
                            endpoint_data.append(self._prepare_endpoint(item.data))
                        elif item.data_type == DataType.PRESSURE_RISE_TEST:
                            prt_data.append(self._prepare_pressure_rise_test(item.data))
                        elif item.data_type == DataType.DEFROST_OPTIMIZATION:
                            defrost_opt_data.append(self._prepare_defrost_optimization(item.data))
                        elif item.data_type == DataType.DEFROST_COMMAND:
                            defrost_cmd_data.append(self._prepare_defrost_command(item.data))
                        elif item.data_type == DataType.DEFROST_STATUS:
                            defrost_status_data.append(self._prepare_defrost_status(item.data))
                        elif item.data_type == DataType.FLEET_SCHEDULE:
                            fleet_schedule_data.append(self._prepare_fleet_schedule(item.data))
                        elif item.data_type == DataType.FLEET_COMMAND:
                            fleet_cmd_data.append(self._prepare_fleet_command(item.data))
                        elif item.data_type == DataType.FLEET_STATUS:
                            fleet_status_data.append(self._prepare_fleet_status(item.data))
                        elif item.data_type == DataType.DEFECT_DETECTION:
                            defect_data.append(self._prepare_defect_detection(item.data))
                        elif item.data_type == DataType.IMAGE_UPLOAD:
                            image_upload_data.append(self._prepare_image_upload(item.data))
                        elif item.data_type == DataType.BATCH_RECORD:
                            batch_record_data.append(self._prepare_batch_record(item.data))
                    except Exception as e:
                        print(f"[{self.service_id}] 数据准备失败: {e}")
                        self._increment_metric("errors")

                if telemetry_data:
                    await self._insert_telemetry(session, telemetry_data)
                if control_data:
                    await self._insert_control(session, control_data)
                if prediction_data:
                    await self._insert_prediction(session, prediction_data)
                if alarm_data:
                    await self._insert_alarm(session, alarm_data)
                if endpoint_data:
                    await self._insert_endpoint(session, endpoint_data)
                if prt_data:
                    await self._insert_pressure_rise_test(session, prt_data)
                if defrost_opt_data:
                    await self._insert_defrost_optimization(session, defrost_opt_data)
                if defrost_cmd_data:
                    await self._insert_defrost_command(session, defrost_cmd_data)
                if defrost_status_data:
                    await self._insert_defrost_status(session, defrost_status_data)
                if fleet_schedule_data:
                    await self._insert_fleet_schedule(session, fleet_schedule_data)
                if fleet_cmd_data:
                    await self._insert_fleet_command(session, fleet_cmd_data)
                if fleet_status_data:
                    await self._insert_fleet_status(session, fleet_status_data)
                if defect_data:
                    await self._insert_defect_detection(session, defect_data)
                if image_upload_data:
                    await self._insert_image_upload(session, image_upload_data)
                if batch_record_data:
                    await self._insert_batch_record(session, batch_record_data)

                await session.commit()

            written_count = (len(telemetry_data) + len(control_data) + len(prediction_data) + 
                           len(alarm_data) + len(endpoint_data) + len(prt_data) +
                           len(defrost_opt_data) + len(defrost_cmd_data) + len(defrost_status_data) +
                           len(fleet_schedule_data) + len(fleet_cmd_data) + len(fleet_status_data) +
                           len(defect_data) + len(image_upload_data) + len(batch_record_data))
            self._metrics["total_written"] += written_count
            print(f"[{self.service_id}] 批量写入完成: {len(batch)}条")
            return True

        except exc.OperationalError as e:
            print(f"[{self.service_id}] 数据库连接异常: {e}")
            self._db_connected = False
            self._increment_metric("db_errors")
            return False

        except Exception as e:
            print(f"[{self.service_id}] 批量写入失败: {e}")
            self._increment_metric("db_errors")
            return False

    def _prepare_telemetry(self, data: Dict) -> Tuple:
        """准备遥测数据"""
        temps = data.get('temperatures', [0.0] * 8)
        vacuums = data.get('vacuum_levels', [0.0] * 2)
        powers = data.get('heating_powers', [0.0] * 8)

        return (
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('device_id', 0),
            data.get('shelf_id', 0),
            temps[0] if len(temps) > 0 else 0.0,
            temps[1] if len(temps) > 1 else 0.0,
            temps[2] if len(temps) > 2 else 0.0,
            temps[3] if len(temps) > 3 else 0.0,
            temps[4] if len(temps) > 4 else 0.0,
            temps[5] if len(temps) > 5 else 0.0,
            temps[6] if len(temps) > 6 else 0.0,
            temps[7] if len(temps) > 7 else 0.0,
            vacuums[0] if len(vacuums) > 0 else 0.0,
            vacuums[1] if len(vacuums) > 1 else 0.0,
            data.get('cold_trap_temp', 0.0),
            powers[0] if len(powers) > 0 else 0.0,
            powers[1] if len(powers) > 1 else 0.0,
            powers[2] if len(powers) > 2 else 0.0,
            powers[3] if len(powers) > 3 else 0.0,
            powers[4] if len(powers) > 4 else 0.0,
            powers[5] if len(powers) > 5 else 0.0,
            powers[6] if len(powers) > 6 else 0.0,
            powers[7] if len(powers) > 7 else 0.0,
        )

    def _prepare_control(self, data: Dict) -> Tuple:
        """准备控制命令数据"""
        adjustments = data.get('power_adjustments', [0.0] * 8)

        return (
            data.get('device_id', 0),
            data.get('shelf_id', 0),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            adjustments[0] if len(adjustments) > 0 else 0.0,
            adjustments[1] if len(adjustments) > 1 else 0.0,
            adjustments[2] if len(adjustments) > 2 else 0.0,
            adjustments[3] if len(adjustments) > 3 else 0.0,
            adjustments[4] if len(adjustments) > 4 else 0.0,
            adjustments[5] if len(adjustments) > 5 else 0.0,
            adjustments[6] if len(adjustments) > 6 else 0.0,
            adjustments[7] if len(adjustments) > 7 else 0.0,
            data.get('auto_mode', True),
        )

    def _prepare_prediction(self, data: Dict) -> Tuple:
        """准备预测结果数据"""
        return (
            data.get('device_id', 0),
            data.get('batch_id'),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('moisture_content', 0.0),
            data.get('moisture_confidence', 0.0),
            data.get('moisture_threshold', 3.0),
            data.get('reconstitution_time', 0.0),
            data.get('reconstitution_confidence', 0.0),
            data.get('reconstitution_threshold', 120.0),
            data.get('drying_rate', 0.0),
            data.get('is_qualified', True),
        )

    def _prepare_alarm(self, data: Dict) -> Tuple:
        """准备告警数据"""
        return (
            data.get('alarm_id'),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('device_id', 0),
            data.get('shelf_id'),
            data.get('alarm_type', ''),
            data.get('severity', ''),
            data.get('message', ''),
            data.get('acknowledged', False),
            data.get('acknowledged_by'),
            data.get('acknowledged_at'),
        )

    def _prepare_endpoint(self, data: Dict) -> Tuple:
        """准备干燥终点判定数据"""
        return (
            data.get('device_id', 0),
            data.get('batch_id', ''),
            data.get('cycle_phase', ''),
            data.get('detection_method', ''),
            data.get('endpoint_timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('detection_confidence', 0.0),
            data.get('pressure_rise_delta'),
            data.get('temp_inflection_point'),
            data.get('temp_first_derivative'),
            data.get('autoencoder_recon_error'),
            data.get('cycle_duration_hours'),
            data.get('estimated_energy_saving'),
        )

    def _prepare_pressure_rise_test(self, data: Dict) -> Tuple:
        """准备压力升测试数据"""
        return (
            data.get('device_id', 0),
            data.get('batch_id'),
            data.get('test_start_time', datetime.now(timezone.utc).isoformat()),
            data.get('test_end_time'),
            data.get('initial_pressure_pa', 0.0),
            data.get('final_pressure_pa', 0.0),
            data.get('pressure_rise_pa_per_min', 0.0),
            data.get('test_duration_seconds', 0),
            data.get('is_endpoint_detected', False),
            data.get('detection_confidence'),
            data.get('test_status', 'completed'),
        )

    def _prepare_defrost_optimization(self, data: Dict) -> Tuple:
        """准备除霜优化数据"""
        return (
            data.get('device_id', 0),
            data.get('batch_id'),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('estimated_frost_thickness_mm', 0.0),
            data.get('cold_trap_temp_avg', 0.0),
            data.get('cold_trap_temp_trend', 0.0),
            data.get('recommended_defrost_interval_hours', 0.0),
            data.get('recommended_heating_power_pct', 0.0),
            data.get('estimated_energy_saving'),
            data.get('defrost_status', 'pending'),
        )

    def _prepare_defrost_command(self, data: Dict) -> Tuple:
        """准备除霜命令数据"""
        return (
            data.get('device_id', 0),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('command', ''),
            data.get('heating_power_pct', 0.0),
            data.get('max_duration_minutes'),
            data.get('batch_id'),
        )

    def _prepare_defrost_status(self, data: Dict) -> Tuple:
        """准备除霜状态数据"""
        return (
            data.get('device_id', 0),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('status', ''),
            data.get('progress_pct'),
            data.get('current_temp'),
            data.get('target_temp'),
            data.get('energy_consumed_kwh'),
            data.get('batch_id'),
        )

    def _prepare_fleet_schedule(self, data: Dict) -> Tuple:
        """准备群控调度计划数据"""
        details = data.get('details', [])
        details_json = json.dumps(details, ensure_ascii=False) if details else None
        return (
            data.get('schedule_id', str(uuid4())),
            data.get('schedule_date', datetime.now(timezone.utc).strftime('%Y-%m-%d')),
            data.get('total_required_batches', 0),
            data.get('estimated_energy_cost', 0.0),
            data.get('optimized_energy_saving', 0.0),
            data.get('solver_status', ''),
            details_json,
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
        )

    def _prepare_fleet_command(self, data: Dict) -> Tuple:
        """准备群控命令数据"""
        return (
            data.get('device_id', 0),
            data.get('command', ''),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('batch_id'),
            data.get('formula_id'),
            data.get('freeze_profile_id'),
            data.get('priority', 0),
        )

    def _prepare_fleet_status(self, data: Dict) -> Tuple:
        """准备群控状态数据"""
        return (
            data.get('device_id', 0),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('batch_id'),
            data.get('batch_status', ''),
            data.get('current_phase'),
            data.get('phase_progress_pct'),
            data.get('estimated_completion_time'),
            data.get('current_power_kw'),
        )

    def _prepare_defect_detection(self, data: Dict) -> Tuple:
        """准备缺陷检测数据"""
        return (
            data.get('device_id', 0),
            data.get('batch_id', ''),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('image_path', ''),
            data.get('image_hash', ''),
            data.get('defect_type', ''),
            data.get('defect_severity', ''),
            data.get('confidence', 0.0),
            data.get('bbox_x'),
            data.get('bbox_y'),
            data.get('bbox_width'),
            data.get('bbox_height'),
            data.get('shelf_id'),
            data.get('vial_position'),
            data.get('is_manual_reviewed', False),
        )

    def _prepare_image_upload(self, data: Dict) -> Tuple:
        """准备图像上传数据"""
        return (
            data.get('device_id', 0),
            data.get('batch_id', ''),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('image_path', ''),
            data.get('image_hash', ''),
            data.get('shelf_id'),
            data.get('vial_position'),
            data.get('file_size_bytes'),
            data.get('content_type'),
        )

    def _prepare_batch_record(self, data: Dict) -> Tuple:
        """准备批次记录数据"""
        return (
            data.get('device_id', 0),
            data.get('batch_id', ''),
            data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            data.get('update_type', ''),
            data.get('freeze_profile_id'),
            data.get('formula_id'),
            data.get('start_time'),
            data.get('end_time'),
            data.get('primary_drying_endpoint'),
            data.get('secondary_drying_endpoint'),
            data.get('avg_moisture_content'),
            data.get('avg_reconstitution_time'),
            data.get('defect_rate'),
            data.get('quality_score'),
            data.get('batch_status'),
            data.get('notes'),
        )

    async def _insert_telemetry(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入遥测数据"""
        stmt = text("""
            INSERT INTO telemetry (
                timestamp, device_id, shelf_id,
                temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8,
                vacuum_1, vacuum_2, cold_trap_temp,
                power_1, power_2, power_3, power_4, power_5, power_6, power_7, power_8
            ) VALUES (
                :timestamp, :device_id, :shelf_id,
                :temp_1, :temp_2, :temp_3, :temp_4, :temp_5, :temp_6, :temp_7, :temp_8,
                :vacuum_1, :vacuum_2, :cold_trap_temp,
                :power_1, :power_2, :power_3, :power_4, :power_5, :power_6, :power_7, :power_8
            )
            ON CONFLICT (timestamp, device_id, shelf_id) DO UPDATE SET
                temp_1 = EXCLUDED.temp_1,
                temp_2 = EXCLUDED.temp_2,
                temp_3 = EXCLUDED.temp_3,
                temp_4 = EXCLUDED.temp_4,
                temp_5 = EXCLUDED.temp_5,
                temp_6 = EXCLUDED.temp_6,
                temp_7 = EXCLUDED.temp_7,
                temp_8 = EXCLUDED.temp_8,
                vacuum_1 = EXCLUDED.vacuum_1,
                vacuum_2 = EXCLUDED.vacuum_2,
                cold_trap_temp = EXCLUDED.cold_trap_temp,
                power_1 = EXCLUDED.power_1,
                power_2 = EXCLUDED.power_2,
                power_3 = EXCLUDED.power_3,
                power_4 = EXCLUDED.power_4,
                power_5 = EXCLUDED.power_5,
                power_6 = EXCLUDED.power_6,
                power_7 = EXCLUDED.power_7,
                power_8 = EXCLUDED.power_8
        """)

        params = [
            {
                'timestamp': row[0],
                'device_id': row[1],
                'shelf_id': row[2],
                'temp_1': row[3], 'temp_2': row[4], 'temp_3': row[5], 'temp_4': row[6],
                'temp_5': row[7], 'temp_6': row[8], 'temp_7': row[9], 'temp_8': row[10],
                'vacuum_1': row[11], 'vacuum_2': row[12], 'cold_trap_temp': row[13],
                'power_1': row[14], 'power_2': row[15], 'power_3': row[16], 'power_4': row[17],
                'power_5': row[18], 'power_6': row[19], 'power_7': row[20], 'power_8': row[21],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_control(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入控制命令"""
        stmt = text("""
            INSERT INTO control_commands (
                device_id, shelf_id, timestamp,
                power_adj_1, power_adj_2, power_adj_3, power_adj_4,
                power_adj_5, power_adj_6, power_adj_7, power_adj_8,
                auto_mode
            ) VALUES (
                :device_id, :shelf_id, :timestamp,
                :power_adj_1, :power_adj_2, :power_adj_3, :power_adj_4,
                :power_adj_5, :power_adj_6, :power_adj_7, :power_adj_8,
                :auto_mode
            )
        """)

        params = [
            {
                'device_id': row[0],
                'shelf_id': row[1],
                'timestamp': row[2],
                'power_adj_1': row[3], 'power_adj_2': row[4], 'power_adj_3': row[5], 'power_adj_4': row[6],
                'power_adj_5': row[7], 'power_adj_6': row[8], 'power_adj_7': row[9], 'power_adj_8': row[10],
                'auto_mode': row[11],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_prediction(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入预测结果"""
        stmt = text("""
            INSERT INTO prediction_results (
                device_id, batch_id, timestamp,
                moisture_pred, moisture_conf, moisture_threshold,
                reconstitution_pred, reconstitution_conf, reconstitution_threshold,
                drying_rate, is_qualified
            ) VALUES (
                :device_id, :batch_id, :timestamp,
                :moisture_pred, :moisture_conf, :moisture_threshold,
                :reconstitution_pred, :reconstitution_conf, :reconstitution_threshold,
                :drying_rate, :is_qualified
            )
        """)

        params = [
            {
                'device_id': row[0],
                'batch_id': row[1],
                'timestamp': row[2],
                'moisture_pred': row[3],
                'moisture_conf': row[4],
                'moisture_threshold': row[5],
                'reconstitution_pred': row[6],
                'reconstitution_conf': row[7],
                'reconstitution_threshold': row[8],
                'drying_rate': row[9],
                'is_qualified': row[10],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_alarm(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入告警"""
        stmt = text("""
            INSERT INTO alarms (
                id, timestamp, device_id, shelf_id,
                alarm_type, severity, message,
                acknowledged, acknowledged_by, acknowledged_at
            ) VALUES (
                :id, :timestamp, :device_id, :shelf_id,
                :alarm_type, :severity, :message,
                :acknowledged, :acknowledged_by, :acknowledged_at
            )
            ON CONFLICT (id) DO UPDATE SET
                acknowledged = EXCLUDED.acknowledged,
                acknowledged_by = EXCLUDED.acknowledged_by,
                acknowledged_at = EXCLUDED.acknowledged_at
        """)

        params = [
            {
                'id': row[0],
                'timestamp': row[1],
                'device_id': row[2],
                'shelf_id': row[3],
                'alarm_type': row[4],
                'severity': row[5],
                'message': row[6],
                'acknowledged': row[7],
                'acknowledged_by': row[8],
                'acknowledged_at': row[9],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_endpoint(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入干燥终点判定"""
        stmt = text("""
            INSERT INTO drying_endpoints (
                device_id, batch_id, cycle_phase, detection_method,
                endpoint_timestamp, detection_confidence, pressure_rise_delta,
                temp_inflection_point, temp_first_derivative, autoencoder_recon_error,
                cycle_duration_hours, estimated_energy_saving
            ) VALUES (
                :device_id, :batch_id, :cycle_phase, :detection_method,
                :endpoint_timestamp, :detection_confidence, :pressure_rise_delta,
                :temp_inflection_point, :temp_first_derivative, :autoencoder_recon_error,
                :cycle_duration_hours, :estimated_energy_saving
            )
        """)

        params = [
            {
                'device_id': row[0],
                'batch_id': row[1],
                'cycle_phase': row[2],
                'detection_method': row[3],
                'endpoint_timestamp': row[4],
                'detection_confidence': row[5],
                'pressure_rise_delta': row[6],
                'temp_inflection_point': row[7],
                'temp_first_derivative': row[8],
                'autoencoder_recon_error': row[9],
                'cycle_duration_hours': row[10],
                'estimated_energy_saving': row[11],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_pressure_rise_test(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入压力升测试"""
        stmt = text("""
            INSERT INTO pressure_rise_tests (
                device_id, batch_id, test_start_time, test_end_time,
                initial_pressure_pa, final_pressure_pa, pressure_rise_pa_per_min,
                test_duration_seconds, is_endpoint_detected, detection_confidence,
                test_status
            ) VALUES (
                :device_id, :batch_id, :test_start_time, :test_end_time,
                :initial_pressure_pa, :final_pressure_pa, :pressure_rise_pa_per_min,
                :test_duration_seconds, :is_endpoint_detected, :detection_confidence,
                :test_status
            )
        """)

        params = [
            {
                'device_id': row[0],
                'batch_id': row[1],
                'test_start_time': row[2],
                'test_end_time': row[3],
                'initial_pressure_pa': row[4],
                'final_pressure_pa': row[5],
                'pressure_rise_pa_per_min': row[6],
                'test_duration_seconds': row[7],
                'is_endpoint_detected': row[8],
                'detection_confidence': row[9],
                'test_status': row[10],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_defrost_optimization(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入除霜优化"""
        stmt = text("""
            INSERT INTO defrost_optimizations (
                device_id, batch_id, timestamp, estimated_frost_thickness_mm,
                cold_trap_temp_avg, cold_trap_temp_trend, recommended_defrost_interval_hours,
                recommended_heating_power_pct, estimated_energy_saving, defrost_status
            ) VALUES (
                :device_id, :batch_id, :timestamp, :estimated_frost_thickness_mm,
                :cold_trap_temp_avg, :cold_trap_temp_trend, :recommended_defrost_interval_hours,
                :recommended_heating_power_pct, :estimated_energy_saving, :defrost_status
            )
        """)

        params = [
            {
                'device_id': row[0],
                'batch_id': row[1],
                'timestamp': row[2],
                'estimated_frost_thickness_mm': row[3],
                'cold_trap_temp_avg': row[4],
                'cold_trap_temp_trend': row[5],
                'recommended_defrost_interval_hours': row[6],
                'recommended_heating_power_pct': row[7],
                'estimated_energy_saving': row[8],
                'defrost_status': row[9],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_defrost_command(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入除霜命令"""
        stmt = text("""
            INSERT INTO defrost_commands (
                device_id, timestamp, command, heating_power_pct,
                max_duration_minutes, batch_id
            ) VALUES (
                :device_id, :timestamp, :command, :heating_power_pct,
                :max_duration_minutes, :batch_id
            )
        """)

        params = [
            {
                'device_id': row[0],
                'timestamp': row[1],
                'command': row[2],
                'heating_power_pct': row[3],
                'max_duration_minutes': row[4],
                'batch_id': row[5],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_defrost_status(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入除霜状态"""
        stmt = text("""
            INSERT INTO defrost_status (
                device_id, timestamp, status, progress_pct,
                current_temp, target_temp, energy_consumed_kwh, batch_id
            ) VALUES (
                :device_id, :timestamp, :status, :progress_pct,
                :current_temp, :target_temp, :energy_consumed_kwh, :batch_id
            )
        """)

        params = [
            {
                'device_id': row[0],
                'timestamp': row[1],
                'status': row[2],
                'progress_pct': row[3],
                'current_temp': row[4],
                'target_temp': row[5],
                'energy_consumed_kwh': row[6],
                'batch_id': row[7],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_fleet_schedule(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入群控调度计划"""
        stmt = text("""
            INSERT INTO fleet_schedules (
                schedule_id, schedule_date, total_required_batches,
                estimated_energy_cost, optimized_energy_saving, solver_status,
                details, timestamp
            ) VALUES (
                :schedule_id, :schedule_date, :total_required_batches,
                :estimated_energy_cost, :optimized_energy_saving, :solver_status,
                :details, :timestamp
            )
            ON CONFLICT (schedule_id) DO UPDATE SET
                total_required_batches = EXCLUDED.total_required_batches,
                estimated_energy_cost = EXCLUDED.estimated_energy_cost,
                optimized_energy_saving = EXCLUDED.optimized_energy_saving,
                solver_status = EXCLUDED.solver_status,
                details = EXCLUDED.details
        """)

        params = [
            {
                'schedule_id': row[0],
                'schedule_date': row[1],
                'total_required_batches': row[2],
                'estimated_energy_cost': row[3],
                'optimized_energy_saving': row[4],
                'solver_status': row[5],
                'details': row[6],
                'timestamp': row[7],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_fleet_command(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入群控命令"""
        stmt = text("""
            INSERT INTO fleet_commands (
                device_id, command, timestamp, batch_id,
                formula_id, freeze_profile_id, priority
            ) VALUES (
                :device_id, :command, :timestamp, :batch_id,
                :formula_id, :freeze_profile_id, :priority
            )
        """)

        params = [
            {
                'device_id': row[0],
                'command': row[1],
                'timestamp': row[2],
                'batch_id': row[3],
                'formula_id': row[4],
                'freeze_profile_id': row[5],
                'priority': row[6],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_fleet_status(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入群控状态"""
        stmt = text("""
            INSERT INTO fleet_status (
                device_id, timestamp, batch_id, batch_status,
                current_phase, phase_progress_pct, estimated_completion_time,
                current_power_kw
            ) VALUES (
                :device_id, :timestamp, :batch_id, :batch_status,
                :current_phase, :phase_progress_pct, :estimated_completion_time,
                :current_power_kw
            )
        """)

        params = [
            {
                'device_id': row[0],
                'timestamp': row[1],
                'batch_id': row[2],
                'batch_status': row[3],
                'current_phase': row[4],
                'phase_progress_pct': row[5],
                'estimated_completion_time': row[6],
                'current_power_kw': row[7],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_defect_detection(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入缺陷检测"""
        stmt = text("""
            INSERT INTO product_defects (
                device_id, batch_id, timestamp, image_path, image_hash,
                defect_type, defect_severity, confidence,
                bbox_x, bbox_y, bbox_width, bbox_height,
                shelf_id, vial_position, is_manual_reviewed
            ) VALUES (
                :device_id, :batch_id, :timestamp, :image_path, :image_hash,
                :defect_type, :defect_severity, :confidence,
                :bbox_x, :bbox_y, :bbox_width, :bbox_height,
                :shelf_id, :vial_position, :is_manual_reviewed
            )
        """)

        params = [
            {
                'device_id': row[0],
                'batch_id': row[1],
                'timestamp': row[2],
                'image_path': row[3],
                'image_hash': row[4],
                'defect_type': row[5],
                'defect_severity': row[6],
                'confidence': row[7],
                'bbox_x': row[8],
                'bbox_y': row[9],
                'bbox_width': row[10],
                'bbox_height': row[11],
                'shelf_id': row[12],
                'vial_position': row[13],
                'is_manual_reviewed': row[14],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_image_upload(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入图像上传"""
        stmt = text("""
            INSERT INTO image_uploads (
                device_id, batch_id, timestamp, image_path, image_hash,
                shelf_id, vial_position, file_size_bytes, content_type
            ) VALUES (
                :device_id, :batch_id, :timestamp, :image_path, :image_hash,
                :shelf_id, :vial_position, :file_size_bytes, :content_type
            )
        """)

        params = [
            {
                'device_id': row[0],
                'batch_id': row[1],
                'timestamp': row[2],
                'image_path': row[3],
                'image_hash': row[4],
                'shelf_id': row[5],
                'vial_position': row[6],
                'file_size_bytes': row[7],
                'content_type': row[8],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _insert_batch_record(self, session: AsyncSession, data: List[Tuple]) -> None:
        """批量插入批次记录"""
        stmt = text("""
            INSERT INTO batch_records (
                device_id, batch_id, timestamp, update_type,
                freeze_profile_id, formula_id, start_time, end_time,
                primary_drying_endpoint, secondary_drying_endpoint,
                avg_moisture_content, avg_reconstitution_time,
                defect_rate, quality_score, batch_status, notes
            ) VALUES (
                :device_id, :batch_id, :timestamp, :update_type,
                :freeze_profile_id, :formula_id, :start_time, :end_time,
                :primary_drying_endpoint, :secondary_drying_endpoint,
                :avg_moisture_content, :avg_reconstitution_time,
                :defect_rate, :quality_score, :batch_status, :notes
            )
            ON CONFLICT (device_id, batch_id, update_type) DO UPDATE SET
                freeze_profile_id = COALESCE(EXCLUDED.freeze_profile_id, batch_records.freeze_profile_id),
                formula_id = COALESCE(EXCLUDED.formula_id, batch_records.formula_id),
                start_time = COALESCE(EXCLUDED.start_time, batch_records.start_time),
                end_time = COALESCE(EXCLUDED.end_time, batch_records.end_time),
                primary_drying_endpoint = COALESCE(EXCLUDED.primary_drying_endpoint, batch_records.primary_drying_endpoint),
                secondary_drying_endpoint = COALESCE(EXCLUDED.secondary_drying_endpoint, batch_records.secondary_drying_endpoint),
                avg_moisture_content = COALESCE(EXCLUDED.avg_moisture_content, batch_records.avg_moisture_content),
                avg_reconstitution_time = COALESCE(EXCLUDED.avg_reconstitution_time, batch_records.avg_reconstitution_time),
                defect_rate = COALESCE(EXCLUDED.defect_rate, batch_records.defect_rate),
                quality_score = COALESCE(EXCLUDED.quality_score, batch_records.quality_score),
                batch_status = COALESCE(EXCLUDED.batch_status, batch_records.batch_status),
                notes = COALESCE(EXCLUDED.notes, batch_records.notes)
        """)

        params = [
            {
                'device_id': row[0],
                'batch_id': row[1],
                'timestamp': row[2],
                'update_type': row[3],
                'freeze_profile_id': row[4],
                'formula_id': row[5],
                'start_time': row[6],
                'end_time': row[7],
                'primary_drying_endpoint': row[8],
                'secondary_drying_endpoint': row[9],
                'avg_moisture_content': row[10],
                'avg_reconstitution_time': row[11],
                'defect_rate': row[12],
                'quality_score': row[13],
                'batch_status': row[14],
                'notes': row[15],
            }
            for row in data
        ]

        await session.execute(stmt, params)

    async def _write_fallback(self, item: WriteItem) -> None:
        """写入降级文件（单条）"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = self._fallback_dir / f"fallback_{timestamp}.jsonl"

            record = {
                'data_type': item.data_type.value,
                'data': item.data,
                'received_at': item.received_at
            }

            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

            self._increment_metric("total_fallback")

        except Exception as e:
            print(f"[{self.service_id}] 降级文件写入失败: {e}")
            self._increment_metric("errors")

    async def _write_fallback_batch(self, batch: List[WriteItem]) -> None:
        """批量写入降级文件"""
        for item in batch:
            await self._write_fallback(item)

    async def _load_fallback_data(self) -> None:
        """加载降级文件中的数据"""
        try:
            loaded_count = 0
            for filepath in sorted(self._fallback_dir.glob("fallback_*.jsonl")):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue

                            record = json.loads(line)
                            item = WriteItem(
                                data_type=DataType(record['data_type']),
                                data=record['data'],
                                received_at=record.get('received_at', 0.0)
                            )

                            await self._write_queue.put(item)
                            loaded_count += 1

                    filepath.unlink()
                    print(f"[{self.service_id}] 已加载降级文件: {filepath.name} ({loaded_count}条)")

                except Exception as e:
                    print(f"[{self.service_id}] 加载降级文件失败 {filepath}: {e}")

            if loaded_count > 0:
                print(f"[{self.service_id}] 共加载降级数据: {loaded_count}条")

        except Exception as e:
            print(f"[{self.service_id}] 加载降级数据异常: {e}")

    async def _reconnect_loop(self) -> None:
        """数据库重连循环"""
        while self._running:
            try:
                if not self._db_connected:
                    await self._reconnect_db()
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{self.service_id}] 重连循环异常: {e}")
                await asyncio.sleep(5)

    async def _flush_queue(self) -> None:
        """刷新队列中的所有数据"""
        print(f"[{self.service_id}] 刷新队列剩余数据...")

        remaining: List[WriteItem] = []
        while not self._write_queue.empty():
            try:
                item = self._write_queue.get_nowait()
                remaining.append(item)
            except asyncio.QueueEmpty:
                break

        if remaining:
            if self._db_connected:
                success = await self._write_batch(remaining)
                if not success:
                    await self._write_fallback_batch(remaining)
            else:
                await self._write_fallback_batch(remaining)

            print(f"[{self.service_id}] 已处理剩余数据: {len(remaining)}条")


async def main() -> None:
    """主函数"""
    redis_config = RedisConfig(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', '6379')),
        db=int(os.environ.get('REDIS_DB', '0'))
    )

    db_config = DBConfig(
        url=os.environ.get(
            'DATABASE_URL',
            'postgresql+asyncpg://postgres:postgres@localhost:5432/freeze_dryer'
        )
    )

    service = DBWriterService(redis_config, db_config)

    print("=" * 60)
    print("数据库写入微服务启动")
    print(f"服务ID: {service.service_id}")
    print(f"服务类型: {service.service_type}")
    print(f"Redis: {redis_config.host}:{redis_config.port}")
    print(f"数据库URL: {db_config.url}")
    print(f"批量大小: {service._batch_size}")
    print(f"刷新间隔: {service._flush_interval}s")
    print("=" * 60)

    try:
        await service.start()

        while True:
            await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\n正在停止服务...")
        await service.stop()
        print("服务已安全退出")
    except Exception as e:
        print(f"服务异常: {e}")
        await service.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
