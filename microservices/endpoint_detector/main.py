"""
干燥终点判定微服务
实现三种判定方法：
1. 一阶导数法 - 检测温度曲线拐点
2. 自编码器 - 学习正常模式，检测模式变化
3. 压力升测试 - 主动测试干燥室压力变化

组合决策：加权投票
"""

import asyncio
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    MicroserviceBase, RedisConfig,
    CHANNELS, SERVICE_IDS, MESSAGE_TYPES,
    TelemetryData, EndpointDetection, PressureRiseTest, BatchRecord,
    MessageFactory, validate_message, extract_payload,
    config_loader, EndpointConfig
)


@dataclass
class DeviceState:
    """设备状态"""
    device_id: int
    batch_id: Optional[str] = None
    current_phase: str = "idle"  # idle, freezing, primary_drying, secondary_drying, completed
    phase_start_time: Optional[float] = None
    
    # 数据缓存
    temp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))  # 1小时（10秒间隔）
    vacuum_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    timestamp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    
    # 判定状态
    primary_endpoint_detected: bool = False
    secondary_endpoint_detected: bool = False
    primary_endpoint_time: Optional[float] = None
    secondary_endpoint_time: Optional[float] = None
    
    # 压力升测试状态
    prt_in_progress: bool = False
    prt_start_time: Optional[float] = None
    prt_initial_pressure: Optional[float] = None
    prt_measurements: List[Tuple[float, float]] = field(default_factory=list)
    last_prt_time: float = 0.0
    
    # 自编码器训练数据
    training_data: List[np.ndarray] = field(default_factory=list)
    autoencoder_trained: bool = False


class FirstDerivativeDetector:
    """一阶导数法终点检测"""
    
    def __init__(self, config: Dict):
        self.window_size = config.get('window_size', 10)
        self.poly_order = config.get('poly_order', 2)
        self.primary_threshold = config.get('primary_drying_threshold', 0.05)
        self.secondary_threshold = config.get('secondary_drying_threshold', 0.02)
        self.consecutive_points = config.get('consecutive_points', 5)
        self._consecutive_count_primary = 0
        self._consecutive_count_secondary = 0
    
    def savitzky_golay(self, y: np.ndarray, window_size: int, poly_order: int) -> np.ndarray:
        """Savitzky-Golay平滑滤波"""
        if len(y) < window_size:
            return y
        
        order = min(poly_order, window_size - 1)
        half_window = window_size // 2
        
        # 计算多项式系数
        coeffs = np.zeros(window_size)
        for i in range(window_size):
            x = np.arange(-half_window, half_window + 1)
            coeffs[i] = np.polyval(np.polyfit(x, y[i:i + window_size] if i + window_size <= len(y) else y[-window_size:], order), 0)
        
        return coeffs
    
    def compute_derivative(self, timestamps: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算一阶导数"""
        if len(timestamps) < 2:
            return np.array([]), np.array([])
        
        # 先平滑
        smoothed = self.savitzky_golay(values, self.window_size, self.poly_order)
        
        # 计算导数（℃/min）
        dt = np.diff(timestamps) / 60.0  # 转换为分钟
        dy = np.diff(smoothed)
        
        # 避免除以零
        dt = np.where(dt == 0, 1e-10, dt)
        derivatives = dy / dt
        
        return derivatives, smoothed
    
    def detect_primary_endpoint(self, timestamps: List[float], temps: List[float]) -> Tuple[bool, float, float]:
        """检测一次干燥终点（温度上升速率下降到阈值以下）"""
        if len(timestamps) < self.window_size * 2:
            return False, 0.0, 0.0
        
        ts_arr = np.array(timestamps)
        temp_arr = np.array(temps)
        
        derivatives, smoothed = self.compute_derivative(ts_arr, temp_arr)
        
        if len(derivatives) == 0:
            return False, 0.0, 0.0
        
        recent_deriv = derivatives[-self.consecutive_points:]
        avg_derivative = np.mean(np.abs(recent_deriv))
        
        # 一次干燥终点：温度上升速率 < 阈值（冰升华完成，温度开始快速上升后趋于稳定）
        is_endpoint = avg_derivative < self.primary_threshold
        
        if is_endpoint:
            self._consecutive_count_primary += 1
        else:
            self._consecutive_count_primary = 0
        
        confirmed = self._consecutive_count_primary >= self.consecutive_points
        return confirmed, float(avg_derivative), float(smoothed[-1]) if len(smoothed) > 0 else 0.0
    
    def detect_secondary_endpoint(self, timestamps: List[float], temps: List[float]) -> Tuple[bool, float, float]:
        """检测二次干燥终点（温度变化率极小）"""
        if len(timestamps) < self.window_size * 2:
            return False, 0.0, 0.0
        
        ts_arr = np.array(timestamps)
        temp_arr = np.array(temps)
        
        derivatives, smoothed = self.compute_derivative(ts_arr, temp_arr)
        
        if len(derivatives) == 0:
            return False, 0.0, 0.0
        
        recent_deriv = derivatives[-self.consecutive_points:]
        avg_derivative = np.mean(np.abs(recent_deriv))
        
        # 二次干燥终点：温度变化率 < 更小的阈值（解析干燥完成）
        is_endpoint = avg_derivative < self.secondary_threshold
        
        if is_endpoint:
            self._consecutive_count_secondary += 1
        else:
            self._consecutive_count_secondary = 0
        
        confirmed = self._consecutive_count_secondary >= self.consecutive_points
        return confirmed, float(avg_derivative), float(smoothed[-1]) if len(smoothed) > 0 else 0.0


class AutoencoderDetector:
    """自编码器异常检测"""
    
    def __init__(self, config: Dict):
        self.input_dim = config.get('input_dim', 11)
        self.latent_dim = config.get('latent_dim', 4)
        self.hidden_layers = config.get('hidden_layers', [32, 16])
        self.threshold = config.get('threshold', 0.1)
        self.min_training_samples = config.get('min_training_samples', 100)
        
        # 简化的自编码器（不依赖外部深度学习库，使用numpy实现）
        self._encoder_weights = None
        self._decoder_weights = None
        self._trained = False
        self._recon_errors: Deque[float] = deque(maxlen=100)
    
    def extract_features(self, temps: List[float], vacuums: List[float], 
                        cold_trap: float, powers: List[float]) -> np.ndarray:
        """从遥测数据提取特征"""
        temp_arr = np.array(temps)
        vacuum_arr = np.array(vacuums)
        power_arr = np.array(powers)
        
        features = np.array([
            np.mean(temp_arr),       # 平均温度
            np.std(temp_arr),        # 温度标准差
            np.max(temp_arr) - np.min(temp_arr),  # 温度差
            np.mean(vacuum_arr),     # 平均真空度
            np.std(vacuum_arr),      # 真空度标准差
            cold_trap,               # 冷阱温度
            np.mean(power_arr),      # 平均功率
            np.std(power_arr),       # 功率标准差
            temp_arr[-1] - temp_arr[0] if len(temp_arr) > 1 else 0,  # 温度趋势
            vacuum_arr[-1] - vacuum_arr[0] if len(vacuum_arr) > 1 else 0,  # 真空趋势
            power_arr[-1] - power_arr[0] if len(power_arr) > 1 else 0,    # 功率趋势
        ])
        
        return features
    
    def train(self, training_data: List[np.ndarray]) -> None:
        """训练自编码器（简化的PCA-based自编码器）"""
        if len(training_data) < self.min_training_samples:
            return
        
        data_matrix = np.array(training_data)
        
        # 标准化
        self._mean = np.mean(data_matrix, axis=0)
        self._std = np.std(data_matrix, axis=0) + 1e-10
        normalized = (data_matrix - self._mean) / self._std
        
        # PCA降维作为编码器
        cov_matrix = np.cov(normalized.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 取前latent_dim个主成分
        idx = np.argsort(eigenvalues)[::-1]
        self._encoder_weights = eigenvectors[:, idx[:self.latent_dim]].real
        self._decoder_weights = self._encoder_weights.T
        
        # 计算重构误差阈值
        recon_errors = []
        for sample in normalized:
            encoded = sample @ self._encoder_weights
            decoded = encoded @ self._decoder_weights
            error = np.mean((sample - decoded) ** 2)
            recon_errors.append(error)
        
        # 阈值设为均值 + 3*标准差
        self.threshold = float(np.mean(recon_errors) + 3 * np.std(recon_errors))
        self._trained = True
        print(f"[Autoencoder] 训练完成，样本数: {len(training_data)}, 阈值: {self.threshold:.4f}")
    
    def predict(self, features: np.ndarray) -> Tuple[float, bool]:
        """预测重构误差"""
        if not self._trained:
            return 0.0, False
        
        # 标准化
        normalized = (features - self._mean) / self._std
        
        # 编码解码
        encoded = normalized @ self._encoder_weights
        decoded = encoded @ self._decoder_weights
        
        # 计算重构误差（MSE）
        recon_error = float(np.mean((normalized - decoded) ** 2))
        self._recon_errors.append(recon_error)
        
        # 检测模式变化（误差显著增大表示终点到达）
        is_endpoint = recon_error > self.threshold
        
        return recon_error, is_endpoint


class PressureRiseTestManager:
    """压力升测试管理器"""
    
    def __init__(self, config: Dict):
        self.enabled = config.get('enabled', True)
        self.test_duration = config.get('test_duration_seconds', 120)
        self.measurement_interval = config.get('measurement_interval_seconds', 5)
        self.endpoint_threshold = config.get('endpoint_threshold_pa_per_min', 0.05)
        self.min_interval = config.get('min_interval_between_tests_minutes', 30) * 60
        self.auto_trigger = config.get('auto_trigger_enabled', True)
    
    def start_test(self, device_state: DeviceState, initial_pressure: float) -> None:
        """开始压力升测试"""
        if not self.enabled:
            return
        
        now = time.time()
        if now - device_state.last_prt_time < self.min_interval:
            return
        
        device_state.prt_in_progress = True
        device_state.prt_start_time = now
        device_state.prt_initial_pressure = initial_pressure
        device_state.prt_measurements = [(now, initial_pressure)]
        device_state.last_prt_time = now
    
    def record_measurement(self, device_state: DeviceState, pressure: float) -> None:
        """记录测量值"""
        if not device_state.prt_in_progress:
            return
        
        now = time.time()
        device_state.prt_measurements.append((now, pressure))
    
    def check_test_complete(self, device_state: DeviceState) -> Optional[PressureRiseTest]:
        """检查测试是否完成并计算结果"""
        if not device_state.prt_in_progress:
            return None
        
        now = time.time()
        elapsed = now - device_state.prt_start_time
        
        if elapsed < self.test_duration:
            return None
        
        # 计算压力升速率
        measurements = device_state.prt_measurements
        if len(measurements) < 2:
            device_state.prt_in_progress = False
            return None
        
        times = np.array([m[0] for m in measurements])
        pressures = np.array([m[1] for m in measurements])
        
        # 线性拟合计算压力升速率（Pa/min）
        dt = (times - times[0]) / 60.0  # 转换为分钟
        slope, intercept = np.polyfit(dt, pressures, 1)
        
        pressure_rise_rate = float(slope)
        is_endpoint = pressure_rise_rate < self.endpoint_threshold
        
        result = PressureRiseTest(
            device_id=device_state.device_id,
            batch_id=device_state.batch_id,
            test_start_time=datetime.fromtimestamp(device_state.prt_start_time, tz=timezone.utc).isoformat(),
            test_end_time=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            initial_pressure_pa=float(pressures[0]),
            final_pressure_pa=float(pressures[-1]),
            pressure_rise_pa_per_min=pressure_rise_rate,
            test_duration_seconds=int(elapsed),
            is_endpoint_detected=is_endpoint,
            detection_confidence=min(1.0, max(0.0, 1.0 - pressure_rise_rate / max(self.endpoint_threshold, 1e-10))),
            test_status="completed"
        )
        
        # 重置测试状态
        device_state.prt_in_progress = False
        device_state.prt_measurements = []
        
        return result


class EndpointDetectorService(MicroserviceBase):
    """干燥终点判定服务"""
    
    def __init__(self):
        super().__init__(SERVICE_IDS['ENDPOINT_DETECTOR'], 'endpoint_detector')
        self.config: EndpointConfig = config_loader.load_endpoint_config()
        self.device_states: Dict[int, DeviceState] = {}
        self.derivative_detectors: Dict[int, FirstDerivativeDetector] = {}
        self.autoencoder_detectors: Dict[int, AutoencoderDetector] = {}
        self.prt_managers: Dict[int, PressureRiseTestManager] = {}
        self._init_detectors()
    
    def _init_detectors(self):
        """为每台设备初始化检测器"""
        for device_id in range(1, 11):
            self.device_states[device_id] = DeviceState(device_id=device_id)
            self.derivative_detectors[device_id] = FirstDerivativeDetector(
                self.config.first_derivative
            )
            self.autoencoder_detectors[device_id] = AutoencoderDetector(
                self.config.autoencoder
            )
            self.prt_managers[device_id] = PressureRiseTestManager(
                self.config.pressure_rise_test
            )
    
    async def _subscribe_channels(self):
        """订阅频道"""
        await self.subscribe(CHANNELS['TELEMETRY_RAW'], self._handle_telemetry)
        await self.subscribe(CHANNELS['PRESSURE_RISE_TEST'], self._handle_prt_command)
        await self.subscribe(CHANNELS['FLEET_COMMAND'], self._handle_fleet_command)
        await self.subscribe(CHANNELS['CONFIG_UPDATE'], self._handle_config_update)
    
    async def _on_start(self):
        """启动时执行"""
        print(f"[{self.service_id}] 干燥终点判定服务启动")
        print(f"  - 检测间隔: {self.config.detection_interval_seconds}s")
        print(f"  - 检测方法: 一阶导数(0.4) + 自编码器(0.3) + 压力升测试(0.3)")
        
        # 启动检测循环
        asyncio.create_task(self._detection_loop())
    
    async def _detection_loop(self):
        """主检测循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.detection_interval_seconds)
                
                for device_id, state in self.device_states.items():
                    if state.current_phase not in ['primary_drying', 'secondary_drying']:
                        continue
                    
                    if not state.batch_id:
                        continue
                    
                    await self._detect_endpoint(device_id, state)
                    
            except Exception as e:
                print(f"[{self.service_id}] 检测循环异常: {e}")
                await asyncio.sleep(5)
    
    async def _handle_telemetry(self, message: Dict):
        """处理遥测数据"""
        try:
            if not validate_message(message, MESSAGE_TYPES['TELEMETRY']):
                return
            
            payload = extract_payload(message)
            telemetry = TelemetryData(**payload)
            device_id = telemetry.device_id
            
            state = self.device_states.get(device_id)
            if not state:
                return
            
            self._increment_metric("messages_received")
            
            # 更新批次ID
            if telemetry.batch_id and state.batch_id != telemetry.batch_id:
                state.batch_id = telemetry.batch_id
                state.primary_endpoint_detected = False
                state.secondary_endpoint_detected = False
                state.primary_endpoint_time = None
                state.secondary_endpoint_time = None
                state.phase_start_time = time.time()
                state.current_phase = "primary_drying"  # 默认从一次干燥开始
                print(f"[{self.service_id}] 设备{device_id} 新批次: {telemetry.batch_id}")
            
            if not state.batch_id:
                return
            
            # 缓存数据
            avg_temp = np.mean(telemetry.temperatures)
            avg_vacuum = np.mean(telemetry.vacuum_levels)
            
            state.temp_history.append(avg_temp)
            state.vacuum_history.append(avg_vacuum)
            state.timestamp_history.append(time.time())
            
            # 记录压力升测试数据
            if state.prt_in_progress:
                self.prt_managers[device_id].record_measurement(state, avg_vacuum)
            
            # 收集自编码器训练数据
            features = self.autoencoder_detectors[device_id].extract_features(
                telemetry.temperatures, telemetry.vacuum_levels,
                telemetry.cold_trap_temp, telemetry.heating_powers
            )
            
            if state.current_phase == "primary_drying" and not self.autoencoder_detectors[device_id]._trained:
                state.training_data.append(features)
                if len(state.training_data) >= self.config.autoencoder.get('min_training_samples', 100):
                    self.autoencoder_detectors[device_id].train(state.training_data)
                    state.autoencoder_trained = True
            
            # 检查压力升测试是否完成
            prt_result = self.prt_managers[device_id].check_test_complete(state)
            if prt_result:
                await self._publish_prt_result(prt_result)
                
        except Exception as e:
            print(f"[{self.service_id}] 处理遥测失败: {e}")
            self._increment_metric("errors")
    
    async def _detect_endpoint(self, device_id: int, state: DeviceState):
        """检测终点"""
        if len(state.temp_history) < 20:
            return
        
        timestamps = list(state.timestamp_history)
        temps = list(state.temp_history)
        vacuums = list(state.vacuum_history)
        
        results = {}
        confidence_scores = {}
        
        # 1. 一阶导数法
        if state.current_phase == "primary_drying" and not state.primary_endpoint_detected:
            detected, derivative, inflection = self.derivative_detectors[device_id].detect_primary_endpoint(
                timestamps, temps
            )
            results['first_derivative'] = detected
            confidence_scores['first_derivative'] = min(1.0, max(0.0, 
                1.0 - abs(derivative) / max(self.config.first_derivative.get('primary_drying_threshold', 0.05), 1e-10)
            ))
            
        elif state.current_phase == "secondary_drying" and not state.secondary_endpoint_detected:
            detected, derivative, inflection = self.derivative_detectors[device_id].detect_secondary_endpoint(
                timestamps, temps
            )
            results['first_derivative'] = detected
            confidence_scores['first_derivative'] = min(1.0, max(0.0,
                1.0 - abs(derivative) / max(self.config.first_derivative.get('secondary_drying_threshold', 0.02), 1e-10)
            ))
        
        # 2. 自编码器
        if state.autoencoder_trained and len(temps) >= 8:
            features = self.autoencoder_detectors[device_id].extract_features(
                [temps[-1]] * 8, [vacuums[-1]] * 2, -70.0, [50.0] * 8
            )
            recon_error, detected = self.autoencoder_detectors[device_id].predict(features)
            results['autoencoder'] = detected
            confidence_scores['autoencoder'] = min(1.0, max(0.0,
                recon_error / max(self.config.autoencoder.get('threshold', 0.1), 1e-10)
            ))
        else:
            results['autoencoder'] = False
            confidence_scores['autoencoder'] = 0.0
        
        # 3. 压力升测试（如果最近有测试结果）
        # 这里简化处理，实际应结合最近的PRT结果
        results['pressure_rise_test'] = False
        confidence_scores['pressure_rise_test'] = 0.0
        
        # 组合决策
        weights = self.config.combined_decision.get('weights', {
            'first_derivative': 0.4,
            'autoencoder': 0.3,
            'pressure_rise_test': 0.3
        })
        
        combined_confidence = 0.0
        weighted_vote = 0.0
        
        for method, result in results.items():
            weight = weights.get(method, 0.0)
            confidence = confidence_scores.get(method, 0.0)
            combined_confidence += weight * confidence
            if result:
                weighted_vote += weight
        
        is_endpoint = weighted_vote >= self.config.combined_decision.get('min_confidence', 0.7)
        combined_confidence = min(1.0, combined_confidence)
        
        if is_endpoint and combined_confidence >= self.config.combined_decision.get('min_confidence', 0.7):
            await self._publish_endpoint_detection(
                device_id, state, results, combined_confidence, temps
            )
    
    async def _publish_endpoint_detection(self, device_id: int, state: DeviceState, 
                                           results: Dict, confidence: float, temps: List[float]):
        """发布终点检测结果"""
        phase = state.current_phase
        now = datetime.now(timezone.utc)
        
        # 计算阶段持续时间
        duration_hours = None
        if state.phase_start_time:
            duration_hours = (time.time() - state.phase_start_time) / 3600.0
        
        # 计算预计节能（相比默认时间提前结束）
        estimated_saving = None
        if duration_hours:
            default_duration = 24.0 if phase == "primary_drying" else 8.0
            if duration_hours < default_duration:
                energy_per_hour = 5.0  # 假设5度电/小时
                estimated_saving = (default_duration - duration_hours) * energy_per_hour
        
        endpoint = EndpointDetection(
            device_id=device_id,
            batch_id=state.batch_id,
            cycle_phase=phase,
            detection_method="combined",
            endpoint_timestamp=now.isoformat(),
            detection_confidence=confidence,
            temp_inflection_point=float(temps[-1]) if temps else None,
            temp_first_derivative=float(np.mean(np.diff(temps[-10:]))) if len(temps) >= 10 else None,
            cycle_duration_hours=duration_hours,
            estimated_energy_saving=estimated_saving,
            is_accepted=True
        )
        
        message = MessageFactory.create_endpoint_detection(endpoint, self.service_id)
        await self.publish(CHANNELS['ENDPOINT_DETECTION'], message)
        self._increment_metric("messages_published")
        
        # 更新状态
        if phase == "primary_drying":
            state.primary_endpoint_detected = True
            state.primary_endpoint_time = time.time()
            state.current_phase = "secondary_drying"
            state.phase_start_time = time.time()
            print(f"[{self.service_id}] 设备{device_id} 一次干燥终点判定, 批次: {state.batch_id}, "
                  f"置信度: {confidence:.3f}, 时长: {duration_hours:.1f}h")
            
            # 发布批次记录更新
            await self._publish_batch_record(device_id, state, "primary_endpoint")
            
        elif phase == "secondary_drying":
            state.secondary_endpoint_detected = True
            state.secondary_endpoint_time = time.time()
            state.current_phase = "completed"
            print(f"[{self.service_id}] 设备{device_id} 二次干燥终点判定, 批次: {state.batch_id}, "
                  f"置信度: {confidence:.3f}, 时长: {duration_hours:.1f}h")
            
            # 发布批次记录更新
            await self._publish_batch_record(device_id, state, "secondary_endpoint")
            await self._publish_batch_record(device_id, state, "complete")
    
    async def _publish_prt_result(self, result: PressureRiseTest):
        """发布压力升测试结果"""
        message = MessageFactory.create_pressure_rise_test(result, self.service_id)
        await self.publish(CHANNELS['PRESSURE_RISE_TEST'], message)
        self._increment_metric("messages_published")
        print(f"[{self.service_id}] 设备{result.device_id} 压力升测试完成, "
              f"速率: {result.pressure_rise_pa_per_min:.4f} Pa/min, "
              f"终点: {result.is_endpoint_detected}")
    
    async def _publish_batch_record(self, device_id: int, state: DeviceState, update_type: str):
        """发布批次记录更新"""
        now = datetime.now(timezone.utc)
        record = BatchRecord(
            device_id=device_id,
            batch_id=state.batch_id,
            timestamp=now.isoformat(),
            update_type=update_type,
            primary_drying_endpoint=datetime.fromtimestamp(state.primary_endpoint_time, tz=timezone.utc).isoformat() if state.primary_endpoint_time else None,
            secondary_drying_endpoint=datetime.fromtimestamp(state.secondary_endpoint_time, tz=timezone.utc).isoformat() if state.secondary_endpoint_time else None,
            batch_status=state.current_phase
        )
        
        message = MessageFactory.create_batch_record(record, self.service_id)
        await self.publish(CHANNELS['DB_WRITE'], message)
        self._increment_metric("messages_published")
    
    async def _handle_prt_command(self, message: Dict):
        """处理压力升测试命令"""
        try:
            payload = extract_payload(message)
            command = payload.get('command')
            device_id = payload.get('device_id')
            
            if command == 'start' and device_id in self.device_states:
                state = self.device_states[device_id]
                if state.current_phase in ['primary_drying', 'secondary_drying']:
                    avg_vacuum = state.vacuum_history[-1] if state.vacuum_history else 0.1
                    self.prt_managers[device_id].start_test(state, avg_vacuum)
                    print(f"[{self.service_id}] 设备{device_id} 开始压力升测试")
                    
        except Exception as e:
            print(f"[{self.service_id}] 处理PRT命令失败: {e}")
    
    async def _handle_fleet_command(self, message: Dict):
        """处理群控命令"""
        try:
            if not validate_message(message, MESSAGE_TYPES['FLEET_COMMAND']):
                return
            
            payload = extract_payload(message)
            cmd = payload.get('command')
            device_id = payload.get('device_id')
            batch_id = payload.get('batch_id')
            
            if cmd == 'start_batch' and device_id in self.device_states:
                state = self.device_states[device_id]
                state.batch_id = batch_id
                state.current_phase = "primary_drying"
                state.phase_start_time = time.time()
                state.primary_endpoint_detected = False
                state.secondary_endpoint_detected = False
                print(f"[{self.service_id}] 设备{device_id} 开始新批次: {batch_id}")
                
                await self._publish_batch_record(device_id, state, "start")
                
        except Exception as e:
            print(f"[{self.service_id}] 处理群控命令失败: {e}")
    
    async def _handle_config_update(self, message: Dict):
        """处理配置更新"""
        try:
            payload = extract_payload(message)
            if payload.get('config_type') == 'endpoint':
                self.config = config_loader.load_endpoint_config()
                print(f"[{self.service_id}] 配置已更新")
        except Exception as e:
            print(f"[{self.service_id}] 配置更新失败: {e}")


if __name__ == "__main__":
    import time
    
    service = EndpointDetectorService()
    
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        asyncio.run(service.stop())
    except Exception as e:
        print(f"服务异常退出: {e}")
