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
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    MicroserviceBase, RedisConfig,
    CHANNELS, SERVICE_IDS, MESSAGE_TYPES,
    TelemetryData, EndpointDetection, PressureRiseTest, BatchRecord,
    MessageFactory, validate_message, extract_payload,
    config_loader, EndpointConfig
)

from modules.endpoint_detector import SignalFilter, FirstDerivativeDetector, AutoEncoderDetector


@dataclass
class DeviceState:
    device_id: int
    batch_id: Optional[str] = None
    current_phase: str = "idle"
    phase_start_time: Optional[float] = None
    
    temp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    vacuum_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    timestamp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    
    filtered_temp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    filtered_vacuum_history: Deque[float] = field(default_factory=lambda: deque(maxlen=360))
    
    vacuum_stability_score: float = 1.0
    signal_quality_history: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    
    primary_endpoint_detected: bool = False
    secondary_endpoint_detected: bool = False
    primary_endpoint_time: Optional[float] = None
    secondary_endpoint_time: Optional[float] = None
    
    primary_confirmation_count: int = 0
    secondary_confirmation_count: int = 0
    autoencoder_confirmation_count: int = 0
    prt_confirmation_count: int = 0
    
    prt_in_progress: bool = False
    prt_start_time: Optional[float] = None
    prt_initial_pressure: Optional[float] = None
    prt_measurements: List[Tuple[float, float]] = field(default_factory=list)
    last_prt_time: float = 0.0
    
    training_data: List[np.ndarray] = field(default_factory=list)
    autoencoder_trained: bool = False


class PressureRiseTestManager:
    
    def __init__(self, config: Dict):
        self.enabled = config.get('enabled', True)
        self.test_duration = config.get('test_duration_seconds', 120)
        self.measurement_interval = config.get('measurement_interval_seconds', 5)
        self.endpoint_threshold = config.get('endpoint_threshold_pa_per_min', 0.05)
        self.min_interval = config.get('min_interval_between_tests_minutes', 30) * 60
        self.auto_trigger = config.get('auto_trigger_enabled', True)
        self.min_data_quality = config.get('min_data_quality', 0.6)
        self.confirmation_count = config.get('confirmation_count', 2)
        
        self._test_results: Deque[bool] = deque(maxlen=5)
        self._confirmed_endpoint = False
    
    def start_test(self, device_state: DeviceState, initial_pressure: float) -> None:
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
        if not device_state.prt_in_progress:
            return
        
        now = time.time()
        device_state.prt_measurements.append((now, pressure))
    
    def _filter_pressure_data(self, times: np.ndarray, pressures: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        if len(pressures) < 5:
            return times, pressures, 1.0
        
        median = np.median(pressures)
        mad = np.median(np.abs(pressures - median))
        threshold = 3.0 * mad / 0.6745
        
        mask = np.abs(pressures - median) <= threshold
        filtered_times = times[mask]
        filtered_pressures = pressures[mask]
        
        if len(filtered_pressures) < 5:
            return times, pressures, 0.3
        
        if len(filtered_pressures) >= 7:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(filtered_pressures, kernel, mode='same')
        else:
            smoothed = filtered_pressures
        
        dt = (filtered_times - filtered_times[0]) / 60.0
        slope, intercept = np.polyfit(dt, filtered_pressures, 1)
        predicted = slope * dt + intercept
        ss_res = np.sum((filtered_pressures - predicted) ** 2)
        ss_tot = np.sum((filtered_pressures - np.mean(filtered_pressures)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        
        consistency = 1.0 - min(1.0, np.std(filtered_pressures) / (np.mean(np.abs(filtered_pressures)) + 1e-10))
        
        quality_score = float(max(0.0, min(1.0, (r_squared * 0.6 + consistency * 0.4))))
        
        return filtered_times, smoothed, quality_score
    
    def check_test_complete(self, device_state: DeviceState) -> Optional[PressureRiseTest]:
        if not device_state.prt_in_progress:
            return None
        
        now = time.time()
        elapsed = now - device_state.prt_start_time
        
        if elapsed < self.test_duration:
            return None
        
        measurements = device_state.prt_measurements
        if len(measurements) < 5:
            device_state.prt_in_progress = False
            return None
        
        times = np.array([m[0] for m in measurements])
        pressures = np.array([m[1] for m in measurements])
        
        filtered_times, filtered_pressures, quality_score = self._filter_pressure_data(times, pressures)
        
        if quality_score < self.min_data_quality:
            device_state.prt_in_progress = False
            device_state.prt_measurements = []
            return None
        
        dt = (filtered_times - filtered_times[0]) / 60.0
        slope, intercept = np.polyfit(dt, filtered_pressures, 1)
        
        pressure_rise_rate = float(slope)
        is_endpoint = pressure_rise_rate < self.endpoint_threshold
        
        self._test_results.append(is_endpoint)
        if len(self._test_results) >= self.confirmation_count:
            recent_results = list(self._test_results)[-self.confirmation_count:]
            if all(recent_results):
                self._confirmed_endpoint = True
        
        result = PressureRiseTest(
            device_id=device_state.device_id,
            batch_id=device_state.batch_id,
            test_start_time=datetime.fromtimestamp(device_state.prt_start_time, tz=timezone.utc).isoformat(),
            test_end_time=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            initial_pressure_pa=float(filtered_pressures[0]),
            final_pressure_pa=float(filtered_pressures[-1]),
            pressure_rise_pa_per_min=pressure_rise_rate,
            test_duration_seconds=int(elapsed),
            is_endpoint_detected=self._confirmed_endpoint,
            detection_confidence=float(quality_score * min(1.0, max(0.0, 1.0 - pressure_rise_rate / max(self.endpoint_threshold, 1e-10)))),
            test_status="completed"
        )
        
        device_state.prt_in_progress = False
        device_state.prt_measurements = []
        
        return result
    
    def reset(self):
        self._test_results.clear()
        self._confirmed_endpoint = False


class EndpointDetectorService(MicroserviceBase):
    
    def __init__(self):
        super().__init__(SERVICE_IDS['ENDPOINT_DETECTOR'], 'endpoint_detector')
        self.config: EndpointConfig = config_loader.load_endpoint_config()
        self.device_states: Dict[int, DeviceState] = {}
        self.derivative_detectors: Dict[int, FirstDerivativeDetector] = {}
        self.autoencoder_detectors: Dict[int, AutoEncoderDetector] = {}
        self.prt_managers: Dict[int, PressureRiseTestManager] = {}
        self.signal_filters: Dict[int, Dict[str, SignalFilter]] = {}
        self._init_detectors()
    
    def _init_detectors(self):
        for device_id in range(1, 11):
            self.device_states[device_id] = DeviceState(device_id=device_id)
            self.derivative_detectors[device_id] = FirstDerivativeDetector(
                self.config.first_derivative
            )
            self.autoencoder_detectors[device_id] = AutoEncoderDetector(
                self.config.autoencoder
            )
            self.prt_managers[device_id] = PressureRiseTestManager(
                self.config.pressure_rise_test
            )
            self.signal_filters[device_id] = {
                'vacuum': SignalFilter(self.config.signal_filter),
                'temperature': SignalFilter(self.config.signal_filter)
            }
    
    async def _subscribe_channels(self):
        await self.subscribe(CHANNELS['TELEMETRY_RAW'], self._handle_telemetry)
        await self.subscribe(CHANNELS['PRESSURE_RISE_TEST'], self._handle_prt_command)
        await self.subscribe(CHANNELS['FLEET_COMMAND'], self._handle_fleet_command)
        await self.subscribe(CHANNELS['CONFIG_UPDATE'], self._handle_config_update)
    
    async def _on_start(self):
        print(f"[{self.service_id}] 干燥终点判定服务启动")
        print(f"  - 检测间隔: {self.config.detection_interval_seconds}s")
        print(f"  - 检测方法: 一阶导数(0.4) + 自编码器(0.3) + 压力升测试(0.3)")
        
        asyncio.create_task(self._detection_loop())
    
    async def _detection_loop(self):
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
            
            if telemetry.batch_id and state.batch_id != telemetry.batch_id:
                state.batch_id = telemetry.batch_id
                state.primary_endpoint_detected = False
                state.secondary_endpoint_detected = False
                state.primary_endpoint_time = None
                state.secondary_endpoint_time = None
                state.phase_start_time = time.time()
                state.current_phase = "primary_drying"
                self.derivative_detectors[device_id].reset()
                self.autoencoder_detectors[device_id].reset()
                self.prt_managers[device_id].reset()
                self.signal_filters[device_id]['vacuum'] = SignalFilter(self.config.signal_filter)
                self.signal_filters[device_id]['temperature'] = SignalFilter(self.config.signal_filter)
                print(f"[{self.service_id}] 设备{device_id} 新批次: {telemetry.batch_id}")
            
            if not state.batch_id:
                return
            
            avg_temp = np.mean(telemetry.temperatures)
            avg_vacuum = np.mean(telemetry.vacuum_levels)
            
            state.temp_history.append(avg_temp)
            state.vacuum_history.append(avg_vacuum)
            state.timestamp_history.append(time.time())
            
            filtered_temp, temp_stability = self.signal_filters[device_id]['temperature'].filter(avg_temp)
            filtered_vacuum, vacuum_stability = self.signal_filters[device_id]['vacuum'].filter(avg_vacuum)
            
            state.filtered_temp_history.append(filtered_temp)
            state.filtered_vacuum_history.append(filtered_vacuum)
            
            state.vacuum_stability_score = float(0.7 * vacuum_stability + 0.3 * temp_stability)
            state.signal_quality_history.append(state.vacuum_stability_score)
            
            if state.prt_in_progress:
                self.prt_managers[device_id].record_measurement(state, avg_vacuum)
            
            features = self.autoencoder_detectors[device_id].extract_features(
                telemetry.temperatures, telemetry.vacuum_levels,
                telemetry.cold_trap_temp, telemetry.heating_powers
            )
            
            if state.current_phase == "primary_drying" and not self.autoencoder_detectors[device_id]._trained:
                state.training_data.append(features)
                if len(state.training_data) >= self.config.autoencoder.get('min_training_samples', 100):
                    self.autoencoder_detectors[device_id].train(state.training_data)
                    state.autoencoder_trained = True
            
            prt_result = self.prt_managers[device_id].check_test_complete(state)
            if prt_result:
                await self._publish_prt_result(prt_result)
                
        except Exception as e:
            print(f"[{self.service_id}] 处理遥测失败: {e}")
            self._increment_metric("errors")
    
    async def _detect_endpoint(self, device_id: int, state: DeviceState):
        if len(state.filtered_temp_history) < 20:
            return
        
        timestamps = list(state.timestamp_history)
        temps = list(state.filtered_temp_history)
        vacuums = list(state.filtered_vacuum_history)
        stability_score = state.vacuum_stability_score
        
        results = {}
        confidence_scores = {}
        
        if state.current_phase == "primary_drying" and not state.primary_endpoint_detected:
            detected, derivative, inflection = self.derivative_detectors[device_id].detect_primary_endpoint(
                timestamps, temps, stability_score
            )
            results['first_derivative'] = detected
            confidence_scores['first_derivative'] = min(1.0, max(0.0, 
                1.0 - abs(derivative) / max(self.config.first_derivative.get('primary_drying_threshold', 0.05), 1e-10)
            )) * stability_score
            
        elif state.current_phase == "secondary_drying" and not state.secondary_endpoint_detected:
            detected, derivative, inflection = self.derivative_detectors[device_id].detect_secondary_endpoint(
                timestamps, temps, stability_score
            )
            results['first_derivative'] = detected
            confidence_scores['first_derivative'] = min(1.0, max(0.0,
                1.0 - abs(derivative) / max(self.config.first_derivative.get('secondary_drying_threshold', 0.02), 1e-10)
            )) * stability_score
        
        if state.autoencoder_trained and len(temps) >= 8:
            features = self.autoencoder_detectors[device_id].extract_features(
                temps[-8:], vacuums[-2:], -70.0, [50.0] * 8
            )
            recon_error, detected = self.autoencoder_detectors[device_id].predict(features, stability_score)
            results['autoencoder'] = detected
            confidence_scores['autoencoder'] = min(1.0, max(0.0,
                recon_error / max(self.config.autoencoder.get('threshold', 0.1), 1e-10)
            )) * stability_score
        else:
            results['autoencoder'] = False
            confidence_scores['autoencoder'] = 0.0
        
        results['pressure_rise_test'] = self.prt_managers[device_id]._confirmed_endpoint
        confidence_scores['pressure_rise_test'] = 0.9 if results['pressure_rise_test'] else 0.0
        
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
        phase = state.current_phase
        now = datetime.now(timezone.utc)
        
        duration_hours = None
        if state.phase_start_time:
            duration_hours = (time.time() - state.phase_start_time) / 3600.0
        
        estimated_saving = None
        if duration_hours:
            default_duration = 24.0 if phase == "primary_drying" else 8.0
            if duration_hours < default_duration:
                energy_per_hour = 5.0
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
        
        if phase == "primary_drying":
            state.primary_endpoint_detected = True
            state.primary_endpoint_time = time.time()
            state.current_phase = "secondary_drying"
            state.phase_start_time = time.time()
            print(f"[{self.service_id}] 设备{device_id} 一次干燥终点判定, 批次: {state.batch_id}, "
                  f"置信度: {confidence:.3f}, 时长: {duration_hours:.1f}h")
            
            await self._publish_batch_record(device_id, state, "primary_endpoint")
            
        elif phase == "secondary_drying":
            state.secondary_endpoint_detected = True
            state.secondary_endpoint_time = time.time()
            state.current_phase = "completed"
            print(f"[{self.service_id}] 设备{device_id} 二次干燥终点判定, 批次: {state.batch_id}, "
                  f"置信度: {confidence:.3f}, 时长: {duration_hours:.1f}h")
            
            await self._publish_batch_record(device_id, state, "secondary_endpoint")
            await self._publish_batch_record(device_id, state, "complete")
    
    async def _publish_prt_result(self, result: PressureRiseTest):
        message = MessageFactory.create_pressure_rise_test(result, self.service_id)
        await self.publish(CHANNELS['PRESSURE_RISE_TEST'], message)
        self._increment_metric("messages_published")
        print(f"[{self.service_id}] 设备{result.device_id} 压力升测试完成, "
              f"速率: {result.pressure_rise_pa_per_min:.4f} Pa/min, "
              f"终点: {result.is_endpoint_detected}")
    
    async def _publish_batch_record(self, device_id: int, state: DeviceState, update_type: str):
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
