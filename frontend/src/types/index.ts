export interface TelemetryData {
  device_id: number;
  shelf_id: number;
  timestamp: string;
  temperatures: number[];
  vacuum_levels: number[];
  cold_trap_temp: number;
  heating_powers: number[];
}

export interface RealtimeData {
  device_id: number;
  shelf_id: number;
  timestamp: string;
  temperatures: number[];
  temperature_diff: number;
  avg_temperature: number;
  vacuum_levels: number[];
  avg_vacuum: number;
  cold_trap_temp: number;
  heating_powers: number[];
  has_alarm: boolean;
}

export interface DeviceInfo {
  id: number;
  name: string;
  location: string;
  status: string;
}

export interface ShelfInfo {
  id: number;
  device_id: number;
  shelf_number: number;
  temp_sensor_count: number;
  vacuum_sensor_count: number;
}

export interface AlarmData {
  id: string;
  timestamp: string;
  device_id: number;
  shelf_id?: number;
  alarm_type: 'temperature_diff' | 'vacuum_abnormal' | 'cold_trap_high' | 'quality_prediction';
  severity: 'warning' | 'critical';
  message: string;
  acknowledged: boolean;
  acknowledged_by?: string;
  acknowledged_at?: string;
}

export interface PredictionResultData {
  device_id: number;
  batch_id?: string;
  moisture_content: {
    predicted: number;
    confidence: number;
    threshold: number;
    is_qualified: boolean;
  };
  reconstitution_time: {
    predicted: number;
    confidence: number;
    threshold: number;
    is_qualified: boolean;
  };
  drying_rate: number;
  is_qualified: boolean;
  timestamp?: string;
}

export interface ControlCommand {
  device_id: number;
  shelf_id: number;
  timestamp?: string;
  power_adjustments: number[];
  auto_mode: boolean;
}

export interface DeviceStats {
  shelf_id: number;
  sample_count: number;
  avg_temp: number;
  max_temp: number;
  min_temp: number;
  temp_diff: number;
  avg_vacuum: number;
  avg_cold_trap: number;
}

export interface VacuumDataPoint {
  timestamp: string;
  shelf_id: number;
  value: number;
}

export interface EndpointDetection {
  device_id: number;
  batch_id: string;
  cycle_phase: string;
  detection_method: string;
  endpoint_timestamp: string;
  detection_confidence: number;
  pressure_rise_delta: number;
  temp_inflection_point: number;
  temp_first_derivative: number;
  autoencoder_recon_error: number;
  cycle_duration_hours: number;
  estimated_energy_saving: number;
}

export interface PressureRiseTest {
  device_id: number;
  batch_id: string;
  test_start_time: string;
  test_end_time: string;
  initial_pressure_pa: number;
  final_pressure_pa: number;
  pressure_rise_pa_per_min: number;
  test_duration_seconds: number;
  is_endpoint_detected: boolean;
  detection_confidence: number;
  test_status: string;
}

export interface DefrostOptimization {
  device_id: number;
  batch_id: string;
  timestamp: string;
  estimated_frost_thickness_mm: number;
  cold_trap_temp_avg: number;
  cold_trap_temp_trend: number;
  recommended_defrost_interval_hours: number;
  recommended_heating_power_pct: number;
  estimated_energy_saving: number;
  defrost_status: string;
}

export interface DefrostStatus {
  device_id: number;
  timestamp: string;
  status: string;
  progress_pct: number;
  current_temp: number;
  target_temp: number;
  energy_consumed_kwh: number;
  batch_id: string;
}

export interface FleetSchedule {
  schedule_id: string;
  schedule_date: string;
  total_required_batches: number;
  estimated_energy_cost: number;
  optimized_energy_saving: number;
  solver_status: string;
  details: string;
  timestamp: string;
}

export interface FleetDeviceStatus {
  device_id: number;
  timestamp: string;
  batch_id: string;
  batch_status: string;
  current_phase: string;
  phase_progress_pct: number;
  estimated_completion_time: string;
  current_power_kw: number;
}

export interface ProductDefect {
  id: string;
  device_id: number;
  batch_id: string;
  timestamp: string;
  image_path: string;
  image_hash: string;
  defect_type: string;
  defect_severity: string;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  shelf_id: number;
  vial_position: string;
  is_manual_reviewed: boolean;
}

export interface BatchRecord {
  id: string;
  device_id: number;
  batch_id: string;
  timestamp: string;
  update_type: string;
  freeze_profile_id: string;
  formula_id: string;
  start_time: string;
  end_time: string;
  primary_drying_endpoint: string;
  secondary_drying_endpoint: string;
  avg_moisture_content: number;
  avg_reconstitution_time: number;
  defect_rate: number;
  quality_score: number;
  batch_status: string;
  notes: string;
}

export type DeviceStatus = 'idle' | 'running' | 'paused' | 'maintenance' | 'defrosting';

export interface FleetDevice {
  device_id: number;
  status: DeviceStatus;
  current_batch: string | null;
  batch_progress: number;
  current_schedule_id: number | null;
  last_command: string | null;
  last_update: string | null;
}

export interface FleetOverview {
  total_devices: number;
  status_summary: Record<DeviceStatus, number>;
  active_schedules: number;
  average_batch_progress: number;
  devices: FleetDevice[];
}

export interface FleetScheduleCreate {
  required_batches: number;
  time_horizon_hours: number;
  device_ids?: number[];
  priority?: string;
}

export interface ScheduledBatch {
  id: string;
  device_id: number;
  schedule_id: number;
  start_time: string;
  end_time: string;
  duration_hours: number;
  status: 'pending' | 'running' | 'completed';
  energy_cost: number;
  optimization_savings: number;
}

export interface ElectricityPrice {
  timestamp: string;
  price: number;
  period: 'peak' | 'valley' | 'normal';
}

export interface FleetStatsResponse {
  time_window_hours: number;
  commands: {
    total: number;
    completed: number;
    failed: number;
    success_rate: number;
  };
  schedules: {
    total: number;
    completed: number;
    total_batches_scheduled: number;
  };
  active_devices: number;
  optimization_score: number;
  total_batches: number;
  total_energy_cost: number;
  total_optimization_savings: number;
}

export interface DefrostStatusData {
  device_id: number;
  timestamp: string;
  is_defrosting: boolean;
  defrost_phase: 'idle' | 'preheating' | 'heating' | 'soaking' | 'cooling';
  current_heating_power_pct: number;
  elapsed_minutes: number;
  remaining_minutes: number;
  estimated_frost_thickness_mm: number | null;
  cold_trap_temp: number | null;
  vacuum_levels: (number | null)[];
  last_telemetry_timestamp: string | null;
  energy_consumed_kwh?: number;
}

export interface DefrostOptimizationData {
  id: string;
  device_id: number;
  timestamp: string;
  optimization_type: string;
  recommended_action: string;
  estimated_frost_thickness_mm: number;
  predicted_energy_saving_kwh: number;
  confidence_score: number;
  scheduled_time: string | null;
  is_approved: boolean;
  recommended_defrost_interval_hours?: number;
  recommended_heating_power_pct?: number;
}

export interface DefrostStatusHistoryItem {
  id: string;
  device_id: number;
  timestamp: string;
  is_defrosting: boolean;
  defrost_phase: string;
  current_heating_power_pct: number;
  elapsed_minutes: number;
  remaining_minutes: number;
  target_temp: number | null;
  current_temp: number | null;
  energy_consumed_kwh?: number;
}

export interface DefrostEnergyStats {
  device_id: number;
  time_window_days: number;
  optimization_stats: {
    total_optimization_count: number;
    approved_count: number;
    approval_rate: number;
    total_predicted_saving_kwh: number;
    actual_energy_saving_kwh: number;
    total_frost_removed_mm: number;
    avg_frost_thickness_mm: number;
    max_frost_thickness_mm: number;
    avg_confidence_score: number;
  };
  command_stats: {
    total_commands: number;
    start_count: number;
    stop_count: number;
    cancel_count: number;
    avg_heating_power_pct: number;
    avg_duration_minutes: number;
  };
  status_stats: {
    defrosting_ratio: number;
    avg_running_power_pct: number;
  };
}

export interface DefrostCommand {
  command: 'start' | 'stop' | 'cancel';
  heating_power_pct?: number;
  max_duration_minutes?: number;
}

export interface ColdTrapTempDataPoint {
  timestamp: string;
  temp: number;
}

export interface EndpointStatus {
  device_id: number;
  batch_id: string | null;
  batch_status: string;
  current_phase: string;
  primary_endpoint_detected: boolean;
  secondary_endpoint_detected: boolean;
  primary_drying_endpoint: string | null;
  secondary_drying_endpoint: string | null;
  cycle_start_time: string | null;
  total_cycle_hours: number | null;
  last_endpoint_detection: {
    phase: string;
    timestamp: string;
    confidence: number;
    method: string;
  } | null;
}

export interface EndpointDetectionData {
  id: string;
  device_id: number;
  batch_id: string;
  cycle_phase: string;
  detection_method: string;
  endpoint_timestamp: string;
  detection_confidence: number;
  pressure_rise_delta: number | null;
  temp_inflection_point: number | null;
  temp_first_derivative: number | null;
  autoencoder_recon_error: number | null;
  cycle_duration_hours: number;
  estimated_energy_saving: number;
  is_accepted: boolean;
  created_at: string;
}

export interface EndpointDetectionMethodConfidence {
  first_derivative: number;
  autoencoder: number;
  pressure_rise_test: number;
}

export interface EndpointCombinedDecision {
  endpoint_detected: boolean;
  current_phase: string;
  confidence: number;
  estimated_time_saving_minutes: number;
  primary_detection_time: string | null;
  secondary_detection_time: string | null;
}

export interface EndpointStats {
  device_id: number;
  time_window_days: number;
  detection_stats: {
    total_detections: number;
    accepted_detections: number;
    acceptance_rate: number;
    primary_drying_detections: number;
    secondary_drying_detections: number;
    average_confidence: number | null;
    average_cycle_duration_hours: number | null;
    average_primary_duration_hours: number | null;
    average_secondary_duration_hours: number | null;
    method_breakdown: {
      method: string;
      count: number;
      avg_confidence: number | null;
    }[];
  };
  prt_stats: {
    total_tests: number;
    endpoint_detected_tests: number;
    average_pressure_rise_pa_per_min: number | null;
    average_test_duration_seconds: number | null;
    average_confidence: number | null;
  };
  batch_stats: {
    total_batches: number;
    completed_batches: number;
    average_total_cycle_hours: number | null;
    average_phase_transition_hours: number | null;
  };
  efficiency_stats: {
    cycle_time_reduction_percent: number | null;
    average_energy_saving_kwh: number | null;
    total_energy_saving_kwh: number | null;
  };
}

export interface TemperatureDataPoint {
  timestamp: string;
  temperature: number;
  first_derivative: number;
}

export type DefectType = 'normal' | 'collapse' | 'atrophy' | 'cracking';

export interface DefectBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DefectResult {
  id: string;
  image_url: string;
  defect_type: DefectType;
  confidence: number;
  bounding_box: DefectBoundingBox;
  reviewed: boolean;
  reviewed_by?: string;
  reviewed_at?: string;
  timestamp: string;
  batch_id: string;
  device_id?: number;
}

export interface DefectDistribution {
  normal: number;
  collapse: number;
  atrophy: number;
  cracking: number;
}

export interface BatchStats {
  batch_id: string;
  defect_rate: number;
  quality_score: number;
  review_status: 'pending' | 'in_progress' | 'completed';
  distribution: DefectDistribution;
  total_images: number;
  reviewed_images: number;
}

export interface BatchDefectRecord {
  id: string;
  batch_id: string;
  image_url: string;
  defect_type: DefectType;
  confidence: number;
  timestamp: string;
}
