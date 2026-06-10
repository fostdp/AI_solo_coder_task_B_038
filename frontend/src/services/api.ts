import axios from 'axios';
import type { 
  DeviceInfo, 
  ShelfInfo, 
  RealtimeData, 
  TelemetryData,
  AlarmData, 
  PredictionResultData,
  DeviceStats,
  EndpointDetection,
  PressureRiseTest,
  DefrostOptimization,
  DefrostStatus,
  FleetSchedule,
  FleetDeviceStatus,
  ProductDefect,
  BatchRecord,
  EndpointStatus,
  EndpointDetectionData,
  EndpointStats,
  TemperatureDataPoint,
  DefectResult,
  BatchStats,
  BatchDefectRecord,
  DefrostStatusData,
  DefrostOptimizationData,
  DefrostStatusHistoryItem,
  DefrostEnergyStats,
  DefrostCommand,
  ColdTrapTempDataPoint,
  FleetScheduleCreate,
  FleetDevice,
  FleetOverview,
  FleetStatsResponse,
  ScheduledBatch,
  ElectricityPrice
} from '@/types';

const API_BASE_URL = '__API_BASE_URL__';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const deviceApi = {
  getDevices: (): Promise<DeviceInfo[]> => 
    api.get('/api/devices').then(res => res.data),
  
  getDevice: (id: number): Promise<DeviceInfo> => 
    api.get(`/api/devices/${id}`).then(res => res.data),
  
  getShelves: (deviceId: number): Promise<ShelfInfo[]> => 
    api.get(`/api/devices/${deviceId}/shelves`).then(res => res.data),
};

export const dataApi = {
  sendTelemetry: (data: TelemetryData) => 
    api.post('/api/data/telemetry', data),
  
  getRealtimeData: (deviceId: number): Promise<RealtimeData[]> => 
    api.get(`/api/data/realtime/${deviceId}`).then(res => res.data),
  
  getHistory: (params: {
    device_id: number;
    shelf_id?: number;
    start_time?: string;
    end_time?: string;
    limit?: number;
  }) => api.get('/api/data/history', { params }).then(res => res.data),
  
  getDeviceStats: (deviceId: number, hours: number = 1): Promise<{
    device_id: number;
    time_window_hours: number;
    stats: DeviceStats[];
  }> => api.get(`/api/data/stats/${deviceId}`, { params: { hours } }).then(res => res.data),
};

export const controlApi = {
  sendCommand: (command: {
    device_id: number;
    shelf_id: number;
    power_adjustments: number[];
    auto_mode: boolean;
  }) => api.post('/api/control/power', command).then(res => res.data),
  
  setMode: (deviceId: number, autoMode: boolean) => 
    api.put('/api/control/mode', { device_id: deviceId, auto_mode: autoMode }).then(res => res.data),
  
  getLatestCommand: (deviceId: number, shelfId?: number) => 
    api.get(`/api/control/latest/${deviceId}`, { params: { shelf_id: shelfId } }).then(res => res.data),
  
  calculateAdjustment: (deviceId: number, shelfId: number) => 
    api.get(`/api/control/calculate/${deviceId}/${shelfId}`).then(res => res.data),
  
  getThreshold: () => api.get('/api/control/threshold').then(res => res.data),
  
  setThreshold: (threshold: number) => 
    api.put('/api/control/threshold', null, { params: { threshold } }).then(res => res.data),
  
  getStatus: (deviceId: number) => 
    api.get(`/api/control/status/${deviceId}`).then(res => res.data),
};

export const predictionApi = {
  predictQuality: (deviceId: number, batchId?: string) => 
    api.post('/api/prediction/quality', null, { params: { device_id: deviceId, batch_id: batchId } }).then(res => res.data),
  
  getResults: (deviceId: number, limit: number = 10) => 
    api.get(`/api/prediction/result/${deviceId}`, { params: { limit } }).then(res => res.data),
  
  getModelInfo: (deviceId: number) => 
    api.get(`/api/prediction/model/${deviceId}`).then(res => res.data),
  
  setThresholds: (moistureMax: number, reconstitutionMax: number) => 
    api.put('/api/prediction/thresholds', null, { 
      params: { moisture_max: moistureMax, reconstitution_max: reconstitutionMax } 
    }).then(res => res.data),
};

export const alarmApi = {
  getCurrentAlarms: (): Promise<{ count: number; alarms: AlarmData[] }> => 
    api.get('/api/alarm/current').then(res => res.data),
  
  getHistory: (params?: {
    device_id?: number;
    alarm_type?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
  }): Promise<{ count: number; alarms: AlarmData[] }> => 
    api.get('/api/alarm/history', { params }).then(res => res.data),
  
  acknowledge: (alarmId: string, acknowledgedBy: string) => 
    api.post('/api/alarm/acknowledge', { 
      alarm_id: alarmId, 
      acknowledged_by: acknowledgedBy 
    }).then(res => res.data),
  
  checkAlarms: (deviceId: number, shelfId: number) => 
    api.post('/api/alarm/check', null, { params: { device_id: deviceId, shelf_id: shelfId } }).then(res => res.data),
  
  getThresholds: () => api.get('/api/alarm/thresholds').then(res => res.data),
  
  setThresholds: (params: {
    temp_diff?: number;
    vacuum_min?: number;
    vacuum_max?: number;
    cold_trap_max?: number;
    moisture_max?: number;
    reconstitution_max?: number;
  }) => api.put('/api/alarm/thresholds', null, { params }).then(res => res.data),
  
  getMqttStatus: () => api.get('/api/alarm/mqtt/status').then(res => res.data),
};

export const endpointApi = {
  getCurrentStatus: (deviceId: number): Promise<EndpointStatus> =>
    api.get(`/api/endpoint/current/${deviceId}`).then(res => res.data),

  getDetectionHistory: (deviceId: number, params?: {
    batch_id?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
  }): Promise<{ count: number; data: EndpointDetectionData[] }> =>
    api.get(`/api/endpoint/detection/${deviceId}`, { params }).then(res => res.data),

  triggerPRT: (deviceId: number) =>
    api.post(`/api/endpoint/prt/${deviceId}`).then(res => res.data),

  getPRTHistory: (deviceId: number, params?: {
    batch_id?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
  }): Promise<{ count: number; data: PressureRiseTest[] }> =>
    api.get(`/api/endpoint/prt/${deviceId}`, { params }).then(res => res.data),

  getStats: (deviceId: number, days?: number): Promise<EndpointStats> =>
    api.get(`/api/endpoint/stats/${deviceId}`, { params: { days } }).then(res => res.data),
};

export const defrostApi = {
  getOptimization: (deviceId: number, batchId?: string): Promise<DefrostOptimization> =>
    api.get('/api/defrost/optimization', { params: { device_id: deviceId, batch_id: batchId } }).then(res => res.data),

  getCurrentStatus: (deviceId: number): Promise<DefrostStatus> =>
    api.get(`/api/defrost/status/${deviceId}`).then(res => res.data),

  sendCommand: (deviceId: number, command: string, params?: object) =>
    api.post('/api/defrost/command', { device_id: deviceId, command, ...params }).then(res => res.data),

  getStatusHistory: (params: {
    device_id: number;
    start_time?: string;
    end_time?: string;
    limit?: number;
  }): Promise<{ count: number; history: DefrostStatus[] }> =>
    api.get('/api/defrost/history', { params }).then(res => res.data),

  getStats: (deviceId: number, hours: number = 24) =>
    api.get(`/api/defrost/stats/${deviceId}`, { params: { hours } }).then(res => res.data),
};

export const fleetApi = {
  createSchedule: (schedule: FleetScheduleCreate): Promise<FleetSchedule> =>
    api.post('/api/fleet/schedule', schedule).then(res => res.data),

  getSchedule: (scheduleId: number): Promise<FleetSchedule> =>
    api.get(`/api/fleet/schedule/${scheduleId}`).then(res => res.data),

  getSchedules: (params?: {
    limit?: number;
    status?: string;
  }): Promise<{ count: number; schedules: FleetSchedule[] }> =>
    api.get('/api/fleet/schedules', { params }).then(res => res.data),

  sendCommand: (deviceId: number, command: string, batchId?: string, parameters?: object) =>
    api.post(`/api/fleet/command/${deviceId}`, { command, batch_id: batchId, parameters }).then(res => res.data),

  getDeviceStatus: (deviceId: number): Promise<FleetDevice> =>
    api.get(`/api/fleet/status/${deviceId}`).then(res => res.data),

  getOverview: (): Promise<FleetOverview> =>
    api.get('/api/fleet/overview').then(res => res.data),

  getStats: (hours: number = 24): Promise<FleetStatsResponse> =>
    api.get('/api/fleet/stats', { params: { hours } }).then(res => res.data),

  getScheduledBatches: (): Promise<ScheduledBatch[]> =>
    api.get('/api/fleet/batches').then(res => res.data),

  getElectricityPrices: (hours: number = 24): Promise<ElectricityPrice[]> =>
    api.get('/api/fleet/electricity-prices', { params: { hours } }).then(res => res.data),
};

export const defectApi = {
  uploadImage: (deviceId: number, batchId: string, shelfId: number, imageFile: File) => {
    const formData = new FormData();
    formData.append('device_id', deviceId.toString());
    formData.append('batch_id', batchId);
    formData.append('shelf_id', shelfId.toString());
    formData.append('image', imageFile);
    return api.post('/api/defect/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(res => res.data);
  },

  detectDefects: (formData: FormData): Promise<DefectResult[]> =>
    api.post('/api/defect/detect', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(res => res.data),

  getDetection: (detectionId: string): Promise<ProductDefect> =>
    api.get(`/api/defect/detection/${detectionId}`).then(res => res.data),

  getBatches: (deviceId?: number): Promise<{ id: string; name: string }[]> =>
    api.get('/api/defect/batches', { params: { device_id: deviceId } }).then(res => res.data),

  getBatchDefects: (params: {
    device_id?: number;
    batch_id: string;
    defect_type?: string;
    reviewed?: boolean;
    limit?: number;
  }): Promise<{ count: number; defects: DefectResult[] }> =>
    api.get('/api/defect/batch', { params }).then(res => res.data),

  getBatchStats: (batchId: string, deviceId?: number): Promise<BatchStats> =>
    api.get(`/api/defect/batch/${batchId}/stats`, { params: { device_id: deviceId } }).then(res => res.data),

  getDistribution: (batchId: string, deviceId?: number): Promise<{
    normal: number;
    collapse: number;
    atrophy: number;
    cracking: number;
  }> =>
    api.get(`/api/defect/batch/${batchId}/distribution`, { params: { device_id: deviceId } }).then(res => res.data),

  getBatchDefectRecords: (batchId: string, deviceId?: number): Promise<BatchDefectRecord[]> =>
    api.get(`/api/defect/batch/${batchId}/records`, { params: { device_id: deviceId } }).then(res => res.data),

  reviewDefect: (defectId: string, reviewed: boolean, reviewedBy: string) =>
    api.post('/api/defect/review', {
      defect_id: defectId,
      reviewed,
      reviewed_by: reviewedBy,
    }).then(res => res.data),

  getStats: (params?: {
    device_id?: number;
    batch_id?: string;
    start_time?: string;
    end_time?: string;
  }) =>
    api.get('/api/defect/stats', { params }).then(res => res.data),

  getBatchRecords: (params?: {
    device_id?: number;
    batch_id?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
  }): Promise<{ count: number; records: BatchRecord[] }> =>
    api.get('/api/defect/batch-records', { params }).then(res => res.data),
};

export default api;
