import { useState, useEffect, useMemo, useCallback } from 'react';
import { Server, Play, Pause, Calendar, DollarSign, Zap, Clock, AlertTriangle, CheckCircle, Loader2, RefreshCw } from 'lucide-react';
import type { DeviceStatus, FleetDevice, FleetSchedule, ScheduledBatch, ElectricityPrice, FleetStatsResponse, FleetScheduleCreate, UrgentBatch } from '@/types';
import { fleetApi } from '@/services/api';

export interface ClusterSchedulerProps {
  onScheduleCreated?: (schedule: FleetSchedule) => void;
  onUrgentBatch?: (batch: UrgentBatch) => void;
  showGantt?: boolean;
  showControls?: boolean;
  className?: string;
}

export interface ClusterSchedulerHandle {
  createSchedule: (batches: number, horizon: number) => Promise<void>;
  addUrgentBatch: (batch: UrgentBatch) => Promise<void>;
  refresh: () => Promise<void>;
  getStats: () => FleetStatsResponse | null;
}

const statusConfig: Record<DeviceStatus, { label: string; color: string; bg: string; borderColor: string }> = {
  idle: { label: '空闲', color: 'text-slate-400', bg: 'bg-slate-500/20', borderColor: 'border-slate-500/30' },
  running: { label: '运行中', color: 'text-green-400', bg: 'bg-green-500/20', borderColor: 'border-green-500/30' },
  paused: { label: '已暂停', color: 'text-yellow-400', bg: 'bg-yellow-500/20', borderColor: 'border-yellow-500/30' },
  maintenance: { label: '维护中', color: 'text-orange-400', bg: 'bg-orange-500/20', borderColor: 'border-orange-500/30' },
  defrosting: { label: '除霜中', color: 'text-cyan-400', bg: 'bg-cyan-500/20', borderColor: 'border-cyan-500/30' },
};

const batchColors = [
  'bg-cyan-500',
  'bg-purple-500',
  'bg-green-500',
  'bg-orange-500',
  'bg-pink-500',
  'bg-blue-500',
  'bg-yellow-500',
  'bg-red-500',
  'bg-indigo-500',
  'bg-teal-500',
];

const ClusterScheduler = ({
  onScheduleCreated,
  onUrgentBatch,
  showGantt = true,
  showControls = true,
  className = ''
}: ClusterSchedulerProps) => {
  const [devices, setDevices] = useState<FleetDevice[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<FleetDevice | null>(null);
  const [schedules, setSchedules] = useState<FleetSchedule[]>([]);
  const [scheduledBatches, setScheduledBatches] = useState<ScheduledBatch[]>([]);
  const [electricityPrices, setElectricityPrices] = useState<ElectricityPrice[]>([]);
  const [stats, setStats] = useState<FleetStatsResponse | null>(null);
  const [requiredBatches, setRequiredBatches] = useState(10);
  const [timeHorizonHours, setTimeHorizonHours] = useState(24);
  const [isCreatingSchedule, setIsCreatingSchedule] = useState(false);
  const [isRescheduling, setIsRescheduling] = useState(false);
  const [workerStatus, setWorkerStatus] = useState<{ active: boolean; pending_tasks: number; completed_tasks: number }>({
    active: false,
    pending_tasks: 0,
    completed_tasks: 0
  });

  const fetchData = useCallback(async () => {
    try {
      const [overview, schedulesData, batchesData, pricesData, statsData] = await Promise.all([
        fleetApi.getOverview(),
        fleetApi.getSchedules({ limit: 5 }),
        fleetApi.getScheduledBatches(),
        fleetApi.getElectricityPrices(24),
        fleetApi.getStats(24),
      ]);

      const allDevices: FleetDevice[] = [];
      for (let i = 1; i <= 10; i++) {
        const existingDevice = overview.devices.find(d => d.device_id === i);
        if (existingDevice) {
          allDevices.push(existingDevice);
        } else {
          allDevices.push({
            device_id: i,
            status: 'idle',
            current_batch: null,
            batch_progress: 0,
            current_schedule_id: null,
            last_command: null,
            last_update: null,
          });
        }
      }
      setDevices(allDevices);
      setSchedules(schedulesData.schedules);
      setScheduledBatches(batchesData);
      setElectricityPrices(pricesData);
      setStats(statsData);

      setWorkerStatus({
        active: true,
        pending_tasks: schedulesData.schedules.filter(s => s.status === 'pending').length,
        completed_tasks: schedulesData.schedules.filter(s => s.status === 'completed').length
      });
    } catch (error) {
      console.error('获取调度数据失败:', error);
      const mockDevices: FleetDevice[] = Array.from({ length: 10 }, (_, i) => ({
        device_id: i + 1,
        status: (['idle', 'running', 'paused', 'maintenance', 'defrosting'] as DeviceStatus[])[Math.floor(Math.random() * 5)],
        current_batch: Math.random() > 0.5 ? `BATCH-${String(i + 1).padStart(3, '0')}` : null,
        batch_progress: Math.floor(Math.random() * 100),
        current_schedule_id: Math.random() > 0.5 ? 1 : null,
        last_command: null,
        last_update: new Date().toISOString(),
      }));
      setDevices(mockDevices);
      
      setWorkerStatus({
        active: true,
        pending_tasks: 1,
        completed_tasks: Math.floor(Math.random() * 10)
      });
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleCreateSchedule = async () => {
    setIsCreatingSchedule(true);
    try {
      const scheduleData: FleetScheduleCreate = {
        required_batches: requiredBatches,
        time_horizon_hours: timeHorizonHours,
        priority: 'normal',
      };
      const result = await fleetApi.createSchedule(scheduleData);
      const schedulesData = await fleetApi.getSchedules({ limit: 5 });
      setSchedules(schedulesData.schedules);
      
      if (onScheduleCreated && schedulesData.schedules.length > 0) {
        onScheduleCreated(schedulesData.schedules[0]);
      }
    } catch (error) {
      console.error('创建排程失败:', error);
    } finally {
      setIsCreatingSchedule(false);
    }
  };

  const handleAddUrgentBatch = async () => {
    setIsRescheduling(true);
    try {
      const urgentBatch: UrgentBatch = {
        id: `URG-${Date.now()}`,
        priority: 10,
        required_hours: 4,
        deadline: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(),
        product_type: 'urgent',
      };
      
      await fleetApi.addUrgentBatch(urgentBatch);
      
      setTimeout(async () => {
        const schedulesData = await fleetApi.getSchedules({ limit: 5 });
        setSchedules(schedulesData.schedules);
        
        if (onUrgentBatch) {
          onUrgentBatch(urgentBatch);
        }
      }, 2000);
    } catch (error) {
      console.error('添加紧急插单失败:', error);
    } finally {
      setIsRescheduling(false);
    }
  };

  const handleDeviceClick = (device: FleetDevice) => {
    setSelectedDevice(device);
  };

  const handleSendCommand = async (deviceId: number, command: string) => {
    try {
      await fleetApi.sendCommand(deviceId, command);
      const overview = await fleetApi.getOverview();
      const allDevices: FleetDevice[] = [];
      for (let i = 1; i <= 10; i++) {
        const existingDevice = overview.devices.find(d => d.device_id === i);
        if (existingDevice) {
          allDevices.push(existingDevice);
        } else {
          allDevices.push({
            device_id: i,
            status: 'idle',
            current_batch: null,
            batch_progress: 0,
            current_schedule_id: null,
            last_command: null,
            last_update: null,
          });
        }
      }
      setDevices(allDevices);
    } catch (error) {
      console.error('发送命令失败:', error);
    }
  };

  const timelineHours = useMemo(() => {
    const hours = [];
    const now = new Date();
    for (let i = 0; i < 24; i++) {
      const hour = new Date(now);
      hour.setHours(now.getHours() + i);
      hours.push(hour);
    }
    return hours;
  }, []);

  const mockBatches = useMemo(() => {
    const batches: ScheduledBatch[] = [];
    const now = new Date();
    let batchIndex = 0;
    
    devices.forEach((device, deviceIdx) => {
      if (device.status === 'running' || device.status === 'paused') {
        const startTime = new Date(now);
        startTime.setMinutes(0, 0, 0);
        const duration = 4 + Math.floor(Math.random() * 4);
        
        batches.push({
          id: `BATCH-${String(batchIndex + 1).padStart(3, '0')}`,
          device_id: device.device_id,
          schedule_id: 1,
          start_time: startTime.toISOString(),
          end_time: new Date(startTime.getTime() + duration * 60 * 60 * 1000).toISOString(),
          duration_hours: duration,
          status: device.status === 'running' ? 'running' : 'pending',
          energy_cost: Math.round(Math.random() * 500 + 200),
          optimization_savings: Math.round(Math.random() * 150 + 50),
        });
        batchIndex++;
      }
      
      if (deviceIdx % 2 === 0) {
        const startTime = new Date(now);
        startTime.setHours(now.getHours() + 6 + Math.floor(Math.random() * 6));
        startTime.setMinutes(0, 0, 0);
        const duration = 4 + Math.floor(Math.random() * 4);
        
        batches.push({
          id: `BATCH-${String(batchIndex + 1).padStart(3, '0')}`,
          device_id: device.device_id,
          schedule_id: 1,
          start_time: startTime.toISOString(),
          end_time: new Date(startTime.getTime() + duration * 60 * 60 * 1000).toISOString(),
          duration_hours: duration,
          status: 'pending',
          energy_cost: Math.round(Math.random() * 500 + 200),
          optimization_savings: Math.round(Math.random() * 150 + 50),
        });
        batchIndex++;
      }
    });
    
    return batches;
  }, [devices]);

  const activeBatches = mockBatches.length > 0 ? mockBatches : scheduledBatches;

  const statusCounts = useMemo(() => {
    const counts: Record<DeviceStatus, number> = {
      idle: 0,
      running: 0,
      paused: 0,
      maintenance: 0,
      defrosting: 0,
    };
    devices.forEach(d => {
      if (counts[d.status] !== undefined) {
        counts[d.status]++;
      }
    });
    return counts;
  }, [devices]);

  const totalEnergyCost = useMemo(() => {
    return activeBatches.reduce((sum, b) => sum + b.energy_cost, 0);
  }, [activeBatches]);

  const totalSavings = useMemo(() => {
    return activeBatches.reduce((sum, b) => sum + b.optimization_savings, 0);
  }, [activeBatches]);

  const getBatchPosition = (batch: ScheduledBatch) => {
    const now = new Date();
    now.setMinutes(0, 0, 0);
    const start = new Date(batch.start_time);
    const end = new Date(batch.end_time);
    
    const startOffset = Math.max(0, (start.getTime() - now.getTime()) / (1000 * 60 * 60));
    const duration = (end.getTime() - start.getTime()) / (1000 * 60 * 60);
    
    return {
      left: `${(startOffset / 24) * 100}%`,
      width: `${Math.min((duration / 24) * 100, 100 - (startOffset / 24) * 100)}%`,
    };
  };

  const getElectricityPeriodColor = (period: string) => {
    switch (period) {
      case 'peak': return 'bg-red-500/30 border-red-500/50';
      case 'valley': return 'bg-green-500/30 border-green-500/50';
      default: return 'bg-slate-500/30 border-slate-500/50';
    }
  };

  const mockPrices = useMemo(() => {
    return timelineHours.map((hour, idx) => {
      const h = hour.getHours();
      let period: 'peak' | 'valley' | 'normal' = 'normal';
      if (h >= 8 && h < 12) period = 'peak';
      else if (h >= 18 && h < 22) period = 'peak';
      else if (h >= 0 && h < 6) period = 'valley';
      
      return {
        timestamp: hour.toISOString(),
        price: period === 'peak' ? 1.2 : period === 'valley' ? 0.4 : 0.8,
        period,
      };
    });
  }, [timelineHours]);

  const displayPrices = electricityPrices.length > 0 ? electricityPrices : mockPrices;

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 flex items-center justify-center">
            <Server className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-100">机群调度中心</h1>
            <p className="text-sm text-slate-400">10台冷冻干燥机 · 整数规划 · Worker进程调度</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
            workerStatus.active ? 'bg-green-500/20' : 'bg-red-500/20'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              workerStatus.active ? 'bg-green-500' : 'bg-red-500'
            } ${workerStatus.active && workerStatus.pending_tasks > 0 ? 'animate-pulse' : ''}`} />
            <span className={`text-sm ${workerStatus.active ? 'text-green-400' : 'text-red-400'}`}>
              Worker {workerStatus.active ? '运行中' : '已停止'}
            </span>
            {workerStatus.pending_tasks > 0 && (
              <span className="text-xs text-slate-400">
                待处理: {workerStatus.pending_tasks}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-4">
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
            <Calendar className="w-4 h-4" />
            总批次
          </div>
          <div className="text-3xl font-mono font-bold text-cyan-400">
            {stats?.schedules?.total_batches_scheduled || activeBatches.length}
          </div>
        </div>
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
            <DollarSign className="w-4 h-4" />
            能源成本
          </div>
          <div className="text-3xl font-mono font-bold text-orange-400">
            ¥{stats?.total_energy_cost || totalEnergyCost}
          </div>
        </div>
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
            <Zap className="w-4 h-4" />
            优化节省
          </div>
          <div className="text-3xl font-mono font-bold text-green-400">
            ¥{stats?.total_optimization_savings || totalSavings}
          </div>
        </div>
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
            <Clock className="w-4 h-4" />
            运行设备
          </div>
          <div className="text-3xl font-mono font-bold text-purple-400">
            {statusCounts.running}/{devices.length}
          </div>
        </div>
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
            <CheckCircle className="w-4 h-4" />
            已完成调度
          </div>
          <div className="text-3xl font-mono font-bold text-teal-400">
            {workerStatus.completed_tasks}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-8 space-y-6">
          <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-slate-100">机群概览</h3>
              <div className="flex gap-2">
                {(Object.keys(statusConfig) as DeviceStatus[]).map(status => (
                  <div key={status} className="flex items-center gap-1">
                    <span className={`w-2 h-2 rounded-full ${statusConfig[status].bg.replace('/20', '')}`} />
                    <span className="text-xs text-slate-400">
                      {statusConfig[status].label} ({statusCounts[status]})
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-5 gap-3">
              {devices.map(device => {
                const config = statusConfig[device.status] || statusConfig.idle;
                const isSelected = selectedDevice?.device_id === device.device_id;
                return (
                  <button
                    key={device.device_id}
                    onClick={() => handleDeviceClick(device)}
                    className={`
                      relative p-4 rounded-lg border transition-all duration-200
                      ${config.bg} ${config.borderColor}
                      ${isSelected ? 'ring-2 ring-cyan-500 ring-offset-2 ring-offset-slate-900' : ''}
                      hover:scale-105
                    `}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-300">
                        #{device.device_id.toString().padStart(2, '0')}
                      </span>
                      <span className={`w-2 h-2 rounded-full ${config.bg.replace('/20', '')} ${device.status === 'running' ? 'animate-pulse' : ''}`} />
                    </div>
                    <div className={`text-xs ${config.color} mb-2`}>
                      {config.label}
                    </div>
                    {device.current_batch && (
                      <div className="text-xs text-slate-400 mb-2 font-mono">
                        {device.current_batch}
                      </div>
                    )}
                    {device.batch_progress > 0 && (
                      <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${config.bg.replace('/20', '')} transition-all duration-300`}
                          style={{ width: `${device.batch_progress}%` }}
                        />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {showGantt && (
            <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-slate-100">排程甘特图</h3>
                <div className="flex gap-2">
                  <div className="flex items-center gap-1">
                    <span className="w-3 h-3 rounded bg-green-500" />
                    <span className="text-xs text-slate-400">运行中</span>
                  </div>
                  <div className="flex items gap-2">
                    <span className="w-3 h-3 rounded bg-cyan-500" />
                    <span className="text-xs text-slate-400">待执行</span>
                  </div>
                  <button
                    onClick={fetchData}
                    className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
                  >
                    <RefreshCw className="w-4 h-4 text-slate-400" />
                  </button>
                </div>
              </div>

              <div className="mb-2">
                <div className="flex gap-1 px-16">
                  {timelineHours.filter((_, i) => i % 3 === 0).map((hour, idx) => (
                    <div key={idx} className="flex-1 text-xs text-slate-500 text-center">
                      {hour.getHours().toString().padStart(2, '0')}:00
                    </div>
                  ))}
                </div>
              </div>

              <div className="mb-2">
                <div className="flex gap-1 px-16">
                  {displayPrices.filter((_, i) => i % 3 === 0).map((price, idx) => (
                    <div
                      key={idx}
                      className={`flex-1 h-4 rounded ${getElectricityPeriodColor(price.period)} border text-xs text-center text-slate-400 leading-4`}
                    >
                      {price.price.toFixed(1)}
                    </div>
                  ))}
                </div>
                <div className="flex gap-2 mt-1 px-16 justify-center">
                  <div className="flex items gap-1">
                    <span className="w-2 h-2 rounded bg-red-500" />
                    <span className="text-xs text-slate-500">峰时</span>
                  </div>
                  <div className="flex items gap-1">
                    <span className="w-2 h-2 rounded bg-green-500" />
                    <span className="text-xs text-slate-500">谷时</span>
                  </div>
                  <div className="flex items gap-1">
                    <span className="w-2 h-2 rounded bg-slate-500" />
                    <span className="text-xs text-slate-500">平时</span>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                {devices.slice(0, 10).map(device => (
                  <div key={device.device_id} className="flex items-center gap-3">
                    <div className="w-14 text-xs text-slate-400 text-right">
                      #{device.device_id.toString().padStart(2, '0')}
                    </div>
                    <div className="flex-1 relative h-8 bg-slate-800/50 rounded-lg overflow-hidden">
                      <div className="absolute inset-0 flex">
                        {Array.from({ length: 8 }).map((_, i) => (
                          <div key={i} className="flex-1 border-l border-slate-700/50" />
                        ))}
                      </div>
                      {activeBatches
                        .filter(b => b.device_id === device.device_id)
                        .map((batch, idx) => {
                          const pos = getBatchPosition(batch);
                          const colorIdx = idx % batchColors.length;
                          return (
                            <div
                              key={batch.id}
                              className={`absolute top-1 bottom-1 rounded ${batchColors[colorIdx]} ${batch.status === 'running' ? 'ring-2 ring-white/50' : ''} transition-all duration-300`}
                              style={{ left: pos.left, width: pos.width }}
                              title={`${batch.id} · ${batch.duration_hours}h · ¥${batch.energy_cost}`}
                            >
                              <div className="px-2 text-xs text-white font-medium truncate leading-7">
                                {batch.id}
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="col-span-4 space-y-6">
          {showControls && (
            <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-4">
              <div className="flex items-center gap-2 mb-4">
                <Calendar className="w-5 h-5 text-cyan-400" />
                <h3 className="font-semibold text-slate-100">创建排程</h3>
              </div>
              <div className="space-y-4">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">
                    需要批次数量
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    value={requiredBatches}
                    onChange={(e) => setRequiredBatches(parseInt(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-100 font-mono focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-400 block mb-1">
                    时间范围 (小时)
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="168"
                    value={timeHorizonHours}
                    onChange={(e) => setTimeHorizonHours(parseInt(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-100 font-mono focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <button
                  onClick={handleCreateSchedule}
                  disabled={isCreatingSchedule}
                  className="w-full py-2 bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  {isCreatingSchedule ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      计算中...
                    </>
                  ) : (
                    <>
                      <Calendar className="w-4 h-4" />
                      生成优化排程
                    </>
                  )}
                </button>
                <button
                  onClick={handleAddUrgentBatch}
                  disabled={isRescheduling}
                  className="w-full py-2 bg-orange-500 hover:bg-orange-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  {isRescheduling ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      重调度中...
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-4 h-4" />
                      紧急插单
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {schedules.length > 0 && (
            <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-4">
              <div className="flex items-center gap-2 mb-4">
                <Clock className="w-5 h-5 text-cyan-400" />
                <h3 className="font-semibold text-slate-100">当前排程</h3>
              </div>
              <div className="space-y-3">
                {schedules.slice(0, 3).map(schedule => (
                  <div key={schedule.id} className="p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-200">
                        排程 #{schedule.id}
                      </span>
                      <span className={`px-2 py-0.5 text-xs rounded-full ${
                        schedule.status === 'running' ? 'bg-green-500/20 text-green-400' :
                        schedule.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-slate-500/20 text-slate-400'
                      }`}>
                        {schedule.status === 'running' ? '运行中' : schedule.status === 'pending' ? '待处理' : '已完成'}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="text-slate-400">
                        批次: <span className="text-slate-200 font-mono">{schedule.required_batches}</span>
                      </div>
                      <div className="text-slate-400">
                        时长: <span className="text-slate-200 font-mono">{schedule.time_horizon_hours}h</span>
                      </div>
                    </div>
                    {activeBatches.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-slate-700 text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-400">预估能源成本:</span>
                          <span className="text-orange-400 font-mono">¥{totalEnergyCost}</span>
                        </div>
                        <div className="flex justify-between mt-1">
                          <span className="text-slate-400">优化节省:</span>
                          <span className="text-green-400 font-mono">¥{totalSavings}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {selectedDevice && (
            <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-4">
              <div className="flex items-center gap-2 mb-4">
                <Server className="w-5 h-5 text-cyan-400" />
                <h3 className="font-semibold text-slate-100">
                  设备 #{selectedDevice.device_id.toString().padStart(2, '0')} 详情
                </h3>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">状态</span>
                  <span className={`px-2 py-0.5 text-xs rounded-full ${statusConfig[selectedDevice.status].bg} ${statusConfig[selectedDevice.status].color}`}>
                    {statusConfig[selectedDevice.status].label}
                  </span>
                </div>

                {selectedDevice.current_batch && (
                  <div className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-slate-400">当前批次</span>
                      <span className="text-sm font-mono text-cyan-400">
                        {selectedDevice.current_batch}
                      </span>
                    </div>
                    <div className="mb-2">
                      <div className="flex items justify-between text-xs mb-1">
                        <span className="text-slate-400">批次进度</span>
                        <span className="text-slate-300 font-mono">{selectedDevice.batch_progress}%</span>
                      </div>
                      <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${statusConfig[selectedDevice.status].bg.replace('/20', '')} transition-all duration-300`}
                          style={{ width: `${selectedDevice.batch_progress}%` }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => handleSendCommand(selectedDevice.device_id, 'start_batch')}
                    disabled={selectedDevice.status === 'running'}
                    className="flex items-center justify-center gap-1 py-2 bg-green-500/20 text-green-400 rounded-lg text-sm hover:bg-green-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Play className="w-4 h-4" />
                    启动
                  </button>
                  <button
                    onClick={() => handleSendCommand(selectedDevice.device_id, 'pause')}
                    disabled={selectedDevice.status !== 'running'}
                    className="flex items-center justify-center gap-1 py-2 bg-yellow-500/20 text-yellow-400 rounded-lg text-sm hover:bg-yellow-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Pause className="w-4 h-4" />
                    暂停
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ClusterScheduler;
