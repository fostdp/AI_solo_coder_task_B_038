import { useState, useEffect, useCallback } from 'react';
import { Activity, ThermometerSun, Settings, Bell, RefreshCw, Target, Snowflake, Server, AlertTriangle } from 'lucide-react';
import Heatmap from './components/Heatmap';
import VacuumChart from './components/VacuumChart';
import DeviceOverview from './components/DeviceOverview';
import ControlPanel from './components/ControlPanel';
import AlarmPanel from './components/AlarmPanel';
import QualityPrediction from './components/QualityPrediction';
import EndpointDashboard from './components/EndpointDashboard';
import DefrostStatus from './components/DefrostStatus';
import FleetControl from './components/FleetControl';
import DefectDetection from './components/DefectDetection';
import { useAppStore } from './store';
import { deviceApi, dataApi, alarmApi } from './services/api';
import type { RealtimeData, VacuumDataPoint, AlarmData } from './types';

function App() {
  const {
    selectedDevice,
    devices,
    realtimeData,
    currentAlarms,
    setSelectedDevice,
    setDevices,
    setRealtimeData,
    setCurrentAlarms,
    lastUpdate,
    isLoading,
    setLoading,
    setError,
  } = useAppStore();

  const [selectedShelf, setSelectedShelf] = useState<number>(1);
  const [vacuumHistory, setVacuumHistory] = useState<VacuumDataPoint[]>([]);
  const [activeTab, setActiveTab] = useState<'heatmap' | 'control' | 'prediction' | 'endpoint' | 'defrost' | 'fleet' | 'defect'>('heatmap');
  const [alarmHistory, setAlarmHistory] = useState<AlarmData[]>([]);

  const currentRealtimeData = realtimeData[selectedDevice] || [];
  const currentShelfData =
    currentRealtimeData.find((d) => d.shelf_id === selectedShelf) || null;
  const shelfIds = [1, 2, 3, 4, 5];

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);

      const [realtime, alarms, alarmHist] = await Promise.all([
        dataApi.getRealtimeData(selectedDevice),
        alarmApi.getCurrentAlarms(),
        alarmApi.getHistory({ limit: 20 }),
      ]);

      setRealtimeData(selectedDevice, realtime);
      setCurrentAlarms(alarms.alarms);
      setAlarmHistory(alarmHist.alarms);

      realtime.forEach((d) => {
        d.vacuum_levels.forEach((v) => {
          setVacuumHistory((prev) => [
            ...prev.slice(-200),
            {
              timestamp: d.timestamp,
              shelf_id: d.shelf_id,
              value: v,
            },
          ]);
        });
      });

      setError(null);
    } catch (err) {
      setError('数据获取数据失败');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [selectedDevice, setRealtimeData, setCurrentAlarms, setLoading, setError]);

  useEffect(() => {
    const fetchDevices = async () => {
      try {
        const data = await deviceApi.getDevices();
        setDevices(data);
      } catch (error) {
        console.error('获取设备列表失败:', error);
      }
    };
    fetchDevices();
  }, [setDevices]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleDeviceChange = (deviceId: number) => {
    setSelectedDevice(deviceId);
    setSelectedShelf(1);
    setVacuumHistory([]);
  };

  const handleShelfClick = (shelfId: number) => {
    setSelectedShelf(shelfId);
  };

  const handleAlarmAcknowledge = () => {
    fetchData();
  };

  const handleQualityPredict = () => {
    fetchData();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-white">
                    生物制药冻干机监控系统
                  </h1>
                  <p className="text-xs text-slate-400">
                    温度均匀性控制 · 质量预测
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <label className="text-sm text-slate-400">选择设备:</label>
                <select
                  value={selectedDevice}
                  onChange={(e) => handleDeviceChange(parseInt(e.target.value))}
                  className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:border-cyan-500"
                >
                  {devices.map((device) => (
                    <option key={device.id} value={device.id}>
                      {device.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-2 text-sm">
                <RefreshCw
                  className={`w-4 h-4 text-green-400 ${
                    isLoading ? 'animate-spin' : ''
                  }`}
                />
                <span className="text-slate-400">
                  {lastUpdate
                    ? `更新于 ${new Date(lastUpdate).toLocaleTimeString('zh-CN')}`
                    : '连接中...'}
                </span>
              </div>

              <button className="relative p-2 text-slate-400 hover:text-white transition-colors">
                <Bell className="w-5 h-5" />
                {currentAlarms.filter((a) => !a.acknowledged).length > 0 && (
                  <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full" />
                )}
              </button>

              <button className="p-2 text-slate-400 hover:text-white transition-colors">
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-6">
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-8 space-y-6">
            <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-6">
              <DeviceOverview
                deviceId={selectedDevice}
                shelfData={currentRealtimeData}
                onShelfClick={handleShelfClick}
                selectedShelf={selectedShelf}
              />
            </div>

            <div className="bg-slate-900/50 rounded-xl border border-slate-700 overflow-hidden">
              <div className="flex border-b border-slate-700 flex-wrap">
                {([
                  { id: 'heatmap', label: '温度热力图', icon: ThermometerSun },
                  { id: 'control', label: '功率控制', icon: Settings },
                  { id: 'prediction', label: '质量预测', icon: Activity },
                  { id: 'endpoint', label: '终点判定', icon: Target },
                  { id: 'defrost', label: '除霜优化', icon: Snowflake },
                  { id: 'fleet', label: '群控调度', icon: Server },
                  { id: 'defect', label: '缺陷检测', icon: AlertTriangle },
                ] as const).map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-colors ${
                      activeTab === tab.id
                        ? 'text-cyan-400 border-b-2 border-cyan-400 bg-slate-800/50'
                        : 'text-slate-400 hover:text-slate-300 hover:bg-slate-800/30'
                    }`}
                  >
                    <tab.icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="p-6">
                {activeTab === 'heatmap' && (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-slate-100">
                        搁板 {selectedShelf} 温度分布
                      </h3>
                      <div className="flex gap-2">
                          {currentRealtimeData.map((d) => (
                            <button
                              key={d.shelf_id}
                              onClick={() => setSelectedShelf(d.shelf_id)}
                              className={`px-3 py-1 text-xs rounded transition-colors ${
                                selectedShelf === d.shelf_id
                                  ? 'bg-cyan-500 text-white'
                                  : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                              }`}
                            >
                              搁板 {d.shelf_id}
                            </button>
                          ))}
                        </div>
                    </div>
                    <Heatmap
                      data={currentShelfData}
                      tempDiffThreshold={1.0}
                      width={800}
                      height={250}
                    />
                  </div>
                )}

                {activeTab === 'control' && (
                  <ControlPanel
                    deviceId={selectedDevice}
                    shelfId={selectedShelf}
                    data={currentShelfData}
                  />
                )}

                {activeTab === 'prediction' && (
                  <QualityPrediction
                    deviceId={selectedDevice}
                    onPredict={handleQualityPredict}
                  />
                )}

                {activeTab === 'endpoint' && (
                  <EndpointDashboard deviceId={selectedDevice} />
                )}

                {activeTab === 'defrost' && (
                  <DefrostStatus deviceId={selectedDevice} />
                )}

                {activeTab === 'fleet' && (
                  <FleetControl />
                )}

                {activeTab === 'defect' && (
                  <DefectDetection deviceId={selectedDevice} />
                )}
              </div>
            </div>

            <VacuumChart
              data={vacuumHistory}
              shelfIds={shelfIds}
              height={300}
            />
          </div>

          <div className="col-span-4 space-y-6">
            <AlarmPanel
              alarms={[...currentAlarms, ...alarmHistory.slice(0, 10)]}
              onAcknowledge={handleAlarmAcknowledge}
            />

            <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-4">
              <h3 className="font-semibold text-slate-100 mb-4">系统状态</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">Profinet连接</span>
                  <span className="flex items-center gap-2 text-green-400 text-sm">
                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    正常
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">MQTT连接</span>
                  <span className="flex items-center gap-2 text-green-400 text-sm">
                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    已连接
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">控制模式</span>
                  <span className="text-cyan-400 text-sm">自动控制</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">采集频率</span>
                  <span className="text-slate-300 text-sm font-mono">10s</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-4">
              <h3 className="font-semibold text-slate-100 mb-4">快速操作</h3>
              <div className="grid grid-cols-2 gap-3">
                <button className="p-3 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors">
                  导出报表
                </button>
                <button className="p-3 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors">
                  历史查询
                </button>
                <button className="p-3 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors">
                  参数设置
                </button>
                <button className="p-3 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors">
                  系统日志
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
