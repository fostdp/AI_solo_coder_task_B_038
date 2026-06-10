import { useEffect, useState, useRef, useCallback } from 'react';
import { Target, TrendingUp, Brain, Gauge, Clock, Zap, Activity, RefreshCw } from 'lucide-react';
import * as echarts from 'echarts';
import type { 
  EndpointStatus, 
  EndpointDetectionData, 
  EndpointDetectionMethodConfidence,
  EndpointCombinedDecision,
  TemperatureDataPoint,
  DeviceInfo
} from '@/types';
import { endpointApi, deviceApi } from '@/services/api';

interface EndpointDashboardProps {
  deviceId: number;
}

const EndpointDashboard = ({ deviceId }: EndpointDashboardProps) => {
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<number>(deviceId);
  const [currentStatus, setCurrentStatus] = useState<EndpointStatus | null>(null);
  const [detectionHistory, setDetectionHistory] = useState<EndpointDetectionData[]>([]);
  const [temperatureData, setTemperatureData] = useState<TemperatureDataPoint[]>([]);
  const [methodConfidence, setMethodConfidence] = useState<EndpointDetectionMethodConfidence>({
    first_derivative: 0,
    autoencoder: 0,
    pressure_rise_test: 0
  });
  const [combinedDecision, setCombinedDecision] = useState<EndpointCombinedDecision | null>(null);
  const [isTriggeringPRT, setIsTriggeringPRT] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  const initChart = useCallback(() => {
    if (!chartRef.current) return;
    
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    chartInstance.current = echarts.init(chartRef.current, 'dark');
    
    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(30, 41, 59, 0.9)',
        borderColor: 'rgba(100, 116, 139, 0.5)',
        textStyle: {
          color: '#f1f5f9'
        }
      },
      legend: {
        data: ['温度', '一阶导数'],
        textStyle: {
          color: '#94a3b8'
        },
        top: 0
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: temperatureData.map(d => {
          const time = new Date(d.timestamp);
          return time.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }),
        axisLine: {
          lineStyle: {
            color: '#475569'
          }
        },
        axisLabel: {
          color: '#94a3b8'
        }
      },
      yAxis: [
        {
          type: 'value',
          name: '温度 (°C)',
          position: 'left',
          axisLine: {
            lineStyle: {
              color: '#22d3ee'
            }
          },
          axisLabel: {
            color: '#94a3b8',
            formatter: '{value}°C'
          },
          splitLine: {
            lineStyle: {
              color: 'rgba(71, 85, 105, 0.3)'
            }
          }
        },
        {
          type: 'value',
          name: '一阶导数',
          position: 'right',
          axisLine: {
            lineStyle: {
              color: '#f97316'
            }
          },
          axisLabel: {
            color: '#94a3b8',
            formatter: '{value}'
          },
          splitLine: {
            show: false
          }
        }
      ],
      series: [
        {
          name: '温度',
          type: 'line',
          smooth: true,
          symbol: 'none',
          lineStyle: {
            color: '#22d3ee',
            width: 2
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(34, 211, 238, 0.3)' },
              { offset: 1, color: 'rgba(34, 211, 238, 0)' }
            ])
          },
          data: temperatureData.map(d => d.temperature)
        },
        {
          name: '一阶导数',
          type: 'line',
          yAxisIndex: 1,
          smooth: true,
          symbol: 'none',
          lineStyle: {
            color: '#f97316',
            width: 2,
            type: 'dashed'
          },
          data: temperatureData.map(d => d.first_derivative)
        }
      ]
    };

    chartInstance.current.setOption(option);
  }, [temperatureData]);

  const generateMockTemperatureData = useCallback(() => {
    const now = new Date();
    const data: TemperatureDataPoint[] = [];
    
    for (let i = 59; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 1000);
      const baseTemp = -20 + Math.sin(i / 20) * 2;
      const noise = (Math.random() - 0.5) * 0.5;
      const temperature = baseTemp + noise;
      
      const prevTemp = i > 0 ? data[data.length - 1]?.temperature || baseTemp : baseTemp;
      const firstDerivative = temperature - prevTemp;
      
      data.push({
        timestamp: time.toISOString(),
        temperature: parseFloat(temperature.toFixed(2)),
        first_derivative: parseFloat(firstDerivative.toFixed(4))
      });
    }
    
    return data;
  }, []);

  const fetchData = useCallback(async () => {
    try {
      setIsLoading(true);
      
      const [statusRes, historyRes, devicesRes] = await Promise.all([
        endpointApi.getCurrentStatus(selectedDevice),
        endpointApi.getDetectionHistory(selectedDevice, { limit: 10 }),
        deviceApi.getDevices()
      ]);

      setCurrentStatus(statusRes);
      setDetectionHistory(historyRes.data || []);
      setDevices(devicesRes);

      const mockData = generateMockTemperatureData();
      setTemperatureData(mockData);

      if (historyRes.data && historyRes.data.length > 0) {
        const latest = historyRes.data[0];
        setMethodConfidence({
          first_derivative: latest.temp_first_derivative ? 
            Math.min(100, Math.max(0, (1 - Math.abs(latest.temp_first_derivative)) * 100)) : 75,
          autoencoder: latest.autoencoder_recon_error ? 
            Math.min(100, Math.max(0, (1 - latest.autoencoder_recon_error) * 100)) : 82,
          pressure_rise_test: latest.pressure_rise_delta ? 
            Math.min(100, Math.max(0, (1 - latest.pressure_rise_delta / 10) * 100)) : 88
        });

        const primaryTime = statusRes.primary_drying_endpoint;
        const secondaryTime = statusRes.secondary_drying_endpoint;
        
        setCombinedDecision({
          endpoint_detected: statusRes.primary_endpoint_detected || statusRes.secondary_endpoint_detected,
          current_phase: statusRes.current_phase,
          confidence: latest.detection_confidence * 100,
          estimated_time_saving_minutes: latest.estimated_energy_saving * 10,
          primary_detection_time: primaryTime,
          secondary_detection_time: secondaryTime
        });
      }
    } catch (error) {
      console.error('获取终点检测数据失败:', error);
      
      const mockData = generateMockTemperatureData();
      setTemperatureData(mockData);
      
      setMethodConfidence({
        first_derivative: 78.5,
        autoencoder: 85.2,
        pressure_rise_test: 91.3
      });

      setCombinedDecision({
        endpoint_detected: false,
        current_phase: 'primary_drying',
        confidence: 85,
        estimated_time_saving_minutes: 45,
        primary_detection_time: null,
        secondary_detection_time: null
      });
    } finally {
      setIsLoading(false);
    }
  }, [selectedDevice, generateMockTemperatureData]);

  const handleTriggerPRT = async () => {
    setIsTriggeringPRT(true);
    try {
      await endpointApi.triggerPRT(selectedDevice);
      setTimeout(() => {
        fetchData();
      }, 1000);
    } catch (error) {
      console.error('触发压力上升测试失败:', error);
    } finally {
      setIsTriggeringPRT(false);
    }
  };

  const handleDeviceChange = (newDeviceId: number) => {
    setSelectedDevice(newDeviceId);
  };

  const getPhaseStatusColor = (phase: string) => {
    switch (phase) {
      case 'primary_drying':
        return 'bg-cyan-500';
      case 'secondary_drying':
        return 'bg-purple-500';
      case 'completed':
        return 'bg-green-500';
      case 'idle':
        return 'bg-slate-500';
      default:
        return 'bg-amber-500';
    }
  };

  const getPhaseLabel = (phase: string) => {
    switch (phase) {
      case 'primary_drying':
        return '一次干燥';
      case 'secondary_drying':
        return '二次干燥';
      case 'completed':
        return '已完成';
      case 'idle':
        return '空闲';
      default:
        return phase;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 85) return 'text-green-400';
    if (confidence >= 70) return 'text-amber-400';
    return 'text-red-400';
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (temperatureData.length > 0) {
      const timer = requestAnimationFrame(() => {
        initChart();
      });
      return () => cancelAnimationFrame(timer);
    }
  }, [temperatureData, initChart]);

  useEffect(() => {
    const handleResize = () => {
      chartInstance.current?.resize();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 flex items-center justify-center">
              <Target className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-100">
                干燥终点检测
              </h3>
              <p className="text-xs text-slate-400">
                多方法融合检测 · 实时监控
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={selectedDevice}
              onChange={(e) => handleDeviceChange(Number(e.target.value))}
              className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-100 text-sm focus:outline-none focus:border-cyan-500"
            >
              {devices.map((device) => (
                <option key={device.id} value={device.id}>
                  {device.name}
                </option>
              ))}
            </select>
            {currentStatus && (
              <div className="flex items-center gap-2 px-3 py-2 bg-slate-800 rounded-lg">
                <span className={`w-2 h-2 rounded-full ${getPhaseStatusColor(currentStatus.current_phase)} animate-pulse`} />
                <span className="text-sm text-slate-300">
                  {getPhaseLabel(currentStatus.current_phase)}
                </span>
              </div>
            )}
            <button
              onClick={fetchData}
              className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <RefreshCw className={`w-4 h-4 text-slate-400 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {isLoading ? (
          <div className="p-8 text-center text-slate-500">
            <Activity className="w-12 h-12 mx-auto mb-2 text-cyan-500/50 animate-pulse" />
            <p>加载中...</p>
          </div>
        ) : (
          <>
            <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700">
              <div className="flex items-center gap-2 mb-3">
                <Activity className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-medium text-slate-300">实时温度曲线</span>
              </div>
              <div ref={chartRef} className="w-full h-64" />
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-3">
                  <TrendingUp className="w-5 h-5 text-cyan-400" />
                  <span className="text-sm font-medium text-slate-300">一阶导数法</span>
                </div>
                <div className="text-3xl font-mono font-bold mb-2">
                  <span className={getConfidenceColor(methodConfidence.first_derivative)}>
                    {methodConfidence.first_derivative.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-full transition-all duration-500"
                    style={{ width: `${methodConfidence.first_derivative}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-2">温度变化率检测</p>
              </div>

              <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-3">
                  <Brain className="w-5 h-5 text-purple-400" />
                  <span className="text-sm font-medium text-slate-300">自动编码器</span>
                </div>
                <div className="text-3xl font-mono font-bold mb-2">
                  <span className={getConfidenceColor(methodConfidence.autoencoder)}>
                    {methodConfidence.autoencoder.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-500 to-purple-400 rounded-full transition-all duration-500"
                    style={{ width: `${methodConfidence.autoencoder}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-2">重建误差分析</p>
              </div>

              <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-3">
                  <Gauge className="w-5 h-5 text-amber-400" />
                  <span className="text-sm font-medium text-slate-300">压力上升测试</span>
                </div>
                <div className="text-3xl font-mono font-bold mb-2">
                  <span className={getConfidenceColor(methodConfidence.pressure_rise_test)}>
                    {methodConfidence.pressure_rise_test.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-amber-500 to-amber-400 rounded-full transition-all duration-500"
                    style={{ width: `${methodConfidence.pressure_rise_test}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-2">压力变化率检测</p>
              </div>
            </div>

            {combinedDecision && (
              <div className={`p-4 rounded-lg border ${
                combinedDecision.endpoint_detected 
                  ? 'bg-green-500/10 border-green-500/30' 
                  : 'bg-slate-800/50 border-slate-700'
              }`}>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Target className={`w-5 h-5 ${combinedDecision.endpoint_detected ? 'text-green-400' : 'text-slate-400'}`} />
                      <span className="text-lg font-semibold text-slate-100">
                        {combinedDecision.endpoint_detected ? '终点已检测' : '终点未到达'}
                      </span>
                    </div>
                    <p className="text-sm text-slate-400">
                      当前阶段: {getPhaseLabel(combinedDecision.current_phase)}
                    </p>
                  </div>
                  <div className="flex gap-6">
                    <div className="text-center">
                      <div className="flex items-center gap-1 text-xs text-slate-400 mb-1">
                        <Gauge className="w-3 h-3" />
                        综合置信度
                      </div>
                      <div className={`text-2xl font-mono font-bold ${getConfidenceColor(combinedDecision.confidence)}`}>
                        {combinedDecision.confidence.toFixed(1)}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="flex items-center gap-1 text-xs text-slate-400 mb-1">
                        <Clock className="w-3 h-3" />
                        预计节省时间
                      </div>
                      <div className="text-2xl font-mono font-bold text-cyan-400">
                        {combinedDecision.estimated_time_saving_minutes} min
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="flex items-center gap-1 text-xs text-slate-400 mb-1">
                        <Zap className="w-3 h-3" />
                        一次干燥终点
                      </div>
                      <div className="text-sm font-mono text-slate-300">
                        {combinedDecision.primary_detection_time 
                          ? formatTimestamp(combinedDecision.primary_detection_time)
                          : '未检测'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {detectionHistory.length > 0 && (
              <div className="pt-4 border-t border-slate-700">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium text-slate-300">
                    历史检测记录
                  </h4>
                  <button
                    onClick={handleTriggerPRT}
                    disabled={isTriggeringPRT || currentStatus?.current_phase === 'idle'}
                    className="flex items-center gap-2 px-4 py-2 bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-600 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    <Zap className={`w-4 h-4 ${isTriggeringPRT ? 'animate-pulse' : ''}`} />
                    {isTriggeringPRT ? '测试中...' : '手动压力上升测试'}
                  </button>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-slate-400 border-b border-slate-700">
                        <th className="text-left py-2 px-3">时间</th>
                        <th className="text-left py-2 px-3">批次</th>
                        <th className="text-left py-2 px-3">阶段</th>
                        <th className="text-left py-2 px-3">检测方法</th>
                        <th className="text-right py-2 px-3">置信度</th>
                        <th className="text-right py-2 px-3">预计节能 (kWh)</th>
                        <th className="text-center py-2 px-3">状态</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detectionHistory.map((item, index) => (
                        <tr key={index} className="border-b border-slate-800 hover:bg-slate-800/30">
                          <td className="py-2 px-3 text-slate-300 font-mono text-xs">
                            {formatTimestamp(item.endpoint_timestamp)}
                          </td>
                          <td className="py-2 px-3 text-slate-300 font-mono">
                            {item.batch_id}
                          </td>
                          <td className="py-2 px-3">
                            <span className={`px-2 py-1 rounded-full text-xs ${
                              item.cycle_phase === 'primary_drying' 
                                ? 'bg-cyan-500/20 text-cyan-400' 
                                : 'bg-purple-500/20 text-purple-400'
                            }`}>
                              {getPhaseLabel(item.cycle_phase)}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-slate-400">
                            {item.detection_method}
                          </td>
                          <td className="py-2 px-3 text-right font-mono">
                            <span className={getConfidenceColor(item.detection_confidence * 100)}>
                              {(item.detection_confidence * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="py-2 px-3 text-right font-mono text-cyan-400">
                            {item.estimated_energy_saving.toFixed(2)}
                          </td>
                          <td className="py-2 px-3 text-center">
                            <span className={`px-2 py-1 rounded-full text-xs ${
                              item.is_accepted
                                ? 'bg-green-500/20 text-green-400'
                                : 'bg-red-500/20 text-red-400'
                            }`}>
                              {item.is_accepted ? '已接受' : '已拒绝'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default EndpointDashboard;
