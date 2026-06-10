import { useState, useEffect, useRef } from 'react';
import { Snowflake, Thermometer, Clock, Zap, TrendingDown, Settings, Play, Square, ChevronUp, ChevronDown } from 'lucide-react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import { defrostApi } from '@/services/api';
import type { 
  DefrostStatusData, 
  DefrostOptimizationData, 
  DefrostStatusHistoryItem, 
  DefrostEnergyStats,
  ColdTrapTempDataPoint 
} from '@/types';

interface DefrostStatusProps {
  deviceId: number;
}

const DefrostStatus = ({ deviceId }: DefrostStatusProps) => {
  const [currentStatus, setCurrentStatus] = useState<DefrostStatusData | null>(null);
  const [optimizationData, setOptimizationData] = useState<DefrostOptimizationData | null>(null);
  const [statusHistory, setStatusHistory] = useState<DefrostStatusHistoryItem[]>([]);
  const [energyStats, setEnergyStats] = useState<DefrostEnergyStats | null>(null);
  const [tempHistory, setTempHistory] = useState<ColdTrapTempDataPoint[]>([]);
  const [heatingPower, setHeatingPower] = useState(30);
  const [maxDuration, setMaxDuration] = useState(60);
  const [isLoading, setIsLoading] = useState(false);
  const [isCommandExecuting, setIsCommandExecuting] = useState(false);

  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const [statusRes, optimizationRes, historyRes, statsRes, tempRes] = await Promise.all([
          defrostApi.getCurrentStatus(deviceId),
          defrostApi.getOptimizationHistory(deviceId, 1),
          defrostApi.getStatusHistory(deviceId, 20),
          defrostApi.getEnergyStats(deviceId, 30),
          defrostApi.getColdTrapTempHistory(deviceId, 24),
        ]);

        setCurrentStatus(statusRes);
        setOptimizationData(optimizationRes.history[0] || null);
        setStatusHistory(historyRes.history);
        setEnergyStats(statsRes);
        setTempHistory(tempRes.history);
      } catch (error) {
        console.error('获取除霜数据失败:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();

    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [deviceId]);

  useEffect(() => {
    if (!chartRef.current || tempHistory.length === 0) return;

    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current, 'dark');
    }

    const sortedData = [...tempHistory].sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    const times = sortedData.map(item => 
      new Date(item.timestamp).toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
      })
    );

    const temps = sortedData.map(item => item.temp);

    const option: EChartsOption = {
      backgroundColor: 'transparent',
      title: {
        text: '冷阱温度趋势 (24小时)',
        textStyle: {
          color: '#CBD5E1',
          fontSize: 14,
          fontWeight: 'normal',
          fontFamily: 'Inter',
        },
        left: 10,
        top: 10,
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        borderColor: '#334155',
        borderWidth: 1,
        textStyle: {
          color: '#F1F5F9',
          fontFamily: 'Inter',
        },
        axisPointer: {
          type: 'cross',
          label: {
            backgroundColor: '#06B6D4',
            fontFamily: 'JetBrains Mono',
          },
        },
        formatter: (params: any) => {
          if (!Array.isArray(params)) return '';
          let result = `${params[0].axisValue}<br/>`;
          params.forEach((param: any) => {
            const value = param.value !== null ? `${Number(param.value).toFixed(1)}℃` : '-';
            result += `${param.marker}冷阱温度: ${value}<br/>`;
          });
          return result;
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '50px',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: times,
        axisLine: {
          lineStyle: {
            color: '#334155',
          },
        },
        axisLabel: {
          color: '#64748B',
          fontSize: 10,
          fontFamily: 'JetBrains Mono',
        },
        splitLine: {
          show: true,
          lineStyle: {
            color: '#1E293B',
            type: 'dashed',
          },
        },
      },
      yAxis: {
        type: 'value',
        name: '℃',
        nameTextStyle: {
          color: '#64748B',
          fontSize: 11,
        },
        axisLine: {
          show: false,
        },
        axisLabel: {
          color: '#64748B',
          fontSize: 10,
          fontFamily: 'JetBrains Mono',
          formatter: (value: number) => value.toFixed(0),
        },
        splitLine: {
          lineStyle: {
            color: '#1E293B',
            type: 'dashed',
          },
        },
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100,
        },
      ],
      series: [
        {
          name: '冷阱温度',
          type: 'line',
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          showSymbol: false,
          lineStyle: {
            width: 2,
            color: '#06B6D4',
          },
          itemStyle: {
            color: '#06B6D4',
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#06B6D440' },
              { offset: 1, color: '#06B6D405' },
            ]),
          },
          markLine: {
            silent: true,
            lineStyle: {
              color: '#F59E0B',
              type: 'dashed',
              width: 1,
            },
            data: [
              {
                yAxis: -50,
                label: {
                  formatter: '阈值 -50℃',
                  color: '#F59E0B',
                  fontSize: 10,
                },
              },
            ],
          },
          data: temps,
          animationDuration: 500,
        },
      ],
    };

    chartInstance.current.setOption(option, true);

    const handleResize = () => {
      chartInstance.current?.resize();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [tempHistory]);

  const handleStartDefrost = async () => {
    if (isCommandExecuting) return;
    setIsCommandExecuting(true);
    try {
      await defrostApi.sendCommand(deviceId, {
        command: 'start',
        heating_power_pct: heatingPower,
        max_duration_minutes: maxDuration,
      });
      const status = await defrostApi.getCurrentStatus(deviceId);
      setCurrentStatus(status);
    } catch (error) {
      console.error('启动除霜失败:', error);
    } finally {
      setIsCommandExecuting(false);
    }
  };

  const handleStopDefrost = async () => {
    if (isCommandExecuting) return;
    setIsCommandExecuting(true);
    try {
      await defrostApi.sendCommand(deviceId, {
        command: 'stop',
      });
      const status = await defrostApi.getCurrentStatus(deviceId);
      setCurrentStatus(status);
    } catch (error) {
      console.error('停止除霜失败:', error);
    } finally {
      setIsCommandExecuting(false);
    }
  };

  const getFrostThicknessColor = (thickness: number | null) => {
    if (thickness === null) return 'text-slate-400';
    if (thickness < 2) return 'text-green-400';
    if (thickness < 3) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getFrostThicknessBgColor = (thickness: number | null) => {
    if (thickness === null) return '#64748B';
    if (thickness < 2) return '#10B981';
    if (thickness < 3) return '#F59E0B';
    return '#EF4444';
  };

  const getDefrostPhaseText = (phase: string) => {
    const phaseMap: Record<string, string> = {
      idle: '待机',
      preheating: '预热中',
      heating: '加热中',
      soaking: '保温中',
      cooling: '冷却中',
    };
    return phaseMap[phase] || phase;
  };

  const getDefrostPhaseColor = (phase: string) => {
    const colorMap: Record<string, string> = {
      idle: 'text-slate-400 bg-slate-500/20',
      preheating: 'text-yellow-400 bg-yellow-500/20',
      heating: 'text-orange-400 bg-orange-500/20',
      soaking: 'text-cyan-400 bg-cyan-500/20',
      cooling: 'text-blue-400 bg-blue-500/20',
    };
    return colorMap[phase] || 'text-slate-400 bg-slate-500/20';
  };

  const formatDateTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatTime = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours > 0) {
      return `${hours}h ${mins}m`;
    }
    return `${mins}m`;
  };

  const progressPercentage = currentStatus?.is_defrosting && currentStatus.elapsed_minutes + currentStatus.remaining_minutes > 0
    ? (currentStatus.elapsed_minutes / (currentStatus.elapsed_minutes + currentStatus.remaining_minutes)) * 100
    : 0;

  const energyConsumed = currentStatus?.is_defrosting
    ? (currentStatus.current_heating_power_pct / 100) * (currentStatus.elapsed_minutes / 60) * 5
    : 0;

  const recommendedInterval = optimizationData?.recommended_defrost_interval_hours || 12;
  const recommendedPower = optimizationData?.recommended_heating_power_pct || 50;

  const frostThickness = currentStatus?.estimated_frost_thickness_mm ?? 0;
  const maxFrostThickness = 5;
  const frostProgress = Math.min((frostThickness / maxFrostThickness) * 100, 100);
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (frostProgress / 100) * circumference;

  if (isLoading && !currentStatus) {
    return (
      <div className="bg-slate-900/50 rounded-xl border border-slate-700 p-6">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500" />
          <span className="ml-3 text-slate-400">加载中...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 flex items-center justify-center">
            <Snowflake className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-100">
              冷阱除霜优化
            </h2>
            <p className="text-sm text-slate-400">
              设备 #{deviceId.toString().padStart(2, '0')} · 智能除霜控制
            </p>
          </div>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-sm font-medium ${
            currentStatus?.is_defrosting
              ? 'bg-orange-500/20 text-orange-400'
              : 'bg-green-500/20 text-green-400'
          }`}
        >
          {currentStatus?.is_defrosting ? '除霜中' : '待机'}
        </span>
      </div>

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-3 bg-slate-900/50 rounded-xl border border-slate-700 p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-4">
            <Snowflake className="w-4 h-4" />
            结霜厚度
          </div>
          <div className="relative w-36 h-36 mx-auto">
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="72"
                cy="72"
                r="45"
                stroke="#1E293B"
                strokeWidth="8"
                fill="none"
              />
              <circle
                cx="72"
                cy="72"
                r="45"
                stroke={getFrostThicknessBgColor(frostThickness)}
                strokeWidth="8"
                fill="none"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                className="transition-all duration-500"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className={`text-3xl font-mono font-bold ${getFrostThicknessColor(frostThickness)}`}>
                {frostThickness !== null ? frostThickness.toFixed(1) : '--'}
              </span>
              <span className="text-xs text-slate-500">mm</span>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
            <div>
              <div className="text-green-400 font-mono">0-2</div>
              <div className="text-slate-500">正常</div>
            </div>
            <div>
              <div className="text-yellow-400 font-mono">2-3</div>
              <div className="text-slate-500">注意</div>
            </div>
            <div>
              <div className="text-red-400 font-mono">3+</div>
              <div className="text-slate-500">告警</div>
            </div>
          </div>
        </div>

        <div className="col-span-5 bg-slate-900/50 rounded-xl border border-slate-700 overflow-hidden">
          <div ref={chartRef} className="w-full h-64" />
        </div>

        <div className="col-span-4 bg-slate-900/50 rounded-xl border border-slate-700 p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-4">
            <Settings className="w-4 h-4" />
            除霜状态
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-sm">当前状态</span>
              <span className={`px-2 py-1 rounded text-xs font-medium ${getDefrostPhaseColor(currentStatus?.defrost_phase || 'idle')}`}>
                {getDefrostPhaseText(currentStatus?.defrost_phase || 'idle')}
              </span>
            </div>

            {currentStatus?.is_defrosting && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">进度</span>
                    <span className="text-cyan-400 font-mono">{progressPercentage.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${progressPercentage}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>已运行 {formatTime(currentStatus.elapsed_minutes)}</span>
                    <span>剩余 {formatTime(currentStatus.remaining_minutes)}</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-700">
                  <div>
                    <div className="text-xs text-slate-500 mb-1">加热功率</div>
                    <div className="text-lg font-mono font-bold text-orange-400">
                      {currentStatus.current_heating_power_pct}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-500 mb-1">
                      <Zap className="w-3 h-3 inline mr-1" />
                      能耗
                    </div>
                    <div className="text-lg font-mono font-bold text-yellow-400">
                      {energyConsumed.toFixed(2)} kWh
                    </div>
                  </div>
                </div>
              </>
            )}

            {!currentStatus?.is_defrosting && (
              <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-700">
                <div>
                  <div className="text-xs text-slate-500 mb-1">
                    <Thermometer className="w-3 h-3 inline mr-1" />
                    冷阱温度
                  </div>
                  <div className={`text-lg font-mono font-bold ${
                    (currentStatus?.cold_trap_temp ?? 0) > -50 ? 'text-yellow-400' : 'text-cyan-400'
                  }`}>
                    {currentStatus?.cold_trap_temp?.toFixed(1) || '--'}℃
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 mb-1">
                    <Snowflake className="w-3 h-3 inline mr-1" />
                    预计厚度
                  </div>
                  <div className={`text-lg font-mono font-bold ${getFrostThicknessColor(frostThickness)}`}>
                    {frostThickness?.toFixed(1) || '--'} mm
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-4 bg-slate-900/50 rounded-xl border border-slate-700 p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-4">
            <TrendingDown className="w-4 h-4 text-green-400" />
            优化建议
          </div>
          
          <div className="space-y-4">
            <div className="p-3 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-lg border border-cyan-500/20">
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-xs text-slate-500 mb-1">推荐除霜间隔</div>
                  <div className="text-2xl font-mono font-bold text-cyan-400">
                    {recommendedInterval}h
                  </div>
                </div>
                <Clock className="w-5 h-5 text-cyan-400/50 mt-1" />
              </div>
              <div className="mt-2 text-xs text-slate-500">
                基于结霜速率和能耗模型优化
              </div>
            </div>

            <div className="p-3 bg-gradient-to-r from-orange-500/10 to-yellow-500/10 rounded-lg border border-orange-500/20">
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-xs text-slate-500 mb-1">推荐加热功率</div>
                  <div className="text-2xl font-mono font-bold text-orange-400">
                    {recommendedPower}%
                  </div>
                </div>
                <Zap className="w-5 h-5 text-orange-400/50 mt-1" />
              </div>
              <div className="mt-2 text-xs text-slate-500">
                平衡除霜效率和能耗
              </div>
            </div>

            {optimizationData && (
              <div className="pt-3 border-t border-slate-700">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">预计节能</span>
                  <span className="text-green-400 font-mono">
                    {optimizationData.predicted_energy_saving_kwh.toFixed(2)} kWh
                  </span>
                </div>
                <div className="flex justify-between text-sm mt-2">
                  <span className="text-slate-400">置信度</span>
                  <span className="text-cyan-400 font-mono">
                    {(optimizationData.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="col-span-4 bg-slate-900/50 rounded-xl border border-slate-700 p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-4">
            <Settings className="w-4 h-4" />
            除霜控制
          </div>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm text-slate-400">加热功率</label>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setHeatingPower(Math.max(20, heatingPower - 5))}
                    className="w-7 h-7 rounded bg-slate-700 hover:bg-slate-600 flex items-center justify-center transition-colors"
                  >
                    <ChevronDown className="w-4 h-4" />
                  </button>
                  <span className="text-lg font-mono font-bold text-orange-400 w-16 text-center">
                    {heatingPower}%
                  </span>
                  <button
                    onClick={() => setHeatingPower(Math.min(80, heatingPower + 5))}
                    className="w-7 h-7 rounded bg-slate-700 hover:bg-slate-600 flex items-center justify-center transition-colors"
                  >
                    <ChevronUp className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <input
                type="range"
                min="20"
                max="80"
                step="5"
                value={heatingPower}
                onChange={(e) => setHeatingPower(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: 'linear-gradient(to right, #f59e0b 0%, #ef4444 100%)',
                }}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm text-slate-400">最长时长</label>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setMaxDuration(Math.max(30, maxDuration - 10))}
                    className="w-7 h-7 rounded bg-slate-700 hover:bg-slate-600 flex items-center justify-center transition-colors"
                  >
                    <ChevronDown className="w-4 h-4" />
                  </button>
                  <span className="text-lg font-mono font-bold text-cyan-400 w-16 text-center">
                    {maxDuration}m
                  </span>
                  <button
                    onClick={() => setMaxDuration(Math.min(120, maxDuration + 10))}
                    className="w-7 h-7 rounded bg-slate-700 hover:bg-slate-600 flex items-center justify-center transition-colors"
                  >
                    <ChevronUp className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <input
                type="range"
                min="30"
                max="120"
                step="10"
                value={maxDuration}
                onChange={(e) => setMaxDuration(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div className="flex gap-3 pt-4 border-t border-slate-700">
              <button
                onClick={handleStartDefrost}
                disabled={currentStatus?.is_defrosting || isCommandExecuting}
                className="flex-1 py-3 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4" />
                {isCommandExecuting ? '执行中...' : '开始除霜'}
              </button>
              <button
                onClick={handleStopDefrost}
                disabled={!currentStatus?.is_defrosting || isCommandExecuting}
                className="flex-1 py-3 bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-600 hover:to-rose-600 disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                <Square className="w-4 h-4" />
                停止除霜
              </button>
            </div>
          </div>
        </div>

        <div className="col-span-4 bg-slate-900/50 rounded-xl border border-slate-700 p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-4">
            <TrendingDown className="w-4 h-4 text-green-400" />
            节能统计
            <span className="text-xs text-slate-500">(近30天)</span>
          </div>
          
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xs text-slate-500 mb-1">实际节能</div>
              <div className="text-xl font-mono font-bold text-green-400">
                {energyStats?.optimization_stats.actual_energy_saving_kwh.toFixed(1) || '--'}
              </div>
              <div className="text-xs text-slate-500">kWh</div>
            </div>
            
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xs text-slate-500 mb-1">除霜次数</div>
              <div className="text-xl font-mono font-bold text-cyan-400">
                {energyStats?.optimization_stats.approved_count || 0}
              </div>
              <div className="text-xs text-slate-500">次</div>
            </div>
            
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xs text-slate-500 mb-1">除霜占比</div>
              <div className="text-xl font-mono font-bold text-purple-400">
                {energyStats?.status_stats.defrosting_ratio.toFixed(1) || '--'}
              </div>
              <div className="text-xs text-slate-500">%</div>
            </div>
            
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xs text-slate-500 mb-1">平均厚度</div>
              <div className="text-xl font-mono font-bold text-orange-400">
                {energyStats?.optimization_stats.avg_frost_thickness_mm.toFixed(1) || '--'}
              </div>
              <div className="text-xs text-slate-500">mm</div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-slate-700">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">总除霜量</span>
              <span className="text-slate-100 font-mono">
                {energyStats?.optimization_stats.total_frost_removed_mm.toFixed(1) || '--'} mm
              </span>
            </div>
            <div className="flex justify-between text-sm mt-2">
              <span className="text-slate-400">平均功率</span>
              <span className="text-slate-100 font-mono">
                {energyStats?.status_stats.avg_running_power_pct.toFixed(0) || '--'}%
              </span>
            </div>
            <div className="flex justify-between text-sm mt-2">
              <span className="text-slate-400">优化批准率</span>
              <span className="text-slate-100 font-mono">
                {energyStats?.optimization_stats.approval_rate.toFixed(0) || '--'}%
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-900/50 rounded-xl border border-slate-700 overflow-hidden">
        <div className="p-4 border-b border-slate-700">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            <h3 className="font-semibold text-slate-100">除霜历史记录</h3>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-800/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                  时间
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                  状态
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                  阶段
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                  功率
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                  已运行
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                  剩余
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                  当前温度
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {statusHistory.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                    暂无历史记录
                  </td>
                </tr>
              ) : (
                statusHistory.slice(0, 10).map((item) => (
                  <tr key={item.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-300 font-mono">
                      {formatDateTime(item.timestamp)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        item.is_defrosting
                          ? 'bg-orange-500/20 text-orange-400'
                          : 'bg-green-500/20 text-green-400'
                      }`}>
                        {item.is_defrosting ? '除霜中' : '待机'}
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-300">
                      {getDefrostPhaseText(item.defrost_phase)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-mono text-orange-400">
                      {item.current_heating_power_pct}%
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-mono text-slate-300">
                      {formatTime(item.elapsed_minutes)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-mono text-slate-300">
                      {formatTime(item.remaining_minutes)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-mono text-cyan-400">
                      {item.current_temp?.toFixed(1) || '--'}℃
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default DefrostStatus;
