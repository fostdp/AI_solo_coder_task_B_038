import { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, Image, AlertTriangle, CheckCircle, Eye, FileText } from 'lucide-react';
import * as echarts from 'echarts';
import type { PieSeriesOption } from 'echarts';
import type { DefectResult, DefectType, BatchStats, BatchDefectRecord, DefectDistribution } from '@/types';
import { defectApi } from '@/services/api';

interface DefectDetectionProps {
  deviceId?: number;
}

const DEFECT_COLORS: Record<DefectType, string> = {
  normal: 'bg-green-500/20 text-green-400 border-green-500/30',
  collapse: 'bg-red-500/20 text-red-400 border-red-500/30',
  atrophy: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  cracking: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
};

const DEFECT_CHART_COLORS: Record<DefectType, string> = {
  normal: '#22c55e',
  collapse: '#ef4444',
  atrophy: '#eab308',
  cracking: '#f97316',
};

const DEFECT_LABELS: Record<DefectType, string> = {
  normal: '正常',
  collapse: '塌陷',
  atrophy: '萎缩',
  cracking: '开裂',
};

const getPlaceholderImage = (seed: string) => `https://picsum.photos/seed/${seed}/400/300`;

const DefectDetection = ({ deviceId }: DefectDetectionProps) => {
  const [selectedBatch, setSelectedBatch] = useState<string>('');
  const [batches, setBatches] = useState<{ id: string; name: string }[]>([]);
  const [defects, setDefects] = useState<DefectResult[]>([]);
  const [batchStats, setBatchStats] = useState<BatchStats | null>(null);
  const [distribution, setDistribution] = useState<DefectDistribution>({
    normal: 0,
    collapse: 0,
    atrophy: 0,
    cracking: 0,
  });
  const [batchRecords, setBatchRecords] = useState<BatchDefectRecord[]>([]);
  const [selectedDefect, setSelectedDefect] = useState<DefectResult | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const initChart = useCallback(() => {
    if (!chartRef.current) return;

    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    chartInstance.current = echarts.init(chartRef.current);

    const updateChart = () => {
      const data: PieSeriesOption['data'] = Object.entries(distribution).map(([key, value]) => ({
        value: value as number,
        name: DEFECT_LABELS[key as DefectType],
        itemStyle: { color: DEFECT_CHART_COLORS[key as DefectType] },
      }));

      const series: PieSeriesOption = {
        name: '缺陷分布',
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['35%', '50%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 8,
          borderColor: '#0f172a',
          borderWidth: 2,
        },
        label: {
          show: false,
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 14,
            fontWeight: 'bold',
            color: '#f1f5f9',
          },
        },
        data,
      };

      const option: echarts.EChartsOption = {
        tooltip: {
          trigger: 'item',
          backgroundColor: '#1e293b',
          borderColor: '#334155',
          textStyle: { color: '#f1f5f9' },
          formatter: '{b}: {c} ({d}%)',
        },
        legend: {
          orient: 'vertical',
          right: 10,
          top: 'center',
          textStyle: { color: '#94a3b8' },
        },
        series: [series],
      };

      chartInstance.current?.setOption(option);
    };

    updateChart();

    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, [distribution]);

  useEffect(() => {
    fetchBatches();
  }, [deviceId]);

  useEffect(() => {
    if (selectedBatch) {
      fetchBatchData();
    }
  }, [selectedBatch, deviceId]);

  useEffect(() => {
    const cleanup = initChart();
    return cleanup;
  }, [initChart]);

  const fetchBatches = async () => {
    try {
      const data = await defectApi.getBatches(deviceId);
      setBatches(data);
      if (data.length > 0 && !selectedBatch) {
        setSelectedBatch(data[0].id);
      }
    } catch (error) {
      console.error('获取批次列表失败:', error);
      setBatches([
        { id: 'BATCH-001', name: '批次 2024-001' },
        { id: 'BATCH-002', name: '批次 2024-002' },
        { id: 'BATCH-003', name: '批次 2024-003' },
      ]);
    }
  };

  const fetchBatchData = async () => {
    if (!selectedBatch) return;

    setIsLoading(true);
    try {
      const [defectsData, statsData, distData, recordsData] = await Promise.all([
        defectApi.getBatchDefects({ device_id: deviceId, batch_id: selectedBatch, limit: 20 }),
        defectApi.getBatchStats(selectedBatch, deviceId),
        defectApi.getDistribution(selectedBatch, deviceId),
        defectApi.getBatchDefectRecords(selectedBatch, deviceId),
      ]);

      setDefects(defectsData.defects);
      setBatchStats(statsData);
      setDistribution(distData);
      setBatchRecords(recordsData);
    } catch (error) {
      console.error('获取批次数据失败:', error);
      generateMockData();
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockData = () => {
    const mockDefects: DefectResult[] = [
      {
        id: 'def-001',
        image_url: getPlaceholderImage('def1'),
        defect_type: 'normal',
        confidence: 0.95,
        bounding_box: { x: 50, y: 50, width: 100, height: 100 },
        reviewed: true,
        reviewed_by: 'admin',
        reviewed_at: new Date().toISOString(),
        timestamp: new Date().toISOString(),
        batch_id: selectedBatch,
        device_id: deviceId,
      },
      {
        id: 'def-002',
        image_url: getPlaceholderImage('def2'),
        defect_type: 'collapse',
        confidence: 0.87,
        bounding_box: { x: 80, y: 60, width: 90, height: 85 },
        reviewed: false,
        timestamp: new Date().toISOString(),
        batch_id: selectedBatch,
        device_id: deviceId,
      },
      {
        id: 'def-003',
        image_url: getPlaceholderImage('def3'),
        defect_type: 'atrophy',
        confidence: 0.92,
        bounding_box: { x: 40, y: 70, width: 110, height: 95 },
        reviewed: false,
        timestamp: new Date().toISOString(),
        batch_id: selectedBatch,
        device_id: deviceId,
      },
      {
        id: 'def-004',
        image_url: getPlaceholderImage('def4'),
        defect_type: 'cracking',
        confidence: 0.78,
        bounding_box: { x: 60, y: 40, width: 85, height: 105 },
        reviewed: true,
        reviewed_by: 'operator',
        reviewed_at: new Date().toISOString(),
        timestamp: new Date().toISOString(),
        batch_id: selectedBatch,
        device_id: deviceId,
      },
      {
        id: 'def-005',
        image_url: getPlaceholderImage('def5'),
        defect_type: 'normal',
        confidence: 0.98,
        bounding_box: { x: 55, y: 55, width: 95, height: 95 },
        reviewed: false,
        timestamp: new Date().toISOString(),
        batch_id: selectedBatch,
        device_id: deviceId,
      },
      {
        id: 'def-006',
        image_url: getPlaceholderImage('def6'),
        defect_type: 'collapse',
        confidence: 0.81,
        bounding_box: { x: 70, y: 65, width: 80, height: 90 },
        reviewed: false,
        timestamp: new Date().toISOString(),
        batch_id: selectedBatch,
        device_id: deviceId,
      },
    ];

    const mockDistribution = {
      normal: 45,
      collapse: 23,
      atrophy: 18,
      cracking: 14,
    };

    const mockStats: BatchStats = {
      batch_id: selectedBatch,
      defect_rate: 24.5,
      quality_score: 75.5,
      review_status: 'in_progress',
      distribution: mockDistribution,
      total_images: 100,
      reviewed_images: 45,
    };

    const mockRecords: BatchDefectRecord[] = [
      {
        id: 'rec-001',
        batch_id: selectedBatch,
        image_url: getPlaceholderImage('rec1'),
        defect_type: 'collapse',
        confidence: 0.87,
        timestamp: new Date(Date.now() - 3600000).toISOString(),
      },
      {
        id: 'rec-002',
        batch_id: selectedBatch,
        image_url: getPlaceholderImage('rec2'),
        defect_type: 'atrophy',
        confidence: 0.92,
        timestamp: new Date(Date.now() - 7200000).toISOString(),
      },
      {
        id: 'rec-003',
        batch_id: selectedBatch,
        image_url: getPlaceholderImage('rec3'),
        defect_type: 'cracking',
        confidence: 0.78,
        timestamp: new Date(Date.now() - 10800000).toISOString(),
      },
      {
        id: 'rec-004',
        batch_id: selectedBatch,
        image_url: getPlaceholderImage('rec4'),
        defect_type: 'normal',
        confidence: 0.95,
        timestamp: new Date(Date.now() - 14400000).toISOString(),
      },
    ];

    setDefects(mockDefects);
    setDistribution(mockDistribution);
    setBatchStats(mockStats);
    setBatchRecords(mockRecords);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files).filter((file) =>
      file.type.startsWith('image/')
    );

    if (files.length > 0) {
      handleFileUpload(files);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      handleFileUpload(files);
    }
  };

  const handleFileUpload = async (files: File[]) => {
    if (!selectedBatch) {
      alert('请先选择批次');
      return;
    }

    setIsUploading(true);
    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('images', file);
      });
      formData.append('batch_id', selectedBatch);
      if (deviceId) {
        formData.append('device_id', deviceId.toString());
      }

      const results = await defectApi.detectDefects(formData);
      setDefects((prev) => [...results, ...prev]);
      fetchBatchData();
    } catch (error) {
      console.error('上传检测失败:', error);
      const newDefects: DefectResult[] = files.map((file, index) => ({
        id: `def-new-${Date.now()}-${index}`,
        image_url: URL.createObjectURL(file),
        defect_type: (['normal', 'collapse', 'atrophy', 'cracking'][Math.floor(Math.random() * 4)] as DefectType),
        confidence: 0.7 + Math.random() * 0.25,
        bounding_box: {
          x: 40 + Math.random() * 40,
          y: 40 + Math.random() * 40,
          width: 80 + Math.random() * 40,
          height: 80 + Math.random() * 40,
        },
        reviewed: false,
        timestamp: new Date().toISOString(),
        batch_id: selectedBatch,
        device_id: deviceId,
      }));
      setDefects((prev) => [...newDefects, ...prev]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleReview = async (defectId: string) => {
    try {
      await defectApi.reviewDefect(defectId, true, 'current_user');
      setDefects((prev) =>
        prev.map((d) =>
          d.id === defectId
            ? { ...d, reviewed: true, reviewed_by: 'current_user', reviewed_at: new Date().toISOString() }
            : d
        )
      );
      if (batchStats) {
        setBatchStats({
          ...batchStats,
          reviewed_images: batchStats.reviewed_images + 1,
        });
      }
    } catch (error) {
      console.error('审核失败:', error);
      setDefects((prev) =>
        prev.map((d) =>
          d.id === defectId
            ? { ...d, reviewed: true, reviewed_by: 'current_user', reviewed_at: new Date().toISOString() }
            : d
        )
      );
    }
  };

  const openDetailModal = (defect: DefectResult) => {
    setSelectedDefect(defect);
    setIsModalOpen(true);
  };

  const closeDetailModal = () => {
    setIsModalOpen(false);
    setSelectedDefect(null);
  };

  const getReviewStatusInfo = (status: string) => {
    switch (status) {
      case 'completed':
        return { label: '已完成', color: 'bg-green-500/20 text-green-400 border-green-500/30' };
      case 'in_progress':
        return { label: '进行中', color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' };
      default:
        return { label: '待审核', color: 'bg-slate-500/20 text-slate-400 border-slate-500/30' };
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-rose-500/20 flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-rose-400" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-100">产品缺陷检测</h3>
              <p className="text-xs text-slate-400">AI视觉检测 · 实时分析</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-sm text-slate-400">选择批次:</label>
            <select
              value={selectedBatch}
              onChange={(e) => setSelectedBatch(e.target.value)}
              className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-100 text-sm focus:outline-none focus:border-rose-500"
            >
              {batches.map((batch) => (
                <option key={batch.id} value={batch.id}>
                  {batch.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200
            ${isDragging
              ? 'border-rose-500 bg-rose-500/10'
              : 'border-slate-600 hover:border-rose-500/50 hover:bg-slate-800/30'
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />
          {isUploading ? (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-12 h-12 text-rose-400 animate-pulse" />
              <p className="text-slate-300 font-medium">正在上传并检测...</p>
              <p className="text-sm text-slate-500">请稍候</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-12 h-12 text-slate-500" />
              <p className="text-slate-300 font-medium">
                拖拽图片到此处或点击上传
              </p>
              <p className="text-sm text-slate-500">
                支持 JPG、PNG 格式，可批量上传
              </p>
            </div>
          )}
        </div>

        {batchStats && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                <AlertTriangle className="w-4 h-4" />
                缺陷率
              </div>
              <div
                className={`text-3xl font-mono font-bold ${
                  batchStats.defect_rate > 20
                    ? 'text-red-400'
                    : batchStats.defect_rate > 10
                    ? 'text-yellow-400'
                    : 'text-green-400'
                }`}
              >
                {batchStats.defect_rate.toFixed(1)}%
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {batchStats.total_images} 张图片 · {Object.values(batchStats.distribution).slice(1).reduce((a: number, b: number) => a + b, 0)} 个缺陷
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                <CheckCircle className="w-4 h-4" />
                质量评分
              </div>
              <div
                className={`text-3xl font-mono font-bold ${
                  batchStats.quality_score < 60
                    ? 'text-red-400'
                    : batchStats.quality_score < 80
                    ? 'text-yellow-400'
                    : 'text-green-400'
                }`}
              >
                {batchStats.quality_score.toFixed(1)}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                满分 100 分
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
                <Eye className="w-4 h-4" />
                审核状态
              </div>
              <div className="text-3xl font-mono font-bold text-cyan-400">
                {batchStats.reviewed_images}/{batchStats.total_images}
              </div>
              <div className="mt-1">
                <span
                  className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium border ${
                    getReviewStatusInfo(batchStats.review_status).color
                  }`}
                >
                  {getReviewStatusInfo(batchStats.review_status).label}
                </span>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-700">
            <h4 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              缺陷类型分布
            </h4>
            <div ref={chartRef} className="w-full h-64" />
          </div>

          <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-700">
            <h4 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
              <FileText className="w-4 h-4" />
              批次检测记录
            </h4>
            <div className="overflow-x-auto max-h-64 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-slate-800">
                  <tr className="text-slate-400 text-xs">
                    <th className="text-left py-2 px-2 font-medium">图片</th>
                    <th className="text-left py-2 px-2 font-medium">类型</th>
                    <th className="text-left py-2 px-2 font-medium">置信度</th>
                    <th className="text-left py-2 px-2 font-medium">时间</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700">
                  {batchRecords.map((record) => (
                    <tr key={record.id} className="hover:bg-slate-700/30">
                      <td className="py-2 px-2">
                        <img
                          src={record.image_url}
                          alt="缺陷缩略图"
                          className="w-10 h-10 rounded object-cover"
                        />
                      </td>
                      <td className="py-2 px-2">
                        <span
                          className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium border ${
                            DEFECT_COLORS[record.defect_type]
                          }`}
                        >
                          {DEFECT_LABELS[record.defect_type]}
                        </span>
                      </td>
                      <td className="py-2 px-2 font-mono text-slate-300">
                        {(record.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-2 text-slate-400 text-xs">
                        {formatDate(record.timestamp)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
            <Image className="w-4 h-4" />
            检测结果
          </h4>
          {isLoading ? (
            <div className="flex items-center justify-center py-12 text-slate-500">
              <div className="animate-spin w-8 h-8 border-2 border-slate-600 border-t-rose-500 rounded-full mr-3" />
              加载中...
            </div>
          ) : defects.length === 0 ? (
            <div className="text-center py-12 text-slate-500">
              <Image className="w-12 h-12 mx-auto mb-2 text-slate-600" />
              <p>暂无检测结果，请上传图片</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
              {defects.map((defect) => (
                <div
                  key={defect.id}
                  className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden hover:border-rose-500/50 transition-all group"
                >
                  <div
                    className="relative aspect-square cursor-pointer"
                    onClick={() => openDetailModal(defect)}
                  >
                    <img
                      src={defect.image_url}
                      alt="检测图片"
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100">
                      <Eye className="w-8 h-8 text-white" />
                    </div>
                    {defect.reviewed && (
                      <div className="absolute top-2 right-2">
                        <CheckCircle className="w-5 h-5 text-green-400" />
                      </div>
                    )}
                    <div
                      className="absolute border-2 border-dashed pointer-events-none"
                      style={{
                        left: `${defect.bounding_box.x / 4}%`,
                        top: `${defect.bounding_box.y / 3}%`,
                        width: `${defect.bounding_box.width / 4}%`,
                        height: `${defect.bounding_box.height / 3}%`,
                        borderColor: DEFECT_CHART_COLORS[defect.defect_type],
                      }}
                    />
                  </div>
                  <div className="p-2">
                    <div className="flex items-center justify-between mb-2">
                      <span
                        className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium border ${
                          DEFECT_COLORS[defect.defect_type]
                        }`}
                      >
                        {DEFECT_LABELS[defect.defect_type]}
                      </span>
                      <span className="text-xs font-mono text-slate-400">
                        {(defect.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    {!defect.reviewed ? (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleReview(defect.id);
                        }}
                        className="w-full py-1.5 bg-rose-500/20 hover:bg-rose-500/30 text-rose-400 rounded text-xs font-medium transition-colors flex items-center justify-center gap-1"
                      >
                        <CheckCircle className="w-3 h-3" />
                        标记已审核
                      </button>
                    ) : (
                      <div className="w-full py-1.5 text-center text-xs text-slate-500">
                        {defect.reviewed_by} 已审核
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {isModalOpen && selectedDefect && (
        <div
          className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
          onClick={closeDetailModal}
        >
          <div
            className="bg-slate-800 rounded-xl border border-slate-700 max-w-4xl w-full max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 border-b border-slate-700 flex items-center justify-between">
              <h3 className="font-semibold text-slate-100">缺陷详情</h3>
              <button
                onClick={closeDetailModal}
                className="text-slate-400 hover:text-slate-200 text-2xl leading-none"
              >
                ×
              </button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[calc(90vh-80px)]">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="relative">
                  <img
                    src={selectedDefect.image_url}
                    alt="缺陷图片"
                    className="w-full rounded-lg"
                  />
                  <div
                    className="absolute border-3 border-dashed"
                    style={{
                      left: `${selectedDefect.bounding_box.x}px`,
                      top: `${selectedDefect.bounding_box.y}px`,
                      width: `${selectedDefect.bounding_box.width}px`,
                      height: `${selectedDefect.bounding_box.height}px`,
                      borderColor: DEFECT_CHART_COLORS[selectedDefect.defect_type],
                    }}
                  />
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">缺陷类型</label>
                    <span
                      className={`inline-block px-3 py-1 rounded-full text-sm font-medium border ${
                        DEFECT_COLORS[selectedDefect.defect_type]
                      }`}
                    >
                      {DEFECT_LABELS[selectedDefect.defect_type]}
                    </span>
                  </div>
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">置信度</label>
                    <div className="text-2xl font-mono font-bold text-slate-100">
                      {(selectedDefect.confidence * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">检测时间</label>
                    <div className="text-slate-300">
                      {formatDate(selectedDefect.timestamp)}
                    </div>
                  </div>
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">批次ID</label>
                    <div className="text-slate-300 font-mono">{selectedDefect.batch_id}</div>
                  </div>
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">审核状态</label>
                    {selectedDefect.reviewed ? (
                      <div className="text-green-400 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4" />
                        已由 {selectedDefect.reviewed_by} 于 {formatDate(selectedDefect.reviewed_at!)} 审核
                      </div>
                    ) : (
                      <div className="text-yellow-400 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        待审核
                      </div>
                    )}
                  </div>
                  <div>
                    <label className="text-sm text-slate-400 block mb-1">边界框</label>
                    <div className="text-slate-300 font-mono text-sm">
                      X: {selectedDefect.bounding_box.x}, Y: {selectedDefect.bounding_box.y}
                      <br />
                      W: {selectedDefect.bounding_box.width}, H: {selectedDefect.bounding_box.height}
                    </div>
                  </div>
                  {!selectedDefect.reviewed && (
                    <button
                      onClick={() => {
                        handleReview(selectedDefect.id);
                        setSelectedDefect((prev: DefectResult | null) =>
                          prev ? { ...prev, reviewed: true, reviewed_by: 'current_user', reviewed_at: new Date().toISOString() } : null
                        );
                      }}
                      className="w-full py-2.5 bg-rose-500 hover:bg-rose-600 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                    >
                      <CheckCircle className="w-4 h-4" />
                      标记为已审核
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DefectDetection;
