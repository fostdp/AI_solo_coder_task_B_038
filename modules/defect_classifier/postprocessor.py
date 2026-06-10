"""
缺陷后处理器
包含NMS、边界框平滑、聚类等功能
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter

from .types import DefectCandidate, BoundingBox, BatchDefectStats


class DefectPostprocessor:
    """
    缺陷后处理器
    
    功能：
    1. 非极大值抑制(NMS)
    2. 边界框平滑
    3. 缺陷聚类
    4. 结果汇总和统计
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._nms_threshold = self.config.get('nms_threshold', 0.5)
        self._min_confidence = self.config.get('min_confidence', 0.3)
        self._smoothing_enabled = self.config.get('smoothing', {}).get('enabled', True)
        self._clustering_enabled = self.config.get('clustering', {}).get('enabled', True)
    
    def apply_nms(self, candidates: List[DefectCandidate]) -> List[DefectCandidate]:
        """
        应用非极大值抑制(NMS)
        
        Args:
            candidates: 缺陷候选列表
            
        Returns:
            过滤后的候选列表
        """
        if len(candidates) <= 1:
            return candidates
        
        candidates_with_bbox = [c for c in candidates if c.bbox is not None]
        candidates_without_bbox = [c for c in candidates if c.bbox is None]
        
        if not candidates_with_bbox:
            return self._filter_by_confidence(candidates)
        
        bboxes = np.array([[c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height] for c in candidates_with_bbox])
        scores = np.array([c.confidence for c in candidates_with_bbox])
        
        keep_indices = self._nms(bboxes, scores, self._nms_threshold)
        
        result = [candidates_with_bbox[i] for i in keep_indices]
        result.extend(candidates_without_bbox)
        
        return self._filter_by_confidence(result)
    
    def _nms(self, bboxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """
        非极大值抑制核心算法
        
        Args:
            bboxes: 边界框数组，shape: [N, 4] (x, y, w, h)
            scores: 置信度数组，shape: [N]
            threshold: IoU阈值
            
        Returns:
            保留的索引列表
        """
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 0] + bboxes[:, 2]
        y2 = bboxes[:, 1] + bboxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _filter_by_confidence(self, candidates: List[DefectCandidate]) -> List[DefectCandidate]:
        """按置信度过滤"""
        return [c for c in candidates if c.confidence >= self._min_confidence]
    
    def smooth_bounding_boxes(self, candidates: List[DefectCandidate], 
                              image_shape: Tuple[int, int]) -> List[DefectCandidate]:
        """
        平滑边界框
        
        Args:
            candidates: 缺陷候选列表
            image_shape: 图像形状 (height, width)
            
        Returns:
            平滑后的候选列表
        """
        if not self._smoothing_enabled:
            return candidates
        
        smoothed = []
        h, w = image_shape[:2]
        
        for candidate in candidates:
            if candidate.bbox is None:
                smoothed.append(candidate)
                continue
            
            bbox = candidate.bbox
            margin = self.config.get('smoothing', {}).get('margin', 5)
            
            new_x = max(0, bbox.x - margin)
            new_y = max(0, bbox.y - margin)
            new_w = min(w - new_x, bbox.width + 2 * margin)
            new_h = min(h - new_y, bbox.height + 2 * margin)
            
            smoothed_bbox = BoundingBox(
                x=new_x,
                y=new_y,
                width=new_w,
                height=new_h,
                confidence=bbox.confidence,
                defect_type=bbox.defect_type
            )
            
            smoothed.append(DefectCandidate(
                defect_type=candidate.defect_type,
                confidence=candidate.confidence,
                bbox=smoothed_bbox,
                features=candidate.features
            ))
        
        return smoothed
    
    def cluster_defects(self, candidates: List[DefectCandidate], 
                        image_shape: Tuple[int, int]) -> List[DefectCandidate]:
        """
        聚类相近的缺陷
        
        Args:
            candidates: 缺陷候选列表
            image_shape: 图像形状
            
        Returns:
            聚类后的候选列表
        """
        if not self._clustering_enabled or len(candidates) < 2:
            return candidates
        
        distance_threshold = self.config.get('clustering', {}).get('distance_threshold', 30)
        
        clusters = []
        used = [False] * len(candidates)
        
        for i, candidate in enumerate(candidates):
            if used[i] or candidate.bbox is None:
                continue
            
            cluster = [candidate]
            used[i] = True
            
            center_i = self._get_center(candidate.bbox)
            
            for j, other in enumerate(candidates):
                if used[j] or other.bbox is None or j == i:
                    continue
                
                if other.defect_type != candidate.defect_type:
                    continue
                
                center_j = self._get_center(other.bbox)
                distance = np.sqrt((center_i[0] - center_j[0])**2 + 
                                   (center_i[1] - center_j[1])**2)
                
                if distance < distance_threshold:
                    cluster.append(other)
                    used[j] = True
            
            if len(cluster) > 1:
                merged = self._merge_cluster(cluster)
                clusters.append(merged)
            else:
                clusters.append(candidate)
        
        for i, candidate in enumerate(candidates):
            if not used[i]:
                clusters.append(candidate)
        
        return clusters
    
    def _get_center(self, bbox: BoundingBox) -> Tuple[float, float]:
        """获取边界框中心"""
        return (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2)
    
    def _merge_cluster(self, cluster: List[DefectCandidate]) -> DefectCandidate:
        """合并一个聚类中的多个候选"""
        defect_type = cluster[0].defect_type
        max_conf = max(c.confidence for c in cluster)
        avg_conf = np.mean([c.confidence for c in cluster])
        
        bboxes = [c.bbox for c in cluster if c.bbox is not None]
        
        if bboxes:
            x_min = min(b.x for b in bboxes)
            y_min = min(b.y for b in bboxes)
            x_max = max(b.x + b.width for b in bboxes)
            y_max = max(b.y + b.height for b in bboxes)
            
            merged_bbox = BoundingBox(
                x=int(x_min),
                y=int(y_min),
                width=int(x_max - x_min),
                height=int(y_max - y_min),
                confidence=max_conf,
                defect_type=defect_type
            )
        else:
            merged_bbox = None
        
        return DefectCandidate(
            defect_type=defect_type,
            confidence=max_conf,
            bbox=merged_bbox,
            features=None
        )
    
    def process(self, candidates: List[DefectCandidate], 
                image_shape: Tuple[int, int]) -> List[DefectCandidate]:
        """
        完整后处理流程
        
        Args:
            candidates: 缺陷候选列表
            image_shape: 图像形状
            
        Returns:
            处理后的候选列表
        """
        result = self.apply_nms(candidates)
        result = self.smooth_bounding_boxes(result, image_shape)
        result = self.cluster_defects(result, image_shape)
        
        return result
    
    def aggregate_results(self, all_candidates: List[List[DefectCandidate]]) -> Dict:
        """
        汇总多张图像的结果
        
        Args:
            all_candidates: 每张图像的候选列表
            
        Returns:
            汇总统计
        """
        defect_types = []
        confidences = []
        total_defects = 0
        
        for candidates in all_candidates:
            for candidate in candidates:
                if candidate.defect_type != 'normal':
                    defect_types.append(candidate.defect_type)
                    confidences.append(candidate.confidence)
                    total_defects += 1
        
        type_counts = dict(Counter(defect_types))
        
        return {
            'total_defects': total_defects,
            'defect_type_counts': type_counts,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'max_confidence': float(np.max(confidences)) if confidences else 0.0,
            'images_with_defects': sum(1 for c in all_candidates 
                                       if any(d.defect_type != 'normal' for d in c))
        }
    
    def update_batch_stats(self, stats: BatchDefectStats, 
                          all_candidates: List[List[DefectCandidate]]) -> BatchDefectStats:
        """
        根据后处理结果更新批次统计
        
        Args:
            stats: 原始批次统计
            all_candidates: 后处理后的候选列表
            
        Returns:
            更新后的批次统计
        """
        aggregated = self.aggregate_results(all_candidates)
        
        defect_images = aggregated['images_with_defects']
        total_images = stats.total_images
        
        defect_counts = {**stats.defect_counts}
        for defect_type, count in aggregated['defect_type_counts'].items():
            defect_counts[defect_type] = defect_counts.get(defect_type, 0)
        
        defect_rate = defect_images / total_images if total_images > 0 else 0.0
        
        return BatchDefectStats(
            batch_id=stats.batch_id,
            total_images=total_images,
            defect_images=defect_images,
            defect_rate=defect_rate,
            defect_counts=defect_counts,
            quality_score=1.0 - defect_rate,
            needs_review=defect_rate > self.config.get('review_threshold', 0.1)
        )
