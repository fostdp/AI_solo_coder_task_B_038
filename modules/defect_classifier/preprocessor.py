"""
图像预处理器
包含光照归一化、数据增强、域适应等功能
"""

import numpy as np
import base64
import io
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import Counter

from .types import ClassifierConfig, ImageQualityResult


class ImagePreprocessor:
    """图像预处理器"""

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.preproc_config = config.image_preprocessing
        self.quality_config = self.preproc_config.get('quality_check', {})
        
        self.target_size = tuple(self.preproc_config.get('resize', [224, 224]))
        self.mean = np.array(self.preproc_config.get('normalization', {}).get(
            'mean', [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(self.preproc_config.get('normalization', {}).get(
            'std', [0.229, 0.224, 0.225]), dtype=np.float32)

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图像（支持文件路径和base64）"""
        try:
            if image_path.startswith('data:image') or image_path.startswith('base64:'):
                return self._load_base64_image(image_path)
            else:
                return self._load_file_image(image_path)
        except Exception as e:
            print(f"[ImagePreprocessor] 加载图像失败: {e}")
            return None

    def _load_file_image(self, image_path: str) -> np.ndarray:
        """从文件加载图像"""
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except ImportError:
            return self._simulate_load_image(image_path)

    def _load_base64_image(self, base64_str: str) -> np.ndarray:
        """从base64加载图像"""
        try:
            from PIL import Image
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            elif base64_str.startswith('base64:'):
                base64_str = base64_str[7:]
            
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except ImportError:
            return self._simulate_load_image(base64_str)

    def _simulate_load_image(self, identifier: str) -> np.ndarray:
        """模拟加载图像（当PIL不可用时）"""
        np.random.seed(hash(identifier) % (2**32))
        img = np.random.randint(180, 255, (224, 224, 3), dtype=np.uint8)
        return img

    def check_quality(self, image: np.ndarray) -> ImageQualityResult:
        """检查图像质量"""
        if not self.quality_config.get('enabled', True):
            return ImageQualityResult(
                passed=True,
                brightness=128.0,
                contrast=50.0,
                blurriness=50.0,
                resolution=image.shape[:2]
            )

        h, w = image.shape[:2]
        min_res = self.quality_config.get('min_resolution', [100, 100])
        if h < min_res[0] or w < min_res[1]:
            return ImageQualityResult(
                passed=False,
                brightness=float(np.mean(image)),
                contrast=float(np.std(image)),
                blurriness=0.0,
                resolution=(h, w),
                error_message=f"分辨率过低: {w}x{h} < {min_res[0]}x{min_res[1]}"
            )

        brightness = float(np.mean(image))
        max_bright = self.quality_config.get('max_brightness', 250)
        min_bright = self.quality_config.get('min_brightness', 5)
        if brightness > max_bright or brightness < min_bright:
            return ImageQualityResult(
                passed=False,
                brightness=brightness,
                contrast=float(np.std(image)),
                blurriness=0.0,
                resolution=(h, w),
                error_message=f"亮度异常: {brightness:.1f} 超出范围 [{min_bright}, {max_bright}]"
            )

        blurriness = self._calculate_blurriness(image)
        blur_threshold = self.quality_config.get('blurriness_threshold', 100)
        if blurriness < blur_threshold:
            return ImageQualityResult(
                passed=False,
                brightness=brightness,
                contrast=float(np.std(image)),
                blurriness=blurriness,
                resolution=(h, w),
                error_message=f"图像模糊: {blurriness:.1f} < {blur_threshold}"
            )

        return ImageQualityResult(
            passed=True,
            brightness=brightness,
            contrast=float(np.std(image)),
            blurriness=blurriness,
            resolution=(h, w)
        )

    def _calculate_blurriness(self, image: np.ndarray) -> float:
        """计算模糊度（拉普拉斯方差）"""
        gray = np.mean(image, axis=2) if image.ndim == 3 else image
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        
        pad_h, pad_w = 1, 1
        padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        laplacian = np.zeros_like(gray, dtype=np.float32)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                region = padded[i:i+3, j:j+3]
                laplacian[i, j] = np.sum(region * kernel)
        
        return float(np.var(laplacian))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        try:
            from PIL import Image
            img = Image.fromarray(image)
            img = img.resize(self.target_size, Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            img_array = self._simulate_resize(image)
        
        img_array = (img_array - self.mean) / self.std
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def _simulate_resize(self, image: np.ndarray) -> np.ndarray:
        """模拟resize"""
        h, w = self.target_size
        if image.shape[0] == h and image.shape[1] == w:
            return image.astype(np.float32) / 255.0
        
        src_h, src_w = image.shape[:2]
        resized = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                src_i = int(i * src_h / h)
                src_j = int(j * src_w / w)
                resized[i, j] = image[src_i, src_j]
        
        return resized / 255.0
    
    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        光照归一化 - 处理光照强度变化和色温偏移
        组合多种方法：直方图均衡化 + 颜色恒常性 + 自适应归一化
        """
        img_float = image.astype(np.float32)
        
        gray_world = self._gray_world_color_constancy(img_float)
        
        clahe_result = self._adaptive_histogram_equalization(gray_world)
        
        normalized = self._brightness_normalization(clahe_result)
        
        enhanced = self._contrast_enhancement(normalized)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _gray_world_color_constancy(self, image: np.ndarray) -> np.ndarray:
        """Gray World假设 - 校正色温偏移"""
        result = image.copy()
        
        avg_r = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_b = np.mean(image[:, :, 2])
        
        avg_gray = (avg_r + avg_g + avg_b) / 3.0
        
        if avg_gray > 0:
            gain_r = avg_gray / (avg_r + 1e-8)
            gain_g = avg_gray / (avg_g + 1e-8)
            gain_b = avg_gray / (avg_b + 1e-8)
            
            gain_r = np.clip(gain_r, 0.5, 2.0)
            gain_g = np.clip(gain_g, 0.5, 2.0)
            gain_b = np.clip(gain_b, 0.5, 2.0)
            
            result[:, :, 0] *= gain_r
            result[:, :, 1] *= gain_g
            result[:, :, 2] *= gain_b
        
        return result
    
    def _adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        自适应直方图均衡化（CLAHE简化版）
        分块处理，避免全局均衡化导致的噪声放大
        """
        result = np.zeros_like(image, dtype=np.float32)
        h, w = image.shape[:2]
        
        tile_size = 32
        clip_limit = 0.02
        
        for channel in range(3):
            channel_data = image[:, :, channel]
            
            for i in range(0, h, tile_size):
                for j in range(0, w, tile_size):
                    i_end = min(i + tile_size, h)
                    j_end = min(j + tile_size, w)
                    
                    tile = channel_data[i:i_end, j:j_end]
                    
                    hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
                    hist = hist.astype(np.float32) / hist.sum()
                    
                    clip_amount = clip_limit / 256.0
                    excess = np.sum(np.maximum(hist - clip_amount, 0))
                    hist = np.minimum(hist, clip_amount)
                    hist += excess / 256.0
                    
                    cdf = np.cumsum(hist) * 255
                    
                    tile_equalized = np.interp(tile.flatten(), np.arange(256), cdf)
                    result[i:i_end, j:j_end, channel] = tile_equalized.reshape(tile.shape)
        
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        for channel in range(3):
            padded = np.pad(result[:, :, channel], ((1, 1), (1, 1)), mode='reflect')
            smooth = np.zeros_like(result[:, :, channel])
            for i in range(h):
                for j in range(w):
                    smooth[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
            result[:, :, channel] = smooth
        
        return result
    
    def _brightness_normalization(self, image: np.ndarray) -> np.ndarray:
        """亮度归一化 - 将平均亮度调整到目标范围"""
        result = image.copy()
        
        current_brightness = np.mean(image)
        target_brightness = 128.0
        
        if current_brightness > 10:
            brightness_factor = target_brightness / current_brightness
            brightness_factor = np.clip(brightness_factor, 0.7, 1.5)
            
            for c in range(3):
                channel = result[:, :, c]
                normalized = channel / 255.0
                adjusted = 1.0 / (1.0 + np.exp(-(normalized - 0.5) * brightness_factor * 4))
                result[:, :, c] = adjusted * 255
        
        return result
    
    def _contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """对比度受限增强"""
        result = image.copy()
        
        current_std = np.std(image)
        target_std = 60.0
        
        if current_std > 0:
            contrast_factor = target_std / current_std
            contrast_factor = np.clip(contrast_factor, 0.8, 1.3)
            
            mean = np.mean(image, axis=(0, 1))
            for c in range(3):
                result[:, :, c] = mean[c] + (image[:, :, c] - mean[c]) * contrast_factor
        
        return result
    
    def augment_image(self, image: np.ndarray, augmentation_config: Optional[Dict] = None) -> List[np.ndarray]:
        """
        数据增强 - 生成多个变体用于推理时的集成预测
        用于提高光照变化下的鲁棒性
        
        返回多个增强后的图像，最终预测取平均
        """
        if augmentation_config is None:
            augmentation_config = self.preproc_config.get('augmentation', {})
        
        if not augmentation_config.get('enabled', False):
            return [image]
        
        augmented = [image]
        
        num_variants = augmentation_config.get('num_variants', 4)
        
        brightness_range = augmentation_config.get('brightness_range', [-0.15, 0.15])
        contrast_range = augmentation_config.get('contrast_range', [-0.1, 0.1])
        
        for i in range(num_variants):
            variant = image.astype(np.float32)
            
            brightness_factor = 1.0 + np.random.uniform(*brightness_range)
            variant *= brightness_factor
            
            contrast_factor = 1.0 + np.random.uniform(*contrast_range)
            mean = np.mean(variant)
            variant = mean + (variant - mean) * contrast_factor
            
            noise_level = augmentation_config.get('noise_std', 3.0)
            noise = np.random.normal(0, noise_level, variant.shape)
            variant += noise
            
            if np.random.random() > 0.5:
                variant = np.fliplr(variant)
            
            rotation_angle = np.random.uniform(-3, 3)
            if abs(rotation_angle) > 0.5:
                variant = self._rotate_image(variant, rotation_angle)
            
            variant = np.clip(variant, 0, 255).astype(np.uint8)
            augmented.append(variant)
        
        return augmented
    
    def _rotate_image(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        """旋转图像（简化实现）"""
        h, w = image.shape[:2]
        center = (h // 2, w // 2)
        angle_rad = np.radians(angle_deg)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotated = np.zeros_like(image, dtype=image.dtype)
        
        for i in range(h):
            for j in range(w):
                y = i - center[0]
                x = j - center[1]
                
                new_y = int(y * cos_a - x * sin_a + center[0])
                new_x = int(y * sin_a + x * cos_a + center[1])
                
                if 0 <= new_y < h and 0 <= new_x < w:
                    rotated[i, j] = image[new_y, new_x]
        
        return rotated
    
    def domain_adaptation_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        域适应预处理 - 减少不同光照条件下的域差异
        基于特征归一化和风格迁移简化版
        """
        normalized = self.normalize_illumination(image)
        
        standardized = np.zeros_like(normalized, dtype=np.float32)
        for c in range(3):
            channel = normalized[:, :, c].astype(np.float32)
            mean = np.mean(channel)
            std = np.std(channel) + 1e-8
            standardized[:, :, c] = (channel - mean) / std
        
        matched = self._histogram_matching(standardized)
        
        smoothed = self._edge_preserving_smoothing(matched)
        
        for c in range(3):
            channel = smoothed[:, :, c]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                smoothed[:, :, c] = (channel - min_val) / (max_val - min_val) * 255
        
        return np.clip(smoothed, 0, 255).astype(np.uint8)
    
    def _histogram_matching(self, image: np.ndarray) -> np.ndarray:
        """
        直方图匹配 - 将图像直方图匹配到参考分布
        参考分布为标准光照下的高斯分布
        """
        result = image.copy()
        
        ref_mean = 0.0
        ref_std = 1.0
        
        for c in range(3):
            channel = image[:, :, c]
            
            sorted_vals = np.sort(channel.flatten())
            cdf = np.arange(len(sorted_vals)) / len(sorted_vals)
            
            matched_vals = ref_mean + ref_std * np.sqrt(2) * np.arcsinh(2 * cdf - 1)
            
            result[:, :, c] = np.interp(channel.flatten(), sorted_vals, matched_vals).reshape(channel.shape)
        
        return result
    
    def _edge_preserving_smoothing(self, image: np.ndarray) -> np.ndarray:
        """
        边缘保持平滑（双边滤波简化版）
        减少噪声但保留缺陷边缘
        """
        result = np.zeros_like(image)
        h, w = image.shape[:2]
        window_size = 3
        sigma_color = 0.1
        sigma_space = 2.0
        
        space_weights = np.zeros((window_size, window_size), dtype=np.float32)
        half = window_size // 2
        for i in range(window_size):
            for j in range(window_size):
                dist = np.sqrt((i - half)**2 + (j - half)**2)
                space_weights[i, j] = np.exp(-dist**2 / (2 * sigma_space**2))
        
        padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode='reflect')
        
        for c in range(3):
            for i in range(h):
                for j in range(w):
                    center_val = padded[i + half, j + half, c]
                    region = padded[i:i+window_size, j:j+window_size, c]
                    
                    color_diff = np.abs(region - center_val)
                    color_weights = np.exp(-color_diff**2 / (2 * sigma_color**2))
                    
                    weights = space_weights * color_weights
                    weights /= weights.sum()
                    
                    result[i, j, c] = np.sum(region * weights)
        
        return result
    
    def preprocess_with_robustness(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        鲁棒性预处理 - 结合光照归一化和域适应
        返回：(预处理后的图像, 处理元数据)
        """
        orig_brightness = float(np.mean(image))
        orig_contrast = float(np.std(image))
        
        normalized = self.normalize_illumination(image)
        
        adapted = self.domain_adaptation_preprocess(normalized)
        
        preprocessed = self.preprocess(adapted)
        
        metadata = {
            'original_brightness': orig_brightness,
            'original_contrast': orig_contrast,
            'processed_brightness': float(np.mean(adapted)),
            'processed_contrast': float(np.std(adapted)),
            'illumination_correction': abs(orig_brightness - 128.0) > 30,
            'contrast_enhancement': abs(orig_contrast - 60.0) > 15
        }
        
        return preprocessed, metadata
    
    def predict_with_augmentation(self, image: np.ndarray, 
                                   predict_func) -> Tuple[str, float, List[float]]:
        """
        使用数据增强进行集成预测
        对多个增强变体进行预测，取平均置信度
        """
        augmented_images = self.augment_image(image)
        
        predictions = []
        confidences = []
        
        for aug_img in augmented_images:
            preprocessed, _ = self.preprocess_with_robustness(aug_img)
            pred, conf = predict_func(preprocessed)
            predictions.append(pred)
            confidences.append(conf)
        
        pred_counter = Counter(predictions)
        final_pred = pred_counter.most_common(1)[0][0]
        
        weights = [1.0 if p == final_pred else 0.5 for p in predictions]
        total_weight = sum(weights)
        avg_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        
        return final_pred, avg_confidence, confidences
