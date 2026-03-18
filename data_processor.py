import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import Optional, Tuple, List

class GestureDataProcessor:
    def __init__(self, window_size: int = 3, eps: float = 0.09, min_samples: int = 2):
        self.window_size = window_size
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.data_buffer: List[np.ndarray] = []
        
    @property
    def feature_dim(self) -> int:
        """
        返回特征维度 (硬编码27维，兼容下游Transformer)
        结构: 质心(3) + 空间分布(3) + 速度(1) + 极差(1) + 点数(1) + 体积(1) + 偏角(1) + Z轴运动(1) + 基础统计(16)
        """
        return 27

    def preprocess_frame(self, frame_data: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """实时数据流入口"""
        if frame_data is None or len(frame_data) == 0:
            return None, None

        filtered_data = self._apply_dbscan(frame_data)
        if len(filtered_data) == 0:
            return None, None

        frame_features = self._compute_frame_features(filtered_data)
        return frame_features, filtered_data

    def _apply_dbscan(self, data: np.ndarray) -> np.ndarray:
        """DBSCAN去噪 & 多普勒一致性校验"""
        if len(data) < self.min_samples:
            return np.array([])

        # 仅取 XYZ 做空间聚类
        clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(data[:, :3])
        
        # 过滤噪声点(-1)
        valid_mask = clusters != -1
        if not np.any(valid_mask):
            return np.array([])

        # 取点数最多的主聚类簇
        largest_cluster = np.bincount(clusters[valid_mask]).argmax()
        filtered_data = data[clusters == largest_cluster]

        # Doppler 突变点过滤 (阈值 0.5)
        if len(filtered_data) >= 2:
            doppler_std = np.std(filtered_data[:, 3])
            if doppler_std > 0.5:
                doppler_mean = np.mean(filtered_data[:, 3])
                filtered_data = filtered_data[np.abs(filtered_data[:, 3] - doppler_mean) < doppler_std]

        return filtered_data

    def _compute_frame_features(self, filtered_data: np.ndarray) -> np.ndarray:
        """提取单帧 27维特征"""
        # 1. 提取基础统计量
        mean_feats = np.mean(filtered_data, axis=0)  
        std_feats = np.std(filtered_data, axis=0)    
        max_feats = np.max(filtered_data, axis=0)    
        min_feats = np.min(filtered_data, axis=0)    

        # 2. 几何与运动特征计算
        points_count = float(len(filtered_data))
        # 用 ptp (peak-to-peak) 计算极差和体积，比 max-min 更快
        velocity_range = np.ptp(filtered_data[:, 3])
        volume = np.prod(np.ptp(filtered_data[:, :3], axis=0)) 
        
        movement_direction = np.arctan2(mean_feats[1], mean_feats[0])
        vertical_movement = mean_feats[2]

        # 3. 拼接27维特征 (注意: 质心即mean[:3]，分布即std[:3]，这里保留重复以兼容旧模型维度)
        features = np.concatenate([
            mean_feats[:3],      # center_of_mass
            std_feats[:3],       # spread
            [mean_feats[3]],     # velocity
            [velocity_range],
            [points_count, volume, movement_direction, vertical_movement],
            mean_feats, std_feats, max_feats, min_feats
        ])
        
        return features

    def update_buffer(self, frame_features: np.ndarray) -> None:
        """维护固定长度的时间滑窗"""
        if frame_features is not None:
            self.data_buffer.append(frame_features)
            if len(self.data_buffer) > self.window_size:
                self.data_buffer.pop(0)

    def get_gesture_window(self) -> Optional[np.ndarray]:
        """获取并标准化滑窗数据"""
        if len(self.data_buffer) < self.window_size:
            return None

        window_data = np.array(self.data_buffer)

        # 核心逻辑：对当前短窗口进行局部 fit_transform
        # 目的：消除绝对幅度差异，仅保留 3帧内数据的相对变化趋势，提升手势形态的识别鲁棒性
        window_data = self.scaler.fit_transform(window_data)

        # reshape -> (batch=1, seq_len, features)
        return window_data.reshape(1, self.window_size, -1).astype(np.float32)

    def clear_buffer(self) -> None:
        self.data_buffer.clear()