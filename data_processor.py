import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


class GestureDataProcessor:
    def __init__(self, window_size=3, eps=0.09, min_samples=2):
        """
        初始化手势数据处理器

        参数:
            window_size (int): 时间窗口大小
            eps (float): DBSCAN的邻域半径参数
            min_samples (int): DBSCAN的最小样本数
        """
        self.window_size = window_size
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.data_buffer = []
        self.filtered_buffer = []

    def preprocess_frame(self, frame_data):
        """
        处理单帧数据

        参数:
            frame_data (np.ndarray): 形状为(N, 4)的帧数据，包含x,y,z坐标和多普勒速度

        返回:
            np.ndarray: 处理后的特征向量，如果数据无效则返回None
        """
        if len(frame_data) == 0:
            return None

        # DBSCAN空间聚类
        filtered_data = self._apply_dbscan(frame_data)
        if len(filtered_data) == 0:
            return None

        # 计算帧特征
        frame_features = self._compute_frame_features(filtered_data)
        return frame_features, filtered_data

    def _apply_dbscan(self, data):
        """
        应用DBSCAN聚类算法并进行噪声过滤

        参数:
            data (np.ndarray): 原始点云数据

        返回:
            np.ndarray: 过滤后的数据
        """
        if len(data) < self.min_samples:
            return np.array([])

        # 空间聚类
        spatial_data = data[:, :3]
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = dbscan.fit_predict(spatial_data)

        # 获取最大簇
        if np.all(clusters == -1):
            return np.array([])

        largest_cluster = max(set(clusters), key=list(clusters).count)
        filtered_data = data[clusters == largest_cluster]

        # 添加速度一致性检查
        if len(filtered_data) >= 2:
            doppler_std = np.std(filtered_data[:, 3])
            if doppler_std > 0.5:  # 如果速度变化太大，可能是噪声
                filtered_data = filtered_data[abs(filtered_data[:, 3] - np.mean(filtered_data[:, 3])) < doppler_std]

        return filtered_data

    def _compute_frame_features(self, filtered_data):
        """
        计算单帧的特征

        参数:
            filtered_data (np.ndarray): 过滤后的点云数据

        返回:
            np.ndarray: 计算得到的特征向量
        """
        # 基础统计特征
        mean_features = np.mean(filtered_data, axis=0)  # [x_mean, y_mean, z_mean, v_mean]
        std_features = np.std(filtered_data, axis=0)  # [x_std, y_std, z_std, v_std]
        max_features = np.max(filtered_data, axis=0)  # [x_max, y_max, z_max, v_max]
        min_features = np.min(filtered_data, axis=0)  # [x_min, y_min, z_min, v_min]

        # 计算空间特征
        center_of_mass = mean_features[:3]  # 质心位置 [x, y, z]
        spread = std_features[:3]  # 空间分布 [x_spread, y_spread, z_spread]
        velocity = mean_features[3]  # 平均速度
        velocity_range = max_features[3] - min_features[3]  # 速度范围

        # 计算点云几何特征
        points_count = len(filtered_data)  # 点数量
        volume = np.prod(max_features[:3] - min_features[:3])  # 包围盒体积

        # 计算运动特征
        movement_direction = np.arctan2(mean_features[1], mean_features[0])  # 运动方向
        vertical_movement = mean_features[2]  # 垂直运动

        # 组合所有特征
        frame_features = np.concatenate([
            center_of_mass,  # 3维: 质心位置
            spread,  # 3维: 空间分布
            [velocity],  # 1维: 平均速度
            [velocity_range],  # 1维: 速度范围
            [points_count],  # 1维: 点数量
            [volume],  # 1维: 体积
            [movement_direction],  # 1维: 运动方向
            [vertical_movement],  # 1维: 垂直运动
            mean_features,  # 4维: 平均特征
            std_features,  # 4维: 标准差特征
            max_features,  # 4维: 最大值特征
            min_features,  # 4维: 最小值特征
        ])

        return frame_features

    def update_buffer(self, frame_features):
        """
        更新数据缓冲区

        参数:
            frame_features (np.ndarray): 单帧的特征向量
        """
        if frame_features is not None:
            self.data_buffer.append(frame_features)
            if len(self.data_buffer) > self.window_size:
                self.data_buffer.pop(0)

    def get_gesture_window(self):
        """
        获取用于手势识别的时间窗口数据

        返回:
            np.ndarray: 形状为(1, window_size, features)的处理后的窗口数据，
                      如果数据不足则返回None
        """
        if len(self.data_buffer) < self.window_size:
            return None

        # 构建特征窗口
        window_data = np.array(self.data_buffer)

        # 标准化
        window_data = self.scaler.fit_transform(window_data)

        # 重塑为模型输入格式 (1, window_size, features)
        window_data = window_data.reshape(1, self.window_size, -1)
        return window_data.astype(np.float32)

    def clear_buffer(self):
        """清空数据缓冲区"""
        self.data_buffer = []
        self.filtered_buffer = []

    def process_raw_data(self, gesture_data, my_idx=["x", "y", "z", "Doppler"]):
        """
        处理原始CSV数据

        参数:
            gesture_data (pd.DataFrame): 包含手势数据的DataFrame
            my_idx (list): 数据列名列表

        返回:
            np.ndarray: 处理后的特征数据
        """
        processed_samples = []

        # 按时间窗口处理数据
        for i in range(0, len(gesture_data) - self.window_size + 1, self.window_size):
            window = gesture_data[i:i + self.window_size][my_idx].values

            # 应用DBSCAN
            filtered_window = self._apply_dbscan(window)
            if len(filtered_window) >= self.window_size:
                # 提取特征
                window_features = self._compute_frame_features(filtered_window)
                processed_samples.append(window_features)

        return np.array(processed_samples) if processed_samples else np.array([])

def get_feature_dim(self):
    """
    获取特征维度的详细计算
    """
    # 1. 空间位置特征 (3维)
    center_of_mass_dim = 3  # [x_center, y_center, z_center]

    # 2. 空间分布特征 (3维)
    spread_dim = 3  # [x_spread, y_spread, z_spread]

    # 3. 速度特征 (2维)
    velocity_dim = 1  # 平均速度
    velocity_range_dim = 1  # 速度范围(最大速度-最小速度)

    # 4. 形状特征 (2维)
    points_count_dim = 1  # 点云中的点数量
    volume_dim = 1  # 点云的包围盒体积

    # 5. 运动特征 (2维)
    movement_direction_dim = 1  # 运动方向(水平面上的角度)
    vertical_movement_dim = 1  # 垂直方向的运动

    # 6. 统计特征 (16维)
    mean_features_dim = 4  # x,y,z,velocity 的平均值
    std_features_dim = 4  # x,y,z,velocity 的标准差
    max_features_dim = 4  # x,y,z,velocity 的最大值
    min_features_dim = 4  # x,y,z,velocity 的最小值

    # 总维度计算
    total_dim = (
            center_of_mass_dim +  # 3
            spread_dim +  # 3
            velocity_dim +  # 1
            velocity_range_dim +  # 1
            points_count_dim +  # 1
            volume_dim +  # 1
            movement_direction_dim +  # 1
            vertical_movement_dim +  # 1
            mean_features_dim +  # 4
            std_features_dim +  # 4
            max_features_dim +  # 4
            min_features_dim  # 4
    )  # = 27维

    return total_dim