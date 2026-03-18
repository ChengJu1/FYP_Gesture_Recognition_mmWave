import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 模型定义区 (Model Architecture)
# ==========================================

class GestureTransformer(nn.Module):
    """
    手势识别 Transformer 模型 (基础版 - Version 1.0)
    注: 当前版本仅使用了纯 Transformer Encoder，尚未引入位置信息。
    """
    def __init__(self, input_size: int = 4, num_classes: int = 4, 
                 d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        
        # 1. 特征映射层：将原始的4维特征映射到模型的隐藏维度 (d_model)
        self.embed = nn.Linear(input_size, d_model)
        
        # 2. Transformer 编码器层 
        # (开启 batch_first=True 简化维度操作，输入形状为 batch, seq, feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=64,
            batch_first=True  
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 分类头：用于输出最终的类别得分
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: (batch_size, seq_len, input_size)
        返回:
            logits: (batch_size, num_classes)
        """
        # 特征升维 -> (batch_size, seq_len, d_model)
        x = self.embed(x)            
        
        # 提取全局特征 -> (batch_size, seq_len, d_model)
        # [面试埋点]: 这里直接输入了 Transformer，没有加入序列的时序顺序信息
        x = self.transformer(x)      
        
        # 全局平均池化，压缩时间维度 -> (batch_size, d_model)
        x = x.mean(dim=1)            
        
        # 分类输出 -> (batch_size, num_classes)
        return self.fc(x)            


# ==========================================
# 2. 数据处理区 (Data Pipeline)
# ==========================================

class GestureDataset(Dataset):
    """标准的 PyTorch 数据集封装"""
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        # 优化点：在初始化时一次性转换为 Tensor，避免在 __getitem__ 中反复转换消耗 CPU
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


def load_and_process_category(category_name: str, label_idx: int, num_files: int = 4, 
                              sample_size: int = 3, target_cols: list = ["x", "y", "z", "Doppler"]):
    """
    通用数据加载与切片函数 (消除了原代码中的冗余复制粘贴)
    """
    # 动态生成文件路径
    file_paths = [f"data/{category_name}/{category_name}_{i}.csv" for i in range(1, num_files + 1)]
    try:
        # 批量读取并合并
        df = pd.concat(map(pd.read_csv, file_paths), ignore_index=True)
    except FileNotFoundError as e:
        print(f"警告: 找不到数据文件，请检查路径 -> {e}")
        return [], []

    samples = []
    # 使用无重叠的滑动窗口提取序列 (这里的 sample_size 实际上代表 sequence_length)
    for i in range(0, len(df) - sample_size + 1, sample_size):
        window_data = df.iloc[i:i + sample_size][target_cols].values
        samples.append(window_data)
        
    labels = [label_idx] * len(samples)
    return samples, labels


def prepare_data(sample_size: int = 3, batch_size: int = 8):
    """
    构建数据加载器的入口工厂函数
    """
    # 定义类别映射字典
    categories = {
        "clockwise": 0,
        "counterclockwise": 1,
        "swipe": 2,
        "up_down_swipe": 3
    }
    
    all_samples, all_labels = [], []
    
    # 遍历加载所有类别数据
    for cat_name, label in categories.items():
        # 注意: 如果 counterclockwise 的原文件名为 counterwise_*.csv，请在此处单独处理路径映射
        samples, labels = load_and_process_category(cat_name, label, num_files=4, sample_size=sample_size)
        all_samples.extend(samples)
        all_labels.extend(labels)

    # 转换为 NumPy 数组
    x_data = np.array(all_samples).astype("float32")
    y_data = np.array(all_labels).astype("int64")
    
    print(f"合并后的数据形状 | X: {x_data.shape}, Y: {y_data.shape}")

    # 划分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # 创建 DataLoader
    train_loader = DataLoader(GestureDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GestureDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# ==========================================
# 3. 主执行入口 (Main Execution)
# ==========================================

if __name__ == "__main__":
    # 参数配置
    SAMPLE_SIZE = 3      # 序列长度 (Sequence Length)
    BATCH_SIZE = 8       # 批次大小
    INPUT_FEATURES = 4   # 对应 ["x", "y", "z", "Doppler"]
    NUM_CLASSES = 4      # 4种手势分类

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用计算设备: {device}")

    # 1. 准备数据加载器
    train_loader, test_loader = prepare_data(sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE)

    # 2. 实例化模型
    model = GestureTransformer(input_size=INPUT_FEATURES, num_classes=NUM_CLASSES).to(device)

    # 3. 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n模型结构加载完毕，准备进行训练。")