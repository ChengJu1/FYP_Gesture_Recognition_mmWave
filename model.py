import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 加载数据
clockwise_train = pd.concat(map(pd.read_csv, ["data/clockwise/clockwise_1.csv", "data/clockwise/clockwise_2.csv", "data/clockwise/clockwise_3.csv", "data/clockwise/clockwise_4.csv"]), ignore_index=True)
counterclockwise_train = pd.concat(map(pd.read_csv, ["data/counterclockwise/counterwise_1.csv", "data/counterclockwise/counterwise_2.csv", "data/counterclockwise/counterwise_3.csv","data/counterclockwise/counterwise_4.csv"]), ignore_index=True)
swipe_train = pd.concat(map(pd.read_csv, ["data/swipe/swipe_1.csv", "data/swipe/swipe_2.csv", "data/swipe/swipe_3.csv", "data/swipe/swipe_4.csv"]), ignore_index=True)
up_down_swipe_train = pd.concat(map(pd.read_csv, ["data/up_down_swipe/up_down_swipe_1.csv", "data/up_down_swipe/up_down_swipe_2.csv", "data/up_down_swipe/up_down_swipe_3.csv", "data/up_down_swipe/up_down_swipe_4.csv"]), ignore_index=True)

x_data = []
my_idx = ["x", "y", "z", "Doppler"]
idx_size = len(my_idx)
sample_size = 3 # 卷积核大小为3

# 处理手势数据
def process_gesture_data(gesture_data):
    gesture_samples = []
    for i in range(0, len(gesture_data) - sample_size + 1, sample_size):
        gesture_samples.append(gesture_data[i:i + sample_size][my_idx].values.tolist())
    return gesture_samples

# 处理每个手势的数据
clockwise_samples = process_gesture_data(clockwise_train)
counterclockwise_samples = process_gesture_data(counterclockwise_train)
swipe_samples = process_gesture_data(swipe_train)
up_down_swipe_samples = process_gesture_data(up_down_swipe_train)

x_data =clockwise_samples + counterclockwise_samples + swipe_samples + up_down_swipe_samples
x_data = np.asarray(x_data).reshape(-1, sample_size, idx_size).astype("float32")
print(x_data.shape)

clockwise_y = np.full(len(clockwise_samples), 0)
counterclockwise_y = np.full(len(counterclockwise_samples), 1)
swipe_y = np.full(len(swipe_samples), 2)
up_down_swipe_y = np.full(len(up_down_swipe_samples), 3)

y_data = np.concatenate([clockwise_y, counterclockwise_y ,swipe_y, up_down_swipe_y])
print(x_data.shape)
print(y_data.shape)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # 数据应为NumPy数组，形状为 (样本数, 时间步, 特征数)
        self.labels = labels  # 标签应为NumPy数组

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        # 将数据转换为Tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return sample, label

    # 创建训练集和验证集的数据集对象
train_dataset = GestureDataset(x_train, y_train)
test_dataset = GestureDataset(x_val, y_val)

# 创建数据加载器
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class GestureCNN(nn.Module):
    def __init__(self, input_size=4, num_classes=4, d_model=32, nhead=4):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)  # (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, features)
        x = self.transformer(x)
        x = x.mean(dim=0)  # 全局平均池化
        return self.fc(x)
# 创建模型、损失函数和优化器
model = GestureCNN()
criterion = nn.CrossEntropyLoss()
