import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model import GestureCNN
from collections import OrderedDict


plt.ion()
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureCNN()  # 首先创建模型实例
checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
if isinstance(checkpoint, GestureCNN):
    # 如果保存的是整个模型，获取其状态字典
    model.load_state_dict(checkpoint.state_dict())
else:
    # 如果是状态字典则直接加载
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 数据加载和处理
clockwise_train = pd.concat(map(pd.read_csv, ["data/clockwise/clockwise_1.csv.csv",
                                              "data/clockwise/clockwise_2.csv.csv",
                                              "data/clockwise/clockwise_3.csv.csv",
                                              "data/clockwise/clockwise_4.csv.csv"]),
                            ignore_index=True)
counterclockwise_train = pd.concat(map(pd.read_csv, ["data/counterclockwise/counterwise_1.csv",
                                                     "data/counterclockwise/counterwise_2.csv",
                                                     "data/counterclockwise/counterwise_3.csv",
                                                     "data/counterclockwise/counterwise_4.csv"]),
                                   ignore_index=True)
swipe_train = pd.concat(map(pd.read_csv, ["data/swipe/swipe_1.csv",
                                          "data/swipe/swipe_2.csv",
                                          "data/swipe/swipe_3.csv",
                                          "data/swipe/swipe_4.csv"]),
                        ignore_index=True)
up_down_swipe_train = pd.concat(map(pd.read_csv, ["data/up_down_swipe/up_down_swipe_1.csv",
                                                   "data/up_down_swipe/up_down_swipe_2.csv",
                                                  "data/up_down_swipe/up_down_swipe_3.csv",
                                                  "data/up_down_swipe/up_down_swipe_4.csv"]),
                                ignore_index=True)

my_idx = ["x", "y", "z", "Doppler"]
idx_size = len(my_idx)
sample_size = 3

encoded_labels = ["Clockwise", "Counter Clockwise", "Swipe", "Swipe Up and Down"]
encoded_labels = np.array(encoded_labels)

def process_gesture_data(gesture_data, sample_size):
    samples = []
    for i in range(int(len(gesture_data) / sample_size)):
        samples.append(gesture_data[sample_size * i:sample_size * i + sample_size][my_idx].values.tolist())
    return samples

# 处理所有手势数据
x_data = []
x_data.extend(process_gesture_data(clockwise_train, sample_size))
x_data.extend(process_gesture_data(counterclockwise_train, sample_size))
x_data.extend(process_gesture_data(swipe_train, sample_size))
x_data.extend(process_gesture_data(up_down_swipe_train, sample_size))

x_data = np.asarray(x_data).reshape(-1, sample_size, idx_size).astype("float32")

def create_labels(length, label_value):
    return np.full(int(length / sample_size), label_value)

y_data = np.concatenate((
    create_labels(len(clockwise_train), 0),
    create_labels(len(counterclockwise_train), 1),
    create_labels(len(swipe_train), 2),
    create_labels(len(up_down_swipe_train), 3)
))

# 数据集分割
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 转换为PyTorch张量
x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
x_test = torch.FloatTensor(x_test).to(device)
y_test = torch.LongTensor(y_test).to(device)

# 模型评估
def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)
        total = y_test.size(0)
        correct = (predicted == y_test).sum().item()
        accuracy = 100 * correct / total
        return predicted, accuracy

predicted, accuracy = evaluate_model(model, x_test, y_test)

# 生成混淆矩阵
cf_matrix = confusion_matrix(y_test.cpu(), predicted.cpu())

# ========= 关键修改：按“行”归一化 (每行加和为1) =========
cf_matrix_normalized = cf_matrix.astype(float) / cf_matrix.sum(axis=1, keepdims=True)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cf_matrix_normalized, annot=True, fmt='.2%', cmap='Blues')
plt.title('Confusion Matrix (Row-normalized)\nPredicted vs Actual Gestures')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.xticks(np.arange(len(encoded_labels)) + 0.5, encoded_labels, rotation=45)
plt.yticks(np.arange(len(encoded_labels)) + 0.5, encoded_labels, rotation=45)
plt.tight_layout()
plt.show()

print(f"Test Accuracy: {accuracy:.2f}%")