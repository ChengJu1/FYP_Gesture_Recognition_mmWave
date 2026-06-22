import torch
from model import GestureTransformer

# 冒烟测试：构造一批随机时序样本，验证模型前向传播与输出维度
# 输入形状: (batch_size, seq_len, input_size) = (32, 3, 4)
x = torch.randn(32, 3, 4)
model = GestureTransformer(num_classes=4)
output = model(x)

print(output)
print("output shape:", output.shape)  # 期望: (32, 4)
assert output.shape == (32, 4), "输出维度应为 (batch_size, num_classes)"
print("Smoke test passed.")
