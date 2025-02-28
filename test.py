import torch
from model import SimpleCNN

x = torch.randn(32, 3, 224, 224)
model = SimpleCNN(num_class=4)
output = model(x)
print(output)
print(output.shape)
