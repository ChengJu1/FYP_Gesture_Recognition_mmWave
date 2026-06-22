import importlib.util
import torch

spec = importlib.util.spec_from_file_location('mold', 'model_b00a7dc.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

model = m.GestureCNN()
state_dict = torch.load('best_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(state_dict)
model.eval()

x_test = m.x_test
y_test = m.y_test

with torch.no_grad():
    out = model(x_test)
    pred = out.argmax(1)


total = y_test.size(0)
correct = (pred == y_test).sum().item()
acc = 100.0 * correct / total

print(f'METRIC_TEST_ACCURACY={acc:.2f}')
print(f'METRIC_TEST_TOTAL={total}')
print(f'METRIC_TEST_CORRECT={correct}')
print(f'METRIC_DATASET_TOTAL_SAMPLES={m.x_data.shape[0]}')
print(f'METRIC_CLOCKWISE_FRAMES={len(m.clockwise_train)}')
print(f'METRIC_COUNTERCLOCKWISE_FRAMES={len(m.counterclockwise_train)}')
print(f'METRIC_SWIPE_FRAMES={len(m.swipe_train)}')
print(f'METRIC_UP_DOWN_SWIPE_FRAMES={len(m.up_down_swipe_train)}')
