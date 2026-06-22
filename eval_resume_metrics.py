import os
import re
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from model import GestureTransformer

DATA_DIR = 'data'
MODEL_PATH = 'best_model.pth'

my_idx = ['x', 'y', 'z', 'Doppler']
sample_size = 3

classes = [
    ('clockwise', 0),
    ('counterclockwise', 1),
    ('swipe', 2),
    ('up_down_swipe', 3),
]

def list_csvs(folder):
    return sorted(glob.glob(os.path.join(DATA_DIR, folder, '*.csv')))

def load_df(path):
    try:
        return pd.read_csv(path)
    except Exception:
        # tolerate malformed old exports
        return pd.read_csv(path, engine='python')

def process_samples(df):
    valid = [c for c in my_idx if c in df.columns]
    if len(valid) < 4:
        # try case-insensitive map
        cols = {c.lower(): c for c in df.columns}
        mapped = []
        for c in my_idx:
            if c.lower() in cols:
                mapped.append(cols[c.lower()])
        if len(mapped) == 4:
            df = df[mapped]
            df.columns = my_idx
        else:
            return []
    else:
        df = df[my_idx]

    n = len(df) // sample_size
    if n <= 0:
        return []
    vals = df.values
    out = []
    for i in range(n):
        out.append(vals[i*sample_size:(i+1)*sample_size].tolist())
    return out

x_data = []
y_data = []
class_frame_counts = {}

for folder, label in classes:
    files = list_csvs(folder)
    frames = 0
    class_samples = 0
    for f in files:
        df = load_df(f)
        frames += len(df)
        samples = process_samples(df)
        class_samples += len(samples)
        x_data.extend(samples)
        y_data.extend([label] * len(samples))
    class_frame_counts[folder] = {'files': len(files), 'frames': frames, 'samples': class_samples}

x_data = np.asarray(x_data, dtype='float32')
y_data = np.asarray(y_data, dtype='int64')

if len(x_data) == 0:
    raise RuntimeError('No valid samples built from data files')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

x_test_t = torch.FloatTensor(x_test)
y_test_t = torch.LongTensor(y_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureTransformer(num_classes=4).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
if isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint.state_dict())

model.eval()
with torch.no_grad():
    outputs = model(x_test_t.to(device))
    pred = outputs.argmax(dim=1)
    total = y_test_t.numel()
    correct = (pred.cpu() == y_test_t).sum().item()
    acc = 100.0 * correct / total

print('METRIC_TEST_ACCURACY=%.2f' % acc)
print('METRIC_TEST_TOTAL=%d' % total)
print('METRIC_TEST_CORRECT=%d' % correct)
print('METRIC_DATASET_TOTAL_SAMPLES=%d' % len(x_data))
for k,v in class_frame_counts.items():
    print(f'METRIC_{k.upper()}_FILES={v["files"]}')
    print(f'METRIC_{k.upper()}_FRAMES={v["frames"]}')
    print(f'METRIC_{k.upper()}_SAMPLES={v["samples"]}')
