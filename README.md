# 毫米波雷达手势识别系统（FYP_Gesture_Recognition_mmWave）

基于 **TI IWR6843AOP 毫米波雷达** 的非接触手势识别与人机交互系统。系统实时采集雷达点云，经 **DBSCAN 去噪 → 时序特征提取 → Transformer 分类**，识别 4 类空中手势，并将识别结果映射为 **macOS 系统媒体控制**（音量、播放/暂停、静音）。

> 本科毕业设计（Final Year Project）。

---

## 功能特性

- **非接触交互**：无需穿戴或触摸，在雷达上方挥动手势即可控制设备。
- **点云去噪**：使用 DBSCAN 空间聚类提取主目标簇，并结合多普勒（Doppler）一致性校验滤除噪点。
- **时序建模**：以 3 帧为滑动窗口构建时序样本，使用 Transformer 编码器进行手势分类。
- **实时推理 + 系统联动**：识别置信度超过阈值后触发 macOS 媒体控制，并设有冷却时间防误触。
- **一体化数据采集**：内置数据采集模式，可直接录制并保存训练用 CSV 数据。
- **可视化评估**：提供混淆矩阵与准确率评估脚本。

---

## 支持的手势与控制映射

| 标签 | 手势 | 类别索引 | macOS 动作 |
|------|------|:------:|------------|
| `clockwise` | 顺时针旋转 | 0 | 音量调高（+10） |
| `counterclockwise` | 逆时针旋转 | 1 | 音量调低（−10） |
| `swipe` | 滑动 | 2 | 播放 / 暂停 |
| `up_down_swipe` | 上下滑动 | 3 | 静音 / 取消静音 |

---

## 系统工作流程

```
毫米波雷达 (IWR6843AOP)
        │  双串口 (CLI 115200 / Data 921600)，pymmWave 驱动
        ▼
原始点云 [x, y, z, Doppler]
        │  DBSCAN 空间聚类，取最大簇；Doppler 突变点过滤
        ▼
去噪后点云 → 单帧特征
        │  3 帧滑动窗口 → 时序张量 (1, 3, 特征维)
        ▼
Transformer 编码器 → 全局平均池化 → 全连接分类头
        │  Softmax 置信度 > 阈值(70%)
        ▼
手势类别 → macOS 媒体控制（含 1.5s 冷却防误触）
```

---

## 硬件要求

- **毫米波雷达**：TI IWR6843AOP（AOP 封装，60GHz）。
- **连接**：两路串口
  - CLI / Config 端口：波特率 `115200`
  - Data 端口：波特率 `921600`
- **运行平台**：macOS（实时控制部分依赖 `osascript` / AppleScript；其余训练、评估脚本跨平台可用）。

> 雷达配置文件位于 `mmwave_config/`，默认使用 `xwr68xx_AOP_config_10FPS_maxRange_30cm.cfg`（10 FPS、最大量程约 30cm）。

---

## 目录结构

```
FYP_Gesture_Recognition_mmWave/
├── data_parse.py          # 主程序：硬件驱动 + 实时推理 + macOS 控制 + 数据采集
├── data_processor.py      # GestureDataProcessor：DBSCAN 去噪 + 27 维特征提取（流式缓冲）
├── model.py               # 重构版：GestureTransformer 模型 + 数据集封装 + prepare_data
├── train.py               # 训练脚本（Adam + 交叉熵 + 早停，保存最优权重）
├── confusion matrix.py    # 混淆矩阵可视化评估
├── eval_resume_metrics.py # 准确率 / 数据集统计指标脚本
├── eval_checkpoint_b00a7dc.py  # 针对历史 checkpoint 的评估脚本
├── model_b00a7dc.py       # 历史版本模型 + 数据加载（旧接口）
├── legacy_model.py        # 早期模型实现（保留备查）
├── best_model.pth         # 训练得到的最优模型权重
├── checkpoints/           # 训练过程 checkpoint
├── mmwave_config/         # 雷达配置文件 (.cfg)
└── data/                  # 手势数据集（CSV，按类别分文件夹）
    ├── clockwise/
    ├── counterclockwise/
    ├── swipe/
    └── up_down_swipe/
```

---

## 环境依赖

- Python 3.10+
- PyTorch
- NumPy、Pandas
- scikit-learn（DBSCAN、数据划分、标准化）
- Matplotlib、Seaborn（可视化）
- tqdm（训练进度条）
- [pymmWave](https://pypi.org/project/pymmWave/)（IWR6843AOP 雷达驱动，仅实时推理 / 采集时需要）

安装示例：

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm pymmwave
```

---

## 使用方法

### 1. 采集训练数据

连接好雷达后运行主程序，选择采集模式：

```bash
python data_parse.py
# 选择 [2] 采集训练数据，输入手势名称（如 clockwise）
```

采集满设定帧数后会自动保存为 `data/<手势名>/<手势名>_<序号>.csv`，列为 `x, y, z, Doppler`。

### 2. 训练模型

```bash
python train.py
```

- 优化器：Adam（学习率 `1e-4`）
- 损失：交叉熵（CrossEntropyLoss）
- 训练轮数：最多 400 epoch，启用早停（patience=30）
- 训练 / 验证按 8:2 划分（`random_state=42`）
- 验证准确率提升时保存为 `best_model.pth`

### 3. 评估模型

```bash
# 混淆矩阵（行归一化，Seaborn 热力图）
python "confusion matrix.py"

# 准确率与数据集统计指标
python eval_resume_metrics.py
```

### 4. 实时手势控制（macOS）

```bash
python data_parse.py
# 选择 [1] 实时手势控制
```

识别置信度超过 `70%` 即触发对应媒体控制，触发后进入 1.5 秒冷却以防误触。

---

## 模型说明

核心模型为 **Transformer 编码器**（在部分历史文件中类名为 `GestureCNN`，实际并非卷积网络）：

- 输入：时序张量 `(batch, seq_len=3, input_size=4)`，特征为 `[x, y, z, Doppler]`
- 结构：线性嵌入（`input_size → d_model=32`）→ `TransformerEncoder`（`nhead=4`、`dim_feedforward=64`、`num_layers=2`）→ 时间维全局平均池化 → 全连接分类头（4 类）

`model.py` 中的 `GestureTransformer` 为重构后的整洁实现，并在注释中标注了「未引入位置编码」这一可改进点。

---

## 数据处理说明

仓库内包含两套特征方案：

1. **基础 4 维方案（当前训练 / 实时推理所用）**
   对去噪后点云逐帧取均值，得到 `[x, y, z, Doppler]` 4 维特征，按 3 帧窗口堆叠为时序样本。

2. **27 维时空特征方案（`data_processor.py` 中的 `GestureDataProcessor`）**
   面向流式实时处理，单帧提取 27 维特征：质心(3) + 空间分布(3) + 速度(1) + 速度极差(1) + 点数(1) + 体积(1) + 运动偏角(1) + Z 轴运动(1) + 基础统计量(16)，并对滑窗做局部标准化以增强形态鲁棒性。

去噪环节统一采用 DBSCAN（`eps=0.09`、`min_samples=2`）取最大簇，再以多普勒标准差阈值 `0.5` 过滤突变点。

---

## 已知问题与注意事项

- **接口不一致**：`train.py` 从 `model` 导入 `GestureCNN, train_loader, test_loader`，而重构后的 `model.py` 提供的是 `GestureTransformer` 与 `prepare_data()`（无模块级 loader）。旧接口保留在 `legacy_model.py` / `model_b00a7dc.py` 中。训练前请确认导入来源一致（建议统一改用 `model.py` 的 `prepare_data()`）。
- **历史文件**：`legacy_model.py`、`model_b00a7dc.py` 含有早期实现，部分中文注释为 GBK 编码、可能显示乱码，仅供备查。
- **`test.py`**：引用了已不存在的 `SimpleCNN`，为历史遗留桩文件，可忽略。
- **平台限制**：实时媒体控制依赖 macOS 的 AppleScript；在其他系统上需自行替换 `MacOSMediaController` 中的控制逻辑。
- **硬编码路径**：部分评估脚本中的模型路径（如 `d:\Edge\best_model.pth`）为本地绝对路径，运行前请按需修改。
