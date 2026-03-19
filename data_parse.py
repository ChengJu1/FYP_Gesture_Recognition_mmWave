import os
import asyncio
import subprocess
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 引入自定义模块
from pymmWave.utils import load_cfg_file
from pymmWave.sensor import Sensor
from pymmWave.IWR6843AOP import IWR6843AOP
from model import GestureCNN

# ==========================================
# 0. 全局配置项 (Configuration)
# ==========================================
class Config:
    # 硬件端口配置 (根据实际环境修改)
    CLI_PORT = '/dev/tty.SLAB_USBtoUART'
    DATA_PORT = '/dev/tty.SLAB_USBtoUART4'
    CFG_FILE_PATH = "./mmwave_config/xwr68xx_AOP_config_10FPS_maxRange_30cm.cfg"
    
    # 模型配置
    MODEL_PATH = 'best_model.pth'
    CONFIDENCE_THRESHOLD = 70.0  # 触发动作的置信度阈值
    
    # 数据采集配置
    DATA_DIR = "./data/"
    MIN_CAPTURE_SAMPLES = 500
    SEQ_LENGTH = 3  # 推理所需的序列长度


# ==========================================
# 1. macOS 系统控制 (System Controller)
# ==========================================
class MacOSMediaController:
    """封装对 MacOS 的系统级音频与媒体控制"""
    
    GESTURE_MAPPING = {
        "Clockwise": "volumeup",
        "Counter Clockwise": "volumedown",
        "Swipe": "playpause",
        "Swipe Up and Down": "mute"
    }

    @staticmethod
    def _run_apple_script(script: str) -> str:
        try:
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.stderr:
                print(f"[MacOS Error] {result.stderr.strip()}")
            return result.stdout.strip()
        except Exception as e:
            print(f"[AppleScript Exec Error]: {e}")
            return ""

    @classmethod
    def get_volume(cls) -> int:
        res = cls._run_apple_script('output volume of (get volume settings)')
        return int(res) if res.isdigit() else 0

    @classmethod
    def execute_action(cls, gesture_name: str) -> bool:
        action = cls.GESTURE_MAPPING.get(gesture_name)
        if not action:
            return False

        try:
            if action == "volumeup":
                new_vol = min(100, cls.get_volume() + 10)
                cls._run_apple_script(f'set volume output volume {new_vol}')
                print(f"🔊 音量调高至: {new_vol}%")
                
            elif action == "volumedown":
                new_vol = max(0, cls.get_volume() - 10)
                cls._run_apple_script(f'set volume output volume {new_vol}')
                print(f"🔉 音量调低至: {new_vol}%")
                
            elif action == "playpause":
                script = '''
                tell application "Music"
                    if player state is playing then
                        pause
                        return "Paused"
                    else
                        play
                        return "Playing"
                    end if
                end tell
                '''
                status = cls._run_apple_script(script)
                print(f"🎵 音乐状态: {status}")
                
            elif action == "mute":
                script = '''
                if output muted of (get volume settings) then
                    set volume without output muted
                    return "Unmuted"
                else
                    set volume with output muted
                    return "Muted"
                end if
                '''
                status = cls._run_apple_script(script)
                print(f"🔇 扬声器: {status}")
            return True
        except Exception as e:
            print(f"媒体控制异常: {e}")
            return False


# ==========================================
# 2. 核心数据管线与推理引擎
# ==========================================
def load_inference_model(device: torch.device) -> torch.nn.Module:
    """加载 PyTorch 推理模型"""
    print(f"[*] 正在加载模型 (Device: {device})...")
    model = GestureCNN()
    try:
        checkpoint = torch.load(Config.MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, GestureCNN):
            model.load_state_dict(checkpoint.state_dict())
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("[+] 模型加载成功。")
        return model
    except Exception as e:
        print(f"[-] 模型加载失败: {e}")
        exit(1)

def apply_dbscan_filtering(data: np.ndarray, eps: float = 0.09, min_samples: int = 2) -> np.ndarray:
    """雷达点云 DBSCAN 去噪"""
    if data is None or len(data) < min_samples:
        return np.array([])

    spatial_data = data[:, :3]
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(spatial_data)

    if np.all(clusters == -1):
        return np.array([])

    largest_cluster = np.bincount(clusters[clusters != -1]).argmax()
    return data[clusters == largest_cluster]


async def process_sensor_stream(sens: Sensor, model: torch.nn.Module, device: torch.device, 
                                mode: int, gesture_name: str, save_path: str):
    """
    mode 1: 实时推理控制
    mode 2: 采集训练数据
    """
    frame_buffer = [] 
    original_points_history, filtered_points_history = [], []
    labels_map = {0: "Clockwise", 1: "Counter Clockwise", 2: "Swipe", 3: "Swipe Up and Down"}
    
    print("\n[*] 传感器数据流已启动，等待雷达数据...")
    await asyncio.sleep(1)

    try:
        while sens.is_alive():
            sensor_object_data = await sens.get_data()
            incoming_data_array = sensor_object_data.get()
            
            if len(incoming_data_array) == 0:
                continue

            original_data = np.array(incoming_data_array)
            filtered_data = apply_dbscan_filtering(incoming_data_array)

            if len(filtered_data) == 0:
                continue

            # 记录历史用于 3D 可视化
            original_points_history.extend(original_data)
            filtered_points_history.extend(filtered_data)

            # 计算单帧均值特征并压入队列
            frame_feature = np.mean(filtered_data, axis=0).reshape(1, 4).astype("float32")
            frame_buffer.append(frame_feature)

            # ----------------------------------------
            # 模式 1: 实时推理与系统控制
            # ----------------------------------------
            if mode == 1 and len(frame_buffer) >= Config.SEQ_LENGTH:
                # 构建时序窗口张量: (1, seq_len, features)
                window_data = np.concatenate(frame_buffer[-Config.SEQ_LENGTH:], axis=0)
                input_tensor = torch.from_numpy(window_data).reshape(1, Config.SEQ_LENGTH, 4).float().to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    confidence, predicted = torch.max(torch.softmax(outputs, dim=1), 1)
                    prediction_label = labels_map[predicted.item()]
                    conf_score = confidence.item() * 100

                print(f"| 动作: {prediction_label:<20} | 置信度: {conf_score:.2f}% |")

                if conf_score > Config.CONFIDENCE_THRESHOLD:
                    MacOSMediaController.execute_action(prediction_label)
                    # 触发动作后清空缓存，设置冷却时间防误触
                    frame_buffer.clear()
                    original_points_history.clear()
                    filtered_points_history.clear()
                    await asyncio.sleep(1.5)  
                else:
                    # 队列滑动
                    frame_buffer.pop(0)

            # ----------------------------------------
            # 模式 2: 训练数据采集
            # ----------------------------------------
            elif mode == 2:
                print(f"采集进度: {len(frame_buffer)}/{Config.MIN_CAPTURE_SAMPLES}", end="\r")
                if len(frame_buffer) >= Config.MIN_CAPTURE_SAMPLES:
                    data_to_store = np.vstack([arr.reshape(-1, 4) for arr in frame_buffer])
                    df = pd.DataFrame(data_to_store, columns=["x", "y", "z", "Doppler"])
                    df.to_csv(save_path, index=False)
                    print(f"\n[+] 数据已保存至: {save_path}")
                    break

            await asyncio.sleep(0.01) # 让出事件循环

    except asyncio.CancelledError:
        print("\n[*] 接收到中断信号，停止数据流...")
    except Exception as e:
        print(f"[-] 数据流处理异常: {e}")


# ==========================================
# 3. 硬件交互与主入口
# ==========================================
def setup_hardware() -> Sensor:
    """初始化并配置雷达硬件"""
    sensor = IWR6843AOP("1", verbose=False)
    if not os.path.exists(Config.CFG_FILE_PATH):
        print(f"[-] 找不到配置文件: {Config.CFG_FILE_PATH}")
        exit(1)
        
    file_cfg = load_cfg_file(Config.CFG_FILE_PATH)

    if not sensor.connect_config(Config.CLI_PORT, 115200):
        print("[-] 雷达 Config 端口连接失败。")
        exit(1)

    if not sensor.connect_data(Config.DATA_PORT, 921600):
        print("[-] 雷达 Data 端口连接失败。")
        exit(1)

    if not sensor.send_config(file_cfg, max_retries=1):
        print("[-] 下发雷达配置失败。")
        exit(1)

    print("[+] 毫米波雷达硬件初始化完成。")
    return sensor


def main():
    print("=======================================")
    print("   mmWave Gesture Control System       ")
    print("=======================================")
    
    mode = 0
    while mode not in [1, 2]:
        try:
            mode = int(input("[1] 实时手势控制  |  [2] 采集训练数据 \n请选择模式: "))
        except ValueError:
            pass

    gesture_name = "inference_mode"
    save_path = ""

    if mode == 2:
        gesture_name = input("请输入要采集的手势名称 (如 clockwise): ").strip()
        target_dir = os.path.join(Config.DATA_DIR, gesture_name)
        os.makedirs(target_dir, exist_ok=True)
        
        file_idx = 1
        while os.path.exists(os.path.join(target_dir, f"{gesture_name}_{file_idx}.csv")):
            file_idx += 1
        save_path = os.path.join(target_dir, f"{gesture_name}_{file_idx}.csv")
        print(f"[*] 数据将保存至: {save_path}")

    # 1. 初始化硬件
    sensor = setup_hardware()
    
    # 2. 初始化 AI 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_inference_model(device) if mode == 1 else None

    # 3. 启动异步事件循环
    event_loop = asyncio.get_event_loop()
    try:
        event_loop.create_task(sensor.start_sensor())
        event_loop.create_task(process_sensor_stream(sensor, model, device, mode, gesture_name, save_path))
        event_loop.run_forever()
    except KeyboardInterrupt:
        print("\n[*] 检测到用户终止程序 (Ctrl+C)")
    finally:
        print("[*] 正在清理资源...")
        sensor.stop_sensor() 
        # 获取所有未完成的任务并取消
        pending = asyncio.all_tasks(loop=event_loop)
        for task in pending:
            task.cancel()
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        event_loop.close()
        print("[+] 系统已安全退出。")


if __name__ == "__main__":
    # 为了避免 Mac 上 Matplotlib 阻塞主线程的玄学问题，使用非阻塞模式
    plt.ion() 
    main()