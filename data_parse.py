import pandas as pd
import numpy as np
import os
import time
import subprocess  # 替换 pyautogui
import torch
import asyncio
from pymmWave.utils import load_cfg_file
from pymmWave.sensor import Sensor
from pymmWave.IWR6843AOP import IWR6843AOP
from asyncio import get_event_loop, sleep
from model import GestureCNN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def run_apple_script(script):
    """运行 AppleScript 命令"""
    try:
        result = subprocess.run(['osascript', '-e', script],
                              capture_output=True,
                              text=True)
        if result.stderr:
            print(f"错误: {result.stderr}")
        return result.stdout.strip()
    except Exception as e:
        print(f"执行出错: {e}")
        return None

def get_system_volume():
    """获取当前系统音量"""
    script = 'output volume of (get volume settings)'
    result = run_apple_script(script)
    try:
        return int(result) if result else 0
    except:
        return 0

def control_media(action):
    """控制媒体功能"""
    try:
        if action == "volumeup":
            current_volume = get_system_volume()
            new_volume = min(100, current_volume + 10)
            script = f'set volume output volume {new_volume}'
            run_apple_script(script)
            after_volume = get_system_volume()
            print(f"音量从 {current_volume} 增加到 {after_volume}")

        elif action == "volumedown":
            current_volume = get_system_volume()
            new_volume = max(0, current_volume - 10)
            script = f'set volume output volume {new_volume}'
            run_apple_script(script)
            after_volume = get_system_volume()
            print(f"音量从 {current_volume} 减少到 {after_volume}")

        elif action == "playpause":
            script = '''
            tell application "Music"
                if player state is playing then
                    pause
                    return "已暂停"
                else
                    play
                    return "开始播放"
                end if
            end tell
            '''
            result = run_apple_script(script)
            print(f"音乐播放状态: {result}")

        elif action == "mute":
            script = '''
            if output muted of (get volume settings) then
                set volume without output muted
                return "已取消静音"
            else
                set volume with output muted
                return "已静音"
            end if
            '''
            result = run_apple_script(script)
            print(result)

        return True

    except Exception as e:
        print(f"执行 {action} 时出错: {e}")
        return False

# 手势映射
gesture_to_action = {
    "Clockwise": "volumeup",        # 音量增加
    "Counter Clockwise": "volumedown",  # 音量减少
    "Swipe": "playpause",           # 播放/暂停
    "Swipe Up and Down": "mute"     # 静音
}

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

def apply_dbscan_filtering(data, eps, min_samples):
    if data is None or len(data) < min_samples:
        print("Too few data points, returning original data")
        return data

    print("\n=== DBSCAN Clustering Analysis ===")
    print(f"Input data points: {len(data)}")

    # Directly use original spatial coordinates for clustering
    spatial_data = data[:, :3]  # Only use x, y, z coordinates

    # Apply DBSCAN without standardization
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(spatial_data)

    print("Clustering results:", clusters)

    # If all points are marked as noise (-1), return empty array
    if np.all(clusters == -1):
        print("All points identified as noise")
        return np.array([])

    largest_cluster = max(set(clusters), key=list(clusters).count)
    filtered_data = data[clusters == largest_cluster]

    print(f"Points retained: {len(filtered_data)}")
    return filtered_data


def user_mode():
    mode = 0
    file_index = 1
    dir_path = ''
    parent_dir = "./data/"
    sensor_data = pd.DataFrame([], columns=["x", "y", "z", "Doppler"])
    print("Select Mode: ")
    while not mode or mode > 2:
        mode = int(input("[1] - Gesture Classification  | [2] - Data Capture \n"))
        print(mode)

    if mode == 1:
        print("[1] Gesture Classification Selected")
        new_gesture_name = "placeholder"
        dir_path = parent_dir

    elif mode == 2:
        print("[2] Data Capture Selected")
        new_gesture_name = str(input("Name of gesture: "))
        dir_path = os.path.join(parent_dir, new_gesture_name)

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            sensor_data.to_csv(f"data/{new_gesture_name}/{new_gesture_name}_{file_index}.csv", index=False)
        else:
            print("Gesture Already Exists, will create new dataset within gesture file")
            while (os.path.exists(f"data/{new_gesture_name}/{new_gesture_name}_{file_index}.csv")):
                file_index += 1

            sensor_data.to_csv(f"data/{new_gesture_name}/{new_gesture_name}_{file_index}.csv", index=False)

    return new_gesture_name, dir_path, mode, file_index


def configure_sensor():
    sensor1 = IWR6843AOP("1", verbose=False)
    file = load_cfg_file("/Users/chengju/Desktop/FYP/simpleCNN (2)/simpleCNN/mmwave_config/xwr68xx_AOP_config_10FPS_maxRange_30cm.cfg")

    # Your CONFIG serial port name
    config_connected = sensor1.connect_config('/dev/tty.SLAB_USBtoUART', 115200)
    if not config_connected:
        print("Config connection failed.")
        exit()

    # Your DATA serial port name
    data_connected = sensor1.connect_data('/dev/tty.SLAB_USBtoUART4', 921600)
    if not data_connected:
        print("Data connection failed.")
        exit()

    if not sensor1.send_config(file, max_retries=1):
        print("Sending configuration failed")
        exit()

    return sensor1


def visualize_dbscan(original_data, filtered_data):
    plt.clf()
    fig = plt.figure(figsize=(15, 10))

    # 设置坐标轴范围
    axis_range = [-0.4, 0.4]
    ticks = np.arange(-0.3, 0.31, 0.01)

    # 创建3D图 - 原始数据
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(original_data[:, 0],
                           original_data[:, 1],
                           original_data[:, 2],
                           c=original_data[:, 3],  # 使用多普勒值作为颜色
                           cmap='jet',  # 使用jet色图
                           s=50)  # 点的大小

    ax1.set_title(f'Original Data (Points: {len(original_data)})')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_xlim(axis_range)
    ax1.set_ylim(axis_range)
    ax1.set_zlim(axis_range)
    fig.colorbar(scatter1, label='Doppler (m/s)')

    # 创建3D图 - 过滤后数据
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(filtered_data[:, 0],
                           filtered_data[:, 1],
                           filtered_data[:, 2],
                           c=filtered_data[:, 3],  # 使用多普勒值作为颜色
                           cmap='jet',  # 使用jet色图
                           s=50)  # 点的大小

    ax2.set_title(f'After DBSCAN (Points: {len(filtered_data)})')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_xlim(axis_range)
    ax2.set_ylim(axis_range)
    ax2.set_zlim(axis_range)
    fig.colorbar(scatter2, label='Doppler (m/s)')

    # 添加网格
    ax1.grid(True)
    ax2.grid(True)

    # 设置视角
    ax1.view_init(elev=20, azim=45)
    ax2.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)


async def print_data(sens: Sensor, gesture_name, directory, mode, index):
    try:
        cache_flush = 0
        train_data = []
        original_points_history = []
        filtered_points_history = []
        encoded_labels = ["Clockwise", "Counter Clockwise", "Swipe", "Swipe Up and Down"]
        label_mapping = {idx: label for idx, label in enumerate(encoded_labels)}
        data_collection_complete = False
        MIN_SAMPLES = 500

        await asyncio.sleep(1)
        while sens.is_alive() and not data_collection_complete:
            try:
                sensor_object_data = await sens.get_data()
                incoming_data_array = sensor_object_data.get()
                print("\n New scan!")

                if cache_flush:
                    incoming_data_array = []
                    cache_flush = 0
                    print("\n Cache flushed!")

                if len(incoming_data_array) > 0:
                    original_data = np.array(incoming_data_array)
                    filtered_data = apply_dbscan_filtering(incoming_data_array, eps=0.09, min_samples=2)

                    if len(original_data) > 0:
                        original_points_history.extend(original_data)
                    if len(filtered_data) > 0:
                        filtered_points_history.extend(filtered_data)

                    if len(filtered_data) > 0:
                        average_of_data = np.mean(filtered_data, axis=0).reshape(1, 4).astype("float32")
                        train_data.append(average_of_data)

                    print("Points collected:", len(train_data))

                    if mode == 1 and len(train_data) >= 3:
                        try:
                            live_test_data = np.concatenate(train_data, axis=0)
                            n_samples = (len(live_test_data) // 3) * 3
                            live_test_data = live_test_data[:n_samples]
                            live_test_data = live_test_data.reshape(-1, 3, 4).astype("float32")
                            print('\n Live test data shape: ', live_test_data.shape)

                            if len(original_points_history) > 0 and len(filtered_points_history) > 0:
                                original_points = np.array(original_points_history)
                                filtered_points = np.array(filtered_points_history)
                                visualize_dbscan(original_points, filtered_points)

                            input_tensor = torch.from_numpy(live_test_data).float().to(device)

                            with torch.no_grad():
                                outputs = model(input_tensor)
                                _, predicted = torch.max(outputs.data, 1)
                                prediction = label_mapping[predicted.item()]
                                confidence = torch.softmax(outputs, dim=1).max().item() * 100

                            print(f"\nThe predicted gesture is: {prediction}, and the confidence is: {confidence}%")

                            # 使用新的媒体控制方法
                            if confidence > 70:
                                if prediction in gesture_to_action:
                                    action = gesture_to_action[prediction]
                                    print(f"执行动作: {action}")
                                    control_media(action)

                            train_data = []
                            original_points_history = []
                            filtered_points_history = []
                            cache_flush = 1
                            await asyncio.sleep(2)

                        except Exception as e:
                            print(f"Error in classification mode: {e}")
                            train_data = []

                    elif mode == 2 and len(train_data) >= MIN_SAMPLES:
                        try:
                            if train_data:
                                data_to_store = np.vstack([arr.reshape(-1, 4) for arr in train_data])
                                gesture_data = pd.DataFrame(
                                    data_to_store,
                                    columns=["x", "y", "z", "Doppler"]
                                )

                                if len(original_points_history) > 0 and len(filtered_points_history) > 0:
                                    original_points = np.array(original_points_history)
                                    filtered_points = np.array(filtered_points_history)
                                    visualize_dbscan(original_points, filtered_points)

                                output_path = f"{directory}/{gesture_name}_{index}.csv"
                                gesture_data.to_csv(output_path, mode='a', index=False, header=False)
                                print(f"Saved {len(gesture_data)} samples to {output_path}")
                                data_collection_complete = True

                        except Exception as e:
                            print(f"Error in data capture mode: {e}")
                            print(f"Data shape before processing: {[arr.shape for arr in train_data]}")

            except asyncio.CancelledError:
                print("Data collection cancelled")
                break
            except Exception as e:
                print(f"Error in data processing loop: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Error in print_data: {e}")
    finally:
        print("Data collection completed")
        data_collection_complete = True

def main():
    gesture_name, directory, mode, index = user_mode()
    sensor = configure_sensor()

    event_loop = get_event_loop()
    event_loop.create_task(sensor.start_sensor())
    event_loop.create_task(print_data(sensor, gesture_name, directory, mode, index))
    event_loop.run_forever()


if __name__ == "__main__":
    main()