#%%
import os
import sys
from pathlib import Path

def find_project_root(current_dir, marker_files=(".git", "pyproject.toml", "setup.py")):
    current_dir = Path(current_dir).absolute()
    
    while current_dir != current_dir.parent:
        for marker in marker_files:
            if (current_dir / marker).exists():
                return current_dir
        current_dir = current_dir.parent
    
    return Path(os.getcwd())

current_dir = Path(__file__).parent.absolute()

project_root = find_project_root(current_dir)
sys.path.append(str(project_root))

import fibre
from pynput import keyboard
import numpy as np
# from __future__ import print_function
from single_arm.real_collector import LeRobotDataCollector
from single_arm.arm_angle import ArmAngle
from single_arm.bi_gripper_open import gripper_open
# logger verbose=True
logger = fibre.utils.Logger(verbose=True)
import time
import torch
import pandas as pd
import os
import numpy as np
import requests
import cv2
from queue import Queue
from threading import Thread


class VideoStream:
    def __init__(self, url, queue_size=128):
        self.url = url
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        cap = cv2.VideoCapture(self.url)
        while not self.stopped:
            if not cap.isOpened():
                print("重新连接摄像头...")
                cap = cv2.VideoCapture(self.url)
                time.sleep(1)
                continue
                
            ret, frame = cap.read()
            if ret:
                if not self.queue.full():
                    self.queue.put(frame)
            else:
                time.sleep(0.01)  # 避免CPU过度使用
                
        cap.release()
        
    def read(self):
        return self.queue.get() if not self.queue.empty() else None
        
    def stop(self):
        self.stopped = True

def get_observation_from_stream(stream, state_data):
    """从视频流和状态数据获取观察数据"""
    frame = stream.read()
    if frame is None:
        raise ValueError("无法读取视频帧")
    
    # 转换为RGB并归一化
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).float() / 255.0  # 归一化到 0-1
    frame = frame.permute(2, 0, 1)  # HWC to CHW format
    print("frame: ", frame.shape)
    # 获取状态数据
    state_tensor = torch.tensor(state_data).float()
    
    # 创建 observation 字典
    observation = {
        "observation.images.cam_wrist": frame,
        "observation.state": state_tensor
    }
    
    return observation

def busy_wait(duration_s: float) -> None:
    """
    Busy wait for the specified duration using monotonic time.
    
    Args:
        duration_s: Duration to wait in seconds
    """
    if duration_s <= 0:
        return
        
    start_time = time.monotonic()
    end_time = start_time + duration_s
    
    while time.monotonic() < end_time:
        pass
#%%
follow_arm = fibre.find_any(serial_number="396636713233", logger=logger)
follow_arm.robot.resting()
joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
follow_arm.robot.set_enable(True)
inference_time_s = 600
fps = 1
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Define the server endpoint
SERVER_URL = "http://localhost:8001/predict"
CAMERA_URL = "http://192.168.65.124:8080/?action=stream"
#%%
current_frame = 0
arm_controller = ArmAngle(None, follow_arm, joint_offset)

print("初始化视频流...")
stream = VideoStream(CAMERA_URL).start()
time.sleep(2.0)  # 等待视频流稳定
for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    # observation = robot.capture_observation()
    follow_joints = arm_controller.get_follow_joints()
    if follow_arm.robot.hand.angle < -165.0:
        gripper = 0.0
    else:
        gripper = 1.0
    current_state = follow_joints.tolist() + [gripper]
    print("current_state: ", current_state)
    # print("gt_action: ", gt_action)
    
    # Prepare data for server request
    
    # Make request to server
    try:
        observation = get_observation_from_stream(stream, current_state)
        
        # 准备服务器请求数据
        image = observation["observation.images.cam_wrist"].numpy().tolist()
        state = observation["observation.state"].numpy().tolist()
        response = requests.post(
            SERVER_URL,
            json={"image": image, "state": state},
            timeout=10
        )
        response.raise_for_status()
        prediction_data = response.json()
        print("服务器返回的预测数据:", prediction_data)
        action = torch.tensor(prediction_data["prediction"])
        follow_arm.robot.move_j(
                action[0],  # joint_1
                action[1],  # joint_2 
                action[2],  # joint_3
                action[3],  # joint_4
                action[4],  # joint_5
                action[5]   # joint_6
            )
        if action[6] >= 0.9:
            follow_arm.robot.hand.set_angle(-129.03999)  
        else:
            follow_arm.robot.hand.set_angle(-165.0) 
    except Exception as e:
        print(f"Error getting prediction from server: {e}")
        # Fallback to dummy prediction if server fails
        # action = torch.ones_like(gt_action)
    
    # print("predicted action: ", action)
    # print("gt - predicted = ", gt_action - action)


    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

    current_frame += 1

# %%
