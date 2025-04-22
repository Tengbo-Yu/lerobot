#%%
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy

PRETRAINED_POLICY_PATH = "/Users/yinzi/Downloads/Dummy_V2_workspace/dummy_ai/dummy_ctrl/checkpoints/train/cube_act_0326/100000/pretrained_model"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
policy = ACTPolicy.from_pretrained(PRETRAINED_POLICY_PATH)
#%%
policy.to(device)
policy.eval()
policy.reset()
#%%
import cv2
import torch
import numpy as np

def get_observation_from_video(video_path, frame_index=0):
    """从视频文件获取观察数据"""
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f'shape=({frame_count}, {height}, {width}, 3) fps={fps}')
    
    # 设置要读取的帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("无法读取视频帧")
    # 转换为RGB并归一化
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 释放视频捕获对象
    cap.release()
    return frame

video_path = "/Users/yinzi/Downloads/Dummy_V2_workspace/dummy_ai/dummy_ctrl/data/pick_cube_20demos/videos/chunk-000/observation.images.cam_wrist/episode_000021.mp4"
image = get_observation_from_video(video_path)
image = torch.from_numpy(image).float().permute(2, 0, 1)  / 255.0
import matplotlib.pyplot as plt
plt.imshow(image.permute(1, 2, 0).cpu().numpy())
plt.show()
#%%
import torch
import pandas as pd
parquet_path = "/Users/yinzi/Downloads/Dummy_V2_workspace/dummy_ai/dummy_ctrl/data/pick_cube_20demos/data/chunk-000/episode_000021.parquet"
# Read state from parquet
df = pd.read_parquet(parquet_path)
closest_idx = 0  # Get the first frame
state = df.iloc[closest_idx]['observation.state']
# Fix: Make numpy array writable before converting to tensor
state = state.copy()  # Create writable copy
state_tensor = torch.from_numpy(state).float()

# Create observation dict
observation = {
    "observation.images.cam_wrist": image.unsqueeze(0).to(device),
    "observation.state": state_tensor.unsqueeze(0).to(device)
}
#%%
with torch.inference_mode():
    action = policy.select_action(observation)
gt = df.iloc[closest_idx]['action']
gt = torch.tensor(gt).float().to(device)
print(action)
print(action.shape)
print(gt)
print(f"action - gt: {torch.round(action - gt)}")
