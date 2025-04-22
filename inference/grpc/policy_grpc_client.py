import os
import sys
from pathlib import Path
import glob
import json
import numpy as np
import time
import cv2
import argparse
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
import io

# Add current directory to path to ensure modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import WandB visualization module - try both relative and absolute import
from wandb_visualizer import (
    TrajectoryVisualizer, 
    DataLogger, 
    ensure_wandb_login, 
    force_create_wandb_project, 
    disable_wandb_sync,
    forward_kinematics
)


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
import grpc
import torch
from datetime import datetime
from queue import Queue
from threading import Thread, Event

from proto import policy_pb2
from proto import policy_pb2_grpc
from single_arm.arm_angle import ArmAngle


def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoStream:
    def __init__(self, url, resolution=None, queue_size=2):
        self.url = url
        self.resolution = resolution  # (width, height) or None for default
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        cap = cv2.VideoCapture(self.url)
        
        # Set resolution if specified
        if self.resolution:
            width, height = self.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info(f"Setting camera resolution to {width}x{height}")
            
        # Get actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Actual camera resolution: {actual_width}x{actual_height}")
        
        while not self.stopped:
            if not cap.isOpened():
                print("Reconnecting camera...")
                cap = cv2.VideoCapture(self.url)
                if self.resolution:
                    width, height = self.resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                time.sleep(1)
                continue
                
            ret, frame = cap.read()
            if ret:
                if not self.queue.full():
                    self.queue.put(frame)
            else:
                time.sleep(0.01)  # Avoid excessive CPU usage
                
        cap.release()
        
    def read(self):
        return self.queue.get() if not self.queue.empty() else None
        
    def stop(self):
        self.stopped = True


class PolicyClient:
    def __init__(self, server_address: str = "localhost:50051"):
        """Initialize the gRPC client with server address"""
        self.server_address = server_address
        # Increase message size limits
        self.channel = grpc.insecure_channel(
            server_address,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
                ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50MB
            ]
        )
        self.stub = policy_pb2_grpc.PolicyServiceStub(self.channel)
        logger.info(f"Connected to gRPC server at {server_address}")
    
    def health_check(self) -> str:
        """Check if the server is running"""
        try:
            response = self.stub.HealthCheck(policy_pb2.HealthCheckRequest())
            return response.status
        except grpc.RpcError as e:
            logger.error(f"Health check failed: {e}")
            return f"Error: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            response = self.stub.GetModelInfo(policy_pb2.ModelInfoRequest())
            return {
                "status": response.status,
                "model_path": response.model_path,
                "device": response.device,
                "input_features": response.input_features,
                "output_features": response.output_features,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def predict(self, image_wrist: np.ndarray, image_head: np.ndarray, state: List[float], task: str = None) -> Tuple[List[float], float]:
        """
        Send prediction request to the server
        
        Args:
            image_wrist: Wrist camera RGB image as numpy array (H, W, C)
            image_head: Head camera RGB image as numpy array (H, W, C), can be None
            state: State vector as list of floats
            task: Task description for language-conditioned policies
            
        Returns:
            Tuple of (prediction, inference_time_ms)
        """
        try:
            # Convert wrist camera image to JPEG bytes
            success1, encoded_img1 = cv2.imencode('.jpg', (image_wrist * 255).astype(np.uint8))
            if not success1:
                raise ValueError("Failed to encode wrist camera image")
            img_bytes1 = encoded_img1.tobytes()
            
            # Get wrist image dimensions
            img1_height, img1_width = image_wrist.shape[0], image_wrist.shape[1]
            
            # Create request with encoded images
            request = policy_pb2.PredictRequest(
                encoded_image=img_bytes1,  # Primary image (wrist)
                image_format="jpeg",
                image_height=img1_height,
                image_width=img1_width,
                state=state
            )
            
            # Add task if provided
            if task:
                request.task = task
                logger.info(f"Added task: '{task}'")
            
            # Add head camera image if provided
            if image_head is not None:
                success2, encoded_img2 = cv2.imencode('.jpg', (image_head * 255).astype(np.uint8))
                if not success2:
                    raise ValueError("Failed to encode head camera image")
                img_bytes2 = encoded_img2.tobytes()
                
                # Get head image dimensions
                img2_height, img2_width = image_head.shape[0], image_head.shape[1]
                
                # Add to request
                request.encoded_image2 = img_bytes2
                request.image2_height = img2_height
                request.image2_width = img2_width
                
                logger.info(f"Sending both camera images - Wrist: {img1_width}x{img1_height}, Head: {img2_width}x{img2_height}")
            else:
                logger.info(f"Sending only wrist camera image: {img1_width}x{img1_height}")
            
            # Time the request
            start_time = time.perf_counter()
            response = self.stub.Predict(request)
            end_time = time.perf_counter()
            
            # Calculate round-trip time
            rtt_ms = (end_time - start_time) * 1000
            logger.info(f"Round-trip time: {rtt_ms:.2f}ms, Server inference time: {response.inference_time_ms:.2f}ms")
            
            # Log image sizes
            wrist_size = len(img_bytes1) / 1024
            logger.info(f"Sent wrist image size: {wrist_size:.2f}KB")
            if image_head is not None:
                head_size = len(img_bytes2) / 1024
                logger.info(f"Sent head image size: {head_size:.2f}KB")
            
            return response.prediction, response.inference_time_ms
        
        except grpc.RpcError as e:
            logger.error(f"Prediction failed: {e}")
            return [], 0.0
    
    def close(self):
        """Close the gRPC channel"""
        self.channel.close()


def get_observation_from_streams(stream_wrist, stream_head, state_data):
    """Get observation data from video streams and state data"""
    frame_wrist = stream_wrist.read()
    if frame_wrist is None:
        raise ValueError("Unable to read frame from wrist camera")
    
    # Convert wrist image to RGB and normalize
    frame_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
    frame_wrist = torch.from_numpy(frame_wrist).float() / 255.0  # Normalize to 0-1
    
    # Get head camera data (if available)
    frame_head = None
    if stream_head is not None:
        frame_head = stream_head.read()
        if frame_head is not None:
            frame_head = cv2.cvtColor(frame_head, cv2.COLOR_BGR2RGB)
            frame_head = torch.from_numpy(frame_head).float() / 255.0
    
    print("frame_wrist shape:", frame_wrist.shape)
    if frame_head is not None:
        print("frame_head shape:", frame_head.shape)
    
    # Get state data
    state_tensor = torch.tensor(state_data).float()
    
    # Create observation dictionary
    observation = {
        "observation.images.cam_wrist": frame_wrist,
        "observation.state": state_tensor
    }
    
    # Add head camera if available
    if frame_head is not None:
        observation["observation.images.cam_head"] = frame_head
    
    return observation


def get_state_from_parquet(parquet_path: str, frame_index: int = 0) -> List[float]:
    """Load state from a parquet file"""
    try:
        df = pd.read_parquet(parquet_path)
        state = df.iloc[frame_index]['observation.state']
        # Make numpy array writable before converting
        state = state.copy()
        return state.tolist()
    except Exception as e:
        logger.error(f"Failed to read state from parquet: {e}")
        raise


class KeyboardMonitor:
    def __init__(self, follower_arm):
        self.follower_arm = follower_arm
        self.enabled = True
        self.stop_event = Event()
        self.thread = Thread(target=self._monitor_keyboard, daemon=True)
        
    def start(self):
        self.thread.start()
        return self
        
    def _monitor_keyboard(self):
        import sys
        import tty
        import termios
        import select
        
        logger.info("Keyboard monitor started. Press Enter to toggle enable/disable.")
        logger.info(f"Current state: {'Enabled' if self.enabled else 'Disabled'}")
        
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            while not self.stop_event.is_set():
                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    # Check for Enter key (both '\n' and '\r' for compatibility)
                    if key in ['\n', '\r']:
                        self.enabled = not self.enabled
                        self.follower_arm.robot.set_enable(self.enabled)
                        status = "Enabled" if self.enabled else "Disabled"
                        logger.info(f"Arm {status}")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser(description="Policy gRPC Client")
    parser.add_argument("--server", default="localhost:50051", help="Server address")
    parser.add_argument("--serial_number", default="396636713233", help="Serial number of the follower arm")
    parser.add_argument("--camera_wrist", default="http://192.168.237.100:8080/?action=stream", help="Wrist camera URL")
    parser.add_argument("--camera_head", default="http://192.168.237.157:8080/?action=stream", help="Head camera URL (optional)")
    parser.add_argument("--wrist_resolution", default="1280x720", help="Wrist camera resolution (WxH)")
    parser.add_argument("--head_resolution", default="1280x720", help="Head camera resolution (WxH)")
    parser.add_argument("--inference_time_s", type=int, default=300, help="Inference time in seconds")
    parser.add_argument("--control_rate", type=int, default=10, help="Control rate in Hz")
    parser.add_argument("--queue_size", type=int, default=2, help="Queue size")
    parser.add_argument("--warm_up", type=int, default=20, help="Warm-up time")
    parser.add_argument("--task", type=str, default="pick the cube into the box", help="Task description for language-conditioned policies")
    parser.add_argument("--log_dir", type=str, default="/Users/jack/lab_intern/dummy_ctrl/log/lerobot_dp_traj", help="Directory to save log data")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Enable logging to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="lerobot_dp_traj", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/username")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="WandB API key (alternative to using wandb login)")
    parser.add_argument("--disable_wandb_sync", action="store_true", help="Disable WandB process sync (use thread mode)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    args = parser.parse_args()

    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("启用调试模式 - 显示详细日志")
        
        # 添加一个专门的wandb调试日志处理器
        try:
            import wandb
            wandb.setup(settings=wandb.Settings(console="debug"))
            print("WandB调试日志已启用")
        except Exception as e:
            print(f"无法设置WandB调试: {e}")
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Disable WandB process sync if requested
    if args.disable_wandb_sync:
        disable_wandb_sync()

    # Set WandB environment variables if specified
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    
    # Set WandB API key if provided
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        logger.info("Using provided WandB API key")
    
    # 为了避免相对导入问题，确保当前目录在Python路径中
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    if current_file_dir not in sys.path:
        sys.path.append(current_file_dir)
        print(f"添加当前目录到Python路径: {current_file_dir}")
    
    # 尝试直接导入wandb_visualizer模块
    try:
        # 首先尝试直接导入
        if args.use_wandb:
            # 确保能找到wandb_visualizer模块
            import importlib.util
            module_path = os.path.join(current_file_dir, "wandb_visualizer.py")
            if os.path.exists(module_path):
                print(f"找到wandb_visualizer模块: {module_path}")
            else:
                print(f"警告: 未找到wandb_visualizer模块: {module_path}")
                
            # 显示当前环境和可用模块
            try:
                import wandb
                print(f"WandB版本: {wandb.__version__}")
                print(f"WandB API密钥可用: {wandb.api.api_key is not None}")
            except Exception as e:
                print(f"导入和检查WandB时出错: {e}")
                
            # 设置项目名称环境变量
            os.environ["WANDB_PROJECT"] = args.wandb_project
            print(f"设置WandB项目名: {args.wandb_project}")
            
            # 尝试登录
            print("正在尝试登录WandB...")
            try:
                # ---> MODIFIED: Only attempt interactive login if no API key is found <---
                if not os.environ.get("WANDB_API_KEY"):
                    logger.info("未检测到 WANDB_API_KEY。尝试交互式登录或使用缓存的凭据。")
                    # Try login without forcing relogin
                    if wandb.login(): 
                        logger.info("WandB 登录成功 (交互式或缓存)。")
                    else:
                        logger.warning("WandB 交互式登录或缓存凭据检查失败。")
                        args.use_wandb = False # Disable wandb if login fails
                else:
                    logger.info("检测到 WANDB_API_KEY。跳过显式 wandb.login() 调用。")
                # ---> END MODIFIED <---

                # Check if login succeeded before proceeding (args.use_wandb might be False now)
                if args.use_wandb:
                    # Verify login status after potential attempt or skip
                    if wandb.api.api_key:
                         logger.info("确认 WandB 已通过 API 密钥或登录进行身份验证。")
                    else:
                        logger.warning("WandB 身份验证检查失败，即使尝试了登录。禁用 WandB。")
                        args.use_wandb = False

            except Exception as e:
                logger.error(f"WandB 登录或检查时发生错误: {e}", exc_info=True)
                args.use_wandb = False
    except Exception as e:
        print(f"导入或初始化WandB时发生错误: {e}")
        import traceback
        traceback.print_exc()
        args.use_wandb = False

    # Parse camera resolutions
    wrist_width, wrist_height = map(int, args.wrist_resolution.split('x'))
    wrist_resolution = (wrist_width, wrist_height)
    
    head_resolution = None
    if args.head_resolution:
        head_width, head_height = map(int, args.head_resolution.split('x'))
        head_resolution = (head_width, head_height)

    # Initialize data logger
    data_logger = DataLogger(
        log_dir=args.log_dir, 
        use_wandb=args.use_wandb, 
        wandb_project=args.wandb_project, 
        wandb_entity=args.wandb_entity, 
        wandb_api_key=args.wandb_api_key
    )

    logger_fibre = fibre.utils.Logger(verbose=True)
    follower_arm = fibre.find_any(serial_number=args.serial_number, logger=logger_fibre)
    follower_arm.robot.resting()
    follower_arm.robot.set_enable(True)
    follower_arm.robot.move_j(0, -30, 90, 0, 70, 0)
    # follower_arm.robot.move_j(0, 0, 90, 0, 0, 0)
    joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
    
    arm_controller = ArmAngle(None, follower_arm, joint_offset)
    
    # Start keyboard monitor
    keyboard_monitor = KeyboardMonitor(follower_arm).start()
    
    # Initialize wrist camera
    print("Initializing wrist camera stream...")
    stream_wrist = VideoStream(url=args.camera_wrist, resolution=wrist_resolution, queue_size=args.queue_size).start()
    
    # Initialize head camera if URL provided
    stream_head = None
    if args.camera_head:
        print("Initializing head camera stream...")
        stream_head = VideoStream(url=args.camera_head, resolution=head_resolution, queue_size=args.queue_size).start()
    
    time.sleep(2.0)  # Wait for streams to start

    # Create client
    client = PolicyClient(args.server)
    
    # Check server health
    health_status = client.health_check()
    logger.info(f"Server health: {health_status}")
    
    # Get model info
    model_info = client.get_model_info()
    logger.info(f"Model info: {model_info}")

    inference_time_s = args.inference_time_s
    control_rate = args.control_rate
    warm_up = args.warm_up

    # 打印开始数据收集的提示信息
    if args.use_wandb:
        print("\n" + "=" * 80)
        print(f"🚀 开始数据收集! 将采集 {inference_time_s} 秒数据，控制率 {control_rate}Hz")
        print(f"📊 轨迹将实时上传到WandB进行可视化")
        print(f"💡 提示: 记得在浏览器中打开WandB链接查看实时轨迹可视化")
        print("=" * 80 + "\n")

    logger.info(f"Begin with {warm_up} warm-up steps...")

    # Combined loop for warm-up and inference
    total_steps = warm_up + (inference_time_s * control_rate)
    for step in range(total_steps):
        is_warmup = step < warm_up
        if is_warmup:
            logger.info(f"Warm-up step {step + 1}/{warm_up}")
        elif step == warm_up:
            logger.info("Warm-up finished, starting inference...")

        # Read the follower state and access the frames from the cameras
        follow_joints = arm_controller.get_follow_joints()
        gripper = follower_arm.robot.hand.angle
        current_state = follow_joints.tolist() + [gripper]
        
        print("current_state: ", current_state)

        try:
            # Get images from video streams
            observation = get_observation_from_streams(stream_wrist, stream_head, current_state)
            
            # Get wrist camera image
            image_wrist = observation["observation.images.cam_wrist"].numpy()
            
            # Get head camera image if available
            image_head = None
            if "observation.images.cam_head" in observation:
                image_head = observation["observation.images.cam_head"].numpy()
            
            state = observation["observation.state"].numpy().tolist()
            
            # Make prediction with both images (for both warm-up and inference)
            prediction, inference_time_ms = client.predict(image_wrist, image_head, state, args.task)
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Prediction type: {type(prediction)}")
            logger.info(f"Current state: {current_state}")
            logger.info(f"Server inference time: {inference_time_ms:.2f}ms")
            
            # Save data to log folder (including prediction results)
            if not is_warmup:
                data_logger.save_data(image_wrist, image_head, state, prediction, inference_time_ms)
            
            # Only move arm if not in warm-up and it's enabled
            if not is_warmup and keyboard_monitor.enabled:
                follower_arm.robot.move_j(
                    prediction[0],  # joint_1
                    prediction[1],  # joint_2 
                    prediction[2],  # joint_3
                    prediction[3],  # joint_4
                    prediction[4],  # joint_5
                    prediction[5]   # joint_6
                )
                if prediction[6] < -155.0:
                    angle = -165.0
                else:
                    angle = prediction[6]
                follower_arm.robot.hand.set_angle(angle)
        except Exception as e:
            error_type = "Warm-up" if is_warmup else "Inference"
            logger.error(f"{error_type} Error: {e}")
            
        precise_sleep(1 / control_rate, time_func=time.monotonic)
    
    # Clean up
    keyboard_monitor.stop()
    stream_wrist.stop()
    if stream_head:
        stream_head.stop()
    client.close()

    # Finalize data logger
    data_logger.finalize()


if __name__ == "__main__":
    main() 