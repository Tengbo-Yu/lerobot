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
import grpc
import torch
import numpy as np
import time
import cv2
import argparse
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
import io
from queue import Queue
from threading import Thread, Event
from single_arm.arm_angle import ArmAngle

from proto import policy_pb2
from proto import policy_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    def predict(self, image_wrist: np.ndarray, image_head: np.ndarray, state: List[float]) -> Tuple[List[float], float]:
        """
        Send prediction request to the server
        
        Args:
            image_wrist: Wrist camera RGB image as numpy array (H, W, C)
            image_head: Head camera RGB image as numpy array (H, W, C), can be None
            state: State vector as list of floats
            
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

def get_observation_from_video(video_path: str, frame_index: int = 0) -> Tuple[np.ndarray, int]:
    """Load a frame from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f'Video shape=({frame_count}, {height}, {width}, 3) fps={fps}')
    
    # Set frame to read
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Failed to read video frame")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    cap.release()
    return frame, frame_count

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

def get_observation_from_streams(stream_wrist, stream_head):
    """Get observation from video streams"""
    frame_wrist = stream_wrist.read()
    if frame_wrist is None:
        raise ValueError("Cannot read frame from wrist camera")
    
    # Convert wrist image to RGB and normalize
    frame_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
    frame_wrist = frame_wrist.astype(np.float32) / 255.0
    
    # Get head camera data if available
    frame_head = None
    if stream_head is not None:
        frame_head = stream_head.read()
        if frame_head is not None:
            frame_head = cv2.cvtColor(frame_head, cv2.COLOR_BGR2RGB)
            frame_head = frame_head.astype(np.float32) / 255.0
    
    return frame_wrist, frame_head

def main():
    parser = argparse.ArgumentParser(description="Policy gRPC Client with File Warmup")
    parser.add_argument("--server", default="localhost:50051", help="Server address")
    parser.add_argument("--wrist_video", default="/Users/jack/Desktop/dummy_ctrl/datasets/pick_place_0406/videos/chunk-000/observation.images.cam_wrist/episode_000000.mp4", help="Path to wrist camera video file (for warmup)")
    parser.add_argument("--head_video", default="/Users/jack/Desktop/dummy_ctrl/datasets/pick_place_0406/videos/chunk-000/observation.images.cam_head/episode_000000.mp4", help="Path to head camera video file (for warmup)")
    parser.add_argument("--parquet", default="/Users/jack/Desktop/dummy_ctrl/datasets/pick_place_0406/data/chunk-000/episode_000000.parquet", help="Path to parquet file with state data (for warmup)")
    parser.add_argument("--serial_number", default="396636713233", help="Serial number of the follower arm")
    parser.add_argument("--camera_wrist", default="http://192.168.237.249:8080/?action=stream", help="Wrist camera URL (for real-time)")
    parser.add_argument("--camera_head", default="http://192.168.237.157:8080/?action=stream", help="Head camera URL (for real-time)")
    parser.add_argument("--wrist_resolution", default="1280x720", help="Wrist camera resolution (WxH)")
    parser.add_argument("--head_resolution", default="1280x720", help="Head camera resolution (WxH)")
    parser.add_argument("--warmup_frames", type=int, default=30, help="Number of frames to use for warmup")
    parser.add_argument("--inference_time_s", type=int, default=60, help="Inference time after warmup in seconds")
    parser.add_argument("--control_rate", type=int, default=1, help="Control rate in Hz")
    args = parser.parse_args()
    
    # Initialize robot arm
    logger_fibre = fibre.utils.Logger(verbose=True)
    follower_arm = fibre.find_any(serial_number=args.serial_number, logger=logger_fibre)
    follower_arm.robot.resting()
    follower_arm.robot.move_j(0, -30, 90, 0, 70, 0)  # Initial position

    joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
    follower_arm.robot.set_enable(True)
    arm_controller = ArmAngle(None, follower_arm, joint_offset)
    
    # Start keyboard monitor for toggling robot control
    keyboard_monitor = KeyboardMonitor(follower_arm).start()
    
    # Parse camera resolutions
    wrist_width, wrist_height = map(int, args.wrist_resolution.split('x'))
    wrist_resolution = (wrist_width, wrist_height)
    
    head_resolution = None
    if args.head_resolution:
        head_width, head_height = map(int, args.head_resolution.split('x'))
        head_resolution = (head_width, head_height)
    
    # Initialize camera streams (for real-time phase)
    logger.info("Initializing wrist camera stream...")
    stream_wrist = VideoStream(url=args.camera_wrist, resolution=wrist_resolution).start()
    
    stream_head = None
    if args.camera_head:
        logger.info("Initializing head camera stream...")
        stream_head = VideoStream(url=args.camera_head, resolution=head_resolution).start()
    
    # Wait for streams to initialize
    time.sleep(2.0)
    
    # Create gRPC client
    client = PolicyClient(args.server)
    
    # Check server health
    health_status = client.health_check()
    logger.info(f"Server health: {health_status}")
    
    # Get model info
    model_info = client.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Load DataFrame for parquet state
    df = pd.read_parquet(args.parquet)
    
    # Wait after initial setup
    logger.info("Waiting for 2 seconds after initial setup...")
    time.sleep(2)
    
    try:
        # Phase 1: Warmup using saved data files
        logger.info(f"=== Starting warmup phase with {args.warmup_frames} frames ===")
        for frame_idx in range(args.warmup_frames):
            logger.info(f"Warmup frame {frame_idx+1}/{args.warmup_frames}")
            
            # Get images from video files
            image_wrist, _ = get_observation_from_video(args.wrist_video, frame_idx)
            
            image_head = None
            if args.head_video:
                image_head, _ = get_observation_from_video(args.head_video, frame_idx)
            
            # Get state from parquet file
            state = df.iloc[frame_idx]['observation.state'].copy().tolist()
            
            # Make prediction
            prediction, inference_time_ms = client.predict(image_wrist, image_head, state)
            
            # Display results
            logger.info(f"Warmup prediction: {prediction}")
            logger.info(f"Server time: {inference_time_ms:.2f}ms")
            
            # Small delay between warmup frames
            time.sleep(0.5)
        
        logger.info("=== Warmup phase completed ===")
        
        # Phase 2: Real-time inference and control
        logger.info(f"=== Starting real-time phase for {args.inference_time_s} seconds ===")
        
        # Calculate number of control steps
        control_steps = args.inference_time_s * args.control_rate
        
        for step in range(control_steps):
            logger.info(f"Real-time step {step+1}/{control_steps}")
            
            # Get current robot state
            follow_joints = arm_controller.get_follow_joints()
            gripper = follower_arm.robot.hand.angle
            current_state = follow_joints.tolist() + [gripper]
            logger.info(f"Current state: {current_state}")
            
            try:
                # Get images from camera streams
                image_wrist, image_head = get_observation_from_streams(stream_wrist, stream_head)
                
                # Make prediction
                prediction, inference_time_ms = client.predict(image_wrist, image_head, current_state)
                
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Server inference time: {inference_time_ms:.2f}ms")
                
                # Move robot based on prediction if enabled
                if keyboard_monitor.enabled and prediction:
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
                logger.error(f"Error in real-time phase: {e}")
            
            # Sleep to maintain control rate
            precise_sleep(1 / args.control_rate)
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        if stream_wrist:
            stream_wrist.stop()
        if stream_head:
            stream_head.stop()
        
        keyboard_monitor.stop()
        follower_arm.robot.resting()
        client.close()
        logger.info("Done.")


if __name__ == "__main__":
    main() 