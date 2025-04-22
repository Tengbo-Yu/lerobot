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
import random
from single_arm.arm_angle import ArmAngle

from proto import policy_pb2
from proto import policy_pb2_grpc

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

# Import Queue and Thread for real-time camera streams
from queue import Queue
from threading import Thread

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


def get_total_frames(video_path: str) -> int:
    """Get the total number of frames in a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


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


def generate_random_state():
    """Generate random state (joint angles and gripper)"""
    # Generate random joint angles between -180 and 180 degrees
    joints = np.random.uniform(-180, 180, 6)
    
    # Generate random gripper angle between -165 and 0 degrees
    gripper = np.random.uniform(-165, 0)
    
    # Combine into a single state
    state = np.concatenate([joints, [gripper]])
    
    return state.tolist()


def generate_random_image(height=720, width=1280, channels=3):
    """Generate a random RGB image"""
    # Create structured random image with colored rectangles
    img = np.zeros((height, width, channels), dtype=np.float32)
    
    # Create 5-10 random colored rectangles
    num_rectangles = np.random.randint(5, 11)
    for _ in range(num_rectangles):
        # Random position and size
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2 = min(x1 + np.random.randint(50, 300), width)
        y2 = min(y1 + np.random.randint(50, 300), height)
        
        # Random color
        color = np.random.uniform(0, 1, 3)
        
        # Fill rectangle
        img[y1:y2, x1:x2] = color
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Policy gRPC Client with MoveJ")
    parser.add_argument("--server", default="localhost:50051", help="Server address")
    parser.add_argument("--wrist_video", default="/Users/jack/Desktop/dummy_ctrl/datasets/pick_place_0406/videos/chunk-000/observation.images.cam_wrist/episode_000000.mp4", help="Path to wrist camera video file (for image=mp4)")
    parser.add_argument("--head_video", default="/Users/jack/Desktop/dummy_ctrl/datasets/pick_place_0406/videos/chunk-000/observation.images.cam_head/episode_000000.mp4", help="Path to head camera video file (for image=mp4)")
    parser.add_argument("--parquet", default="/Users/jack/Desktop/dummy_ctrl/datasets/pick_place_0406/data/chunk-000/episode_000000.parquet", help="Path to parquet file with state data (for state=parquet)")
    parser.add_argument("--frame_start", type=int, default=0, help="Frame index to start (for image=mp4)")
    parser.add_argument("--frame_end", type=int, default=-1, help="Frame index to end, -1 for all frames (for image=mp4)")
    parser.add_argument("--serial_number", default="396636713233", help="Serial number of the follower arm")
    parser.add_argument("--camera_wrist", default="http://192.168.237.249:8080/?action=stream", help="Wrist camera URL (for image=real)")
    parser.add_argument("--camera_head", default="http://192.168.237.157:8080/?action=stream", help="Head camera URL (for image=real)")
    parser.add_argument("--wrist_resolution", default="1280x720", help="Wrist camera resolution (WxH) (for image=real)")
    parser.add_argument("--head_resolution", default="1280x720", help="Head camera resolution (WxH) (for image=real)")
    parser.add_argument("--image", default="mp4", choices=["real", "mp4", "random"], help="Image source: real=cameras, mp4=video files, random=generated images")
    parser.add_argument("--state", default="parquet", choices=["real", "parquet", "random"], help="State source: real=from robot, parquet=from file, random=generated")
    args = parser.parse_args()
    
    # Initialize robot arm
    logger_fibre = fibre.utils.Logger(verbose=True)
    follower_arm = fibre.find_any(serial_number=args.serial_number, logger=logger_fibre)
    follower_arm.robot.resting()
    follower_arm.robot.move_j(0,0,90,0,0,0)
    joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
    follower_arm.robot.set_enable(True)
    arm_controller = ArmAngle(None, follower_arm, joint_offset)
    
    # Create client
    client = PolicyClient(args.server)
    
    # Check server health
    health_status = client.health_check()
    logger.info(f"Server health: {health_status}")
    
    # Get model info
    model_info = client.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Initialize streams for real camera option
    stream_wrist = None
    stream_head = None
    if args.image == "real":
        # Parse camera resolutions
        wrist_width, wrist_height = map(int, args.wrist_resolution.split('x'))
        wrist_resolution = (wrist_width, wrist_height)
        
        head_resolution = None
        if args.head_resolution:
            head_width, head_height = map(int, args.head_resolution.split('x'))
            head_resolution = (head_width, head_height)
        
        # Initialize wrist camera
        logger.info("Initializing wrist camera stream...")
        stream_wrist = VideoStream(url=args.camera_wrist, resolution=wrist_resolution).start()
        
        # Initialize head camera if URL provided
        if args.camera_head:
            logger.info("Initializing head camera stream...")
            stream_head = VideoStream(url=args.camera_head, resolution=head_resolution).start()
        
        # Wait for streams to initialize
        time.sleep(2.0)
    
    # Load DataFrame for parquet state option
    df = None
    if args.state == "parquet" or args.image == "mp4":
        df = pd.read_parquet(args.parquet)
    # Wait for 5 seconds after initial move_j
    logger.info("Waiting for 5 seconds after initial move_j...")
    time.sleep(5)  
    try:
        # Determine number of frames for processing
        total_frames = 1
        end_frame = 0
        
        if args.image == "mp4":
            # Get total number of frames from wrist video
            total_frames = get_total_frames(args.wrist_video)
            logger.info(f"Total frames in wrist video: {total_frames}")
            
            # Determine end frame
            end_frame = args.frame_end
            if end_frame == -1 or end_frame > total_frames:
                end_frame = total_frames - 1
                
            logger.info(f"Processing frames from {args.frame_start} to {end_frame}")
            
            # Check if head video is provided
            has_head_video = args.head_video and args.head_video.strip()
            if has_head_video:
                head_frames = get_total_frames(args.head_video)
                logger.info(f"Total frames in head video: {head_frames}")
                if head_frames < total_frames:
                    logger.warning(f"Head video has fewer frames ({head_frames}) than wrist video ({total_frames})")
                    end_frame = min(end_frame, head_frames - 1)
        else:
            # For real or random images, just set a single frame range for the loop
            end_frame = 100  # Process 100 frames for real or random modes
        
        # Statistics tracking
        total_server_time = 0.0
        total_round_trip_time = 0.0
        inference_count = 0
        
        # Process frames in range
        for frame_idx in range(args.frame_start if args.image == "mp4" else 0, end_frame + 1):
            logger.info(f"Processing frame {frame_idx}/{end_frame}")

            # Get images based on the selected source
            image_wrist = None
            image_head = None
            
            if args.image == "mp4":
                # Get images from video files
                image_wrist, _ = get_observation_from_video(args.wrist_video, frame_idx)
                
                has_head_video = args.head_video and args.head_video.strip()
                if has_head_video:
                    image_head, _ = get_observation_from_video(args.head_video, frame_idx)
            
            elif args.image == "real":
                # Get images from camera streams
                image_wrist, image_head = get_observation_from_streams(stream_wrist, stream_head)
            
            elif args.image == "random":
                # Generate random images
                image_wrist = generate_random_image(height=720, width=1280)
                
                # Also generate a random head image
                image_head = generate_random_image(height=720, width=1280)
            
            # Get state based on the selected source
            state = None
            if args.state == "parquet":
                # Get state from parquet file
                if df is not None:
                    state = df.iloc[frame_idx]['observation.state'].copy().tolist()
                else:
                    raise ValueError("DataFrame not loaded for parquet state")
            
            elif args.state == "real":
                # Get state from robot arm
                follow_joints = arm_controller.get_follow_joints()
                gripper = follower_arm.robot.hand.angle
                state = follow_joints.tolist() + [gripper]
            
            elif args.state == "random":
                # Generate random state
                state = generate_random_state()
            
            # Time the entire process
            start_time = time.perf_counter()
            
            # Make prediction
            prediction, inference_time_ms = client.predict(image_wrist, image_head, state)
            
            # Calculate round-trip time
            end_time = time.perf_counter()
            round_trip_ms = (end_time - start_time) * 1000
            
            # Update statistics
            total_server_time += inference_time_ms
            total_round_trip_time += round_trip_ms
            inference_count += 1
            
            # Get ground truth for comparison if available
            gt = None
            if args.state == "parquet" and df is not None:
                gt = df.iloc[frame_idx]['action']
            
            # Move robot arm based on prediction
            if prediction:
                # Skip the first 10 frames - don't move the robot
                if frame_idx >= 10:
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
                else:
                    logger.info(f"Skipping robot movement for frame {frame_idx} (first 10 frames)")
            
            # Display results
            logger.info(f"Frame {frame_idx} - Server time: {inference_time_ms:.2f}ms, Round-trip: {round_trip_ms:.2f}ms")
            logger.info(f"Prediction: {prediction}")
            
            if gt is not None:
                logger.info(f"Ground truth: {gt.tolist()}")
                logger.info(f"Difference: {np.round(np.array(prediction) - np.array(gt.tolist()), 2)}")
            
            # Add small delay between frames for real and random modes
            if args.image != "mp4":
                time.sleep(0.1)
                
            # Break loop if using only one frame or if Ctrl+C is pressed
            if args.image != "mp4" and frame_idx >= 100:
                break
            
        # Display summary statistics
        if inference_count > 0:
            logger.info("\n--- Performance Summary ---")
            logger.info(f"Processed {inference_count} frames")
            logger.info(f"Average server inference time: {total_server_time / inference_count:.2f}ms")
            logger.info(f"Average round-trip time: {total_round_trip_time / inference_count:.2f}ms")
            logger.info(f"Total processing time: {total_round_trip_time / 1000:.2f}s")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up resources
        if stream_wrist:
            stream_wrist.stop()
        if stream_head:
            stream_head.stop()
        
        follower_arm.robot.resting()
        client.close()


if __name__ == "__main__":
    main() 