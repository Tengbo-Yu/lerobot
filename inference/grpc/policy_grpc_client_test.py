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

from proto import policy_pb2
from proto import policy_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


def main():
    parser = argparse.ArgumentParser(description="Policy gRPC Client")
    parser.add_argument("--server", default="localhost:50051", help="Server address")
    parser.add_argument("--wrist_video", default="/Users/jack/lab_intern/dummy_ctrl/data/pick_place_0414/videos/chunk-000/observation.images.cam_wrist/episode_000054.mp4", help="Path to wrist camera video file")
    parser.add_argument("--head_video", default="/Users/jack/lab_intern/dummy_ctrl/data/pick_place_0414/videos/chunk-000/observation.images.cam_head/episode_000054.mp4", help="Path to head camera video file (optional)")
    parser.add_argument("--parquet", default="/Users/jack/lab_intern/dummy_ctrl/data/pick_place_0414/data/chunk-000/episode_000054.parquet", help="Path to parquet file with state data")
    parser.add_argument("--frame_start", type=int, default=0, help="Frame index to start")
    parser.add_argument("--frame_end", type=int, default=-1, help="Frame index to end (-1 for all frames)")
    parser.add_argument("--input", default="all", choices=["all", "image"], help="Input mode: 'all' uses dataset states for each frame, 'image' uses previous predictions")
    args = parser.parse_args()
    
    # Create client
    client = PolicyClient(args.server)
    
    # Check server health
    health_status = client.health_check()
    logger.info(f"Server health: {health_status}")
    
    # Get model info
    model_info = client.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Load DataFrame once
    df = pd.read_parquet(args.parquet)
    
    try:
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
        
        # Statistics tracking
        total_server_time = 0.0
        total_round_trip_time = 0.0
        inference_count = 0
        
        # For 'image' mode, we'll keep track of the current state
        if args.input == "image":
            current_state = df.iloc[args.frame_start]['observation.state'].copy().tolist()
            logger.info(f"Initial state: {[round(x, 2) for x in current_state]}")
            # For comparing prediction with ground truth
            all_predictions = []
            all_ground_truths = []
        
        # Process frames in range
        for frame_idx in range(args.frame_start, end_frame + 1):
            logger.info(f"Processing frame {frame_idx}/{end_frame}")

            # Get wrist image from video
            image_wrist, _ = get_observation_from_video(args.wrist_video, frame_idx)
            
            # Get head image from video if available
            image_head = None
            if has_head_video:
                image_head, _ = get_observation_from_video(args.head_video, frame_idx)
            
            # Get input state based on mode
            if args.input == "all":
                # Use state from the dataset for each frame
                state = df.iloc[frame_idx]['observation.state'].copy().tolist()
            else:  # args.input == "image"
                # Use current state (which is either initial or previous prediction)
                state = current_state
            
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
            
            # Handle results based on input mode
            if args.input == "all":
                # Display results with appropriate ground truth comparison (from inference_all)
                if frame_idx < end_frame:  # Make sure we're not at the last frame
                    # Get next state as ground truth for comparison
                    gt = df.iloc[frame_idx + 1]['observation.state'].copy().tolist()
                    
                    # Display results
                    logger.info(f"Frame {frame_idx} - Server time: {inference_time_ms:.2f}ms, Round-trip: {round_trip_ms:.2f}ms")
                    logger.info(f"Current state: {[round(x, 2) for x in state]}")
                    logger.info(f"Prediction: {[round(x, 2) for x in prediction]}")
                    logger.info(f"Next state (ground truth): {[round(x, 2) for x in gt]}")
                    logger.info(f"Difference (prediction vs next state): {np.round(np.array(prediction) - np.array(gt), 2)}")
                    logger.info("---")
                else:
                    # For the last frame, we don't have a next state to compare with
                    logger.info(f"Frame {frame_idx} - Server time: {inference_time_ms:.2f}ms, Round-trip: {round_trip_ms:.2f}ms")
                    logger.info(f"Current state: {[round(x, 2) for x in state]}")
                    logger.info(f"Prediction: {[round(x, 2) for x in prediction]}")
                    logger.info("Last frame - no next state available for comparison")
                    logger.info("---")
            else:  # args.input == "image"
                # Display results (from inference_img)
                ground_truth_state = df.iloc[frame_idx]['observation.state'].copy().tolist()
                
                # Store for analysis
                all_predictions.append(prediction)
                all_ground_truths.append(ground_truth_state)
                
                # Display results
                logger.info(f"Frame {frame_idx} - Server time: {inference_time_ms:.2f}ms, Round-trip: {round_trip_ms:.2f}ms")
                logger.info(f"Current state (input): {[round(x, 2) for x in current_state]}")
                logger.info(f"Prediction (next state): {[round(x, 2) for x in prediction]}")
                logger.info(f"Ground truth state: {[round(x, 2) for x in ground_truth_state]}")
                logger.info(f"Difference (prediction vs ground truth): {np.round(np.array(prediction) - np.array(ground_truth_state), 2)}")
                logger.info("---")
                
                # Update current state with prediction for next iteration
                current_state = prediction
            
        # Display summary statistics
        if inference_count > 0:
            logger.info("\n--- Performance Summary ---")
            logger.info(f"Input mode: {args.input}")
            logger.info(f"Processed {inference_count} frames")
            logger.info(f"Average server inference time: {total_server_time / inference_count:.2f}ms")
            logger.info(f"Average round-trip time: {total_round_trip_time / inference_count:.2f}ms")
            logger.info(f"Total processing time: {total_round_trip_time / 1000:.2f}s")
            
            # Add additional metrics for 'image' mode
            if args.input == "image" and len(all_predictions) > 0:
                # Convert lists to numpy arrays for easier calculation
                predictions_array = np.array(all_predictions)
                ground_truths_array = np.array(all_ground_truths)
                
                # Calculate mean absolute error for each joint
                mae = np.mean(np.abs(predictions_array - ground_truths_array), axis=0)
                
                logger.info(f"Mean absolute error per joint: {mae}")
                logger.info(f"Overall MAE: {np.mean(mae):.4f}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # Close client
    client.close()


if __name__ == "__main__":
    main() 