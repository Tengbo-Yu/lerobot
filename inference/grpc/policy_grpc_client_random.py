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

import grpc
import numpy as np
import time
import argparse
import logging
import cv2

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
    
    def get_model_info(self):
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
    
    def predict(self, image_wrist, image_head, state):
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
            
            return response.prediction, response.inference_time_ms
        
        except grpc.RpcError as e:
            logger.error(f"Prediction failed: {e}")
            return [], 0.0
    
    def close(self):
        """Close the gRPC channel"""
        self.channel.close()


def generate_random_state():
    """
    Generate random joint angles and gripper angle
    Returns:
        List of 7 values: [joint1, joint2, joint3, joint4, joint5, joint6, gripper]
    """
    # Generate random joint angles between -180 and 180 degrees
    joints = np.random.uniform(-180, 180, 6)
    
    # Generate random gripper angle between -165 and 0 degrees
    gripper = np.random.uniform(-165, 0)
    
    # Combine into a single prediction
    prediction = np.concatenate([joints, [gripper]])
    
    return prediction.tolist()


def generate_random_image(height=720, width=1280, channels=3):
    """Generate a random RGB image"""
    # Option 1: Completely random image
    # return np.random.uniform(0, 1, (height, width, channels)).astype(np.float32)
    
    # Option 2: Random color blocks (more structured)
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
    parser = argparse.ArgumentParser(description="Random Client for Policy gRPC Server")
    parser.add_argument("--server", default="localhost:50051", help="Server address")
    parser.add_argument("--num_predictions", type=int, default=10, help="Number of predictions to generate")
    parser.add_argument("--delay_ms", type=float, default=500.0, help="Delay between predictions in milliseconds")
    parser.add_argument("--image_height", type=int, default=720, help="Generated image height")
    parser.add_argument("--image_width", type=int, default=1280, help="Generated image width")
    parser.add_argument("--single_camera", action="store_true", help="Use only wrist camera image")
    args = parser.parse_args()
    
    # Create client
    client = PolicyClient(args.server)
    
    # Check server health
    health_status = client.health_check()
    logger.info(f"Server health: {health_status}")
    
    # Get model info
    model_info = client.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    logger.info(f"Generating {args.num_predictions} random requests with {args.delay_ms}ms delay")
    
    try:
        for i in range(args.num_predictions):
            # Generate random state
            state = generate_random_state()
            
            # Generate random images
            image_wrist = generate_random_image(args.image_height, args.image_width)
            
            # Generate head camera image by default unless single_camera flag is set
            image_head = None
            if not args.single_camera:
                image_head = generate_random_image(args.image_height, args.image_width)
            
            # Log the input
            logger.info(f"Request {i+1}/{args.num_predictions}:")
            logger.info(f"  Random state: {[round(x, 2) for x in state]}")
            
            # Make prediction
            start_time = time.perf_counter()
            prediction, inference_time_ms = client.predict(image_wrist, image_head, state)
            end_time = time.perf_counter()
            
            # Calculate round-trip time
            round_trip_ms = (end_time - start_time) * 1000
            
            # Log the results
            logger.info(f"Response {i+1}/{args.num_predictions}:")
            if prediction:
                logger.info(f"  Prediction: {[round(x, 2) for x in prediction]}")
                logger.info(f"  Server inference time: {inference_time_ms:.2f}ms")
                logger.info(f"  Round-trip time: {round_trip_ms:.2f}ms")
            else:
                logger.error("  No prediction received from server")
            
            # Add delay between requests
            time.sleep(args.delay_ms / 1000.0)
            
    except KeyboardInterrupt:
        logger.info("Stopping random client")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main() 