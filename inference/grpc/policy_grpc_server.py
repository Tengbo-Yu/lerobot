import grpc
import torch
import numpy as np
import time
import os
import logging
from concurrent import futures
import sys
from typing import Dict, Any
import cv2

from proto import policy_pb2
from proto import policy_pb2_grpc

# Import the ACT policy model
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
# Set PyTorch seed for reproducibility
# seed = 1000
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the ACT policy model
# Update this path to your model location
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Policy gRPC Server")
parser.add_argument("--model_path", type=str, required=True,help="Path to the pretrained policy model")
parser.add_argument("--policy", type=str, required=True, help="Choose policy")
parser.add_argument("--target_resolution", type=str, default="720x1280", help="Target resolution for resizing images (HxW)")
parser.add_argument("--task", type=str, default="pick the cube into the box", help="Default task description for PI0/PI0fast policies")
args = parser.parse_args()
PRETRAINED_POLICY_PATH = args.model_path

# Parse target resolution
try:
    TARGET_HEIGHT, TARGET_WIDTH = map(int, args.target_resolution.split('x'))
    logger.info(f"Target resolution set to {TARGET_HEIGHT}x{TARGET_WIDTH}")
except Exception as e:
    logger.warning(f"Invalid resolution format: {args.target_resolution}. Using default 480x640")
    TARGET_HEIGHT, TARGET_WIDTH = 480, 640

# Determine the device based on platform and availability
if torch.cuda.is_available():
    device = "cuda"  
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Use Metal Performance Shaders for Mac with Apple Silicon
else:
    device = "cpu"  # Fallback to CPU

logger.info(f"Using device: {device}")

class PolicyServicer(policy_pb2_grpc.PolicyServiceServicer):
    def __init__(self):
        super().__init__()
        self.policy = None
        self.prediction_enabled = True  # Add this flag
        self.load_policy()
        
    def load_policy(self):
        """Load the policy model from the pretrained path"""
        try:
            logger.info(f"Loading policy from {PRETRAINED_POLICY_PATH}")
            if args.policy == "act":
                self.policy = ACTPolicy.from_pretrained(PRETRAINED_POLICY_PATH)
            elif args.policy == "diffusion":
                self.policy = DiffusionPolicy.from_pretrained(PRETRAINED_POLICY_PATH)
            elif args.policy == "pi0":
                self.policy = PI0Policy.from_pretrained(PRETRAINED_POLICY_PATH)
            elif args.policy == "pi0fast":
                self.policy = PI0FASTPolicy.from_pretrained(PRETRAINED_POLICY_PATH)
            self.policy.to(device)
            self.policy.eval()  # Set to evaluation mode
            self.policy.reset()  # Reset policy state
            logger.info(f"Successfully loaded policy")
        except Exception as e:
            self.policy = None
            # logger.error(f"Could not load ACT policy model: {e}")
            # logger.warning("Will use placeholder prediction (ones tensor)")
            raise e
    
    def reshape_image(self, flat_image, channels, height, width):
        """Reshape flat image data to tensor format [C, H, W]"""
        return torch.tensor(flat_image, dtype=torch.float).reshape(channels, height, width)
    
    def decode_image(self, encoded_image, image_format, height, width):
        """Decode compressed image data to tensor format [C, H, W]"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(encoded_image, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image with format {image_format}")
        
        # Resize image if needed
        if height != TARGET_HEIGHT or width != TARGET_WIDTH:
            img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
            logger.info(f"Resized image from {height}x{width} to {TARGET_HEIGHT}x{TARGET_WIDTH}")
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Convert to CHW format (channels, height, width)
        img = img.transpose(2, 0, 1)
        
        # Convert to tensor
        return torch.tensor(img, dtype=torch.float)
    
    def Predict(self, request, context):
        """Handle prediction requests"""
        try:
            if not self.prediction_enabled:
                logger.info("Prediction is currently disabled")
                
                return policy_pb2.PredictResponse(
                    prediction=[0.0] * 7,  # Return zeros when disabled
                    inference_time_ms=0.0
                )
            
            # Process first camera (cam_wrist)
            if request.encoded_image:
                logger.info(f"Received encoded image (cam_wrist) in {request.image_format} format, size: {len(request.encoded_image)/1024:.2f}KB")
                # Decode the encoded image
                image_wrist = self.decode_image(
                    request.encoded_image,
                    request.image_format,
                    request.image_height,
                    request.image_width
                )
            else:
                logger.info("No cam_wrist image received")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Missing cam_wrist image")
                return policy_pb2.PredictResponse()
                
            # Process second camera (cam_head) if available
            if hasattr(request, 'encoded_image2') and request.encoded_image2:
                logger.info(f"Received encoded image (cam_head) in {request.image_format} format, size: {len(request.encoded_image2)/1024:.2f}KB")
                # Decode the encoded image
                image_head = self.decode_image(
                    request.encoded_image2,
                    request.image_format,
                    request.image2_height,
                    request.image2_width
                )
                has_head_camera = True
            else:
                logger.info("No cam_head image received, using only cam_wrist")
                has_head_camera = False
            
            # Convert state data to tensor
            state_tensor = torch.tensor([request.state], dtype=torch.float)
            
            # Validate input shapes
            if image_wrist.shape[0] != 3:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Image must have 3 channels (RGB)")
                return policy_pb2.PredictResponse()
            
            if has_head_camera and image_head.shape[0] != 3:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Second image must have 3 channels (RGB)")
                return policy_pb2.PredictResponse()
            
            if state_tensor.shape != (1, 7):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"State must have shape (1, 7), got {state_tensor.shape}")
                return policy_pb2.PredictResponse()
            
            # Move tensors to device
            image_wrist = image_wrist.to(device)
            if has_head_camera:
                image_head = image_head.to(device)
            state_tensor = state_tensor.to(device)
            
            # Create the policy input dictionary
            observation = {
                "observation.state": state_tensor,
                "observation.images.cam_wrist": image_wrist.unsqueeze(0),  # Add batch dimension
            }
            
            # Add head camera if available
            if has_head_camera:
                # Check if the policy supports dual camera input
                supports_dual_camera = False
                if hasattr(self.policy.config, "input_features"):
                    if "observation.images.cam_head" in self.policy.config.input_features:
                        supports_dual_camera = True
                
                if supports_dual_camera:
                    observation["observation.images.cam_head"] = image_head.unsqueeze(0)
                    logger.info("Using dual camera mode (wrist + head)")
                else:
                    logger.warning("Second camera provided but model doesn't support dual camera mode. Using only wrist camera.")
            
            # Add task from request if provided and using PI0/PI0FAST policy
            if args.policy == "pi0" or args.policy == "pi0fast":
                if hasattr(request, 'task') and request.task:
                    observation["task"] = [request.task]
                    logger.info(f"Using task from client: '{request.task}'")
            
            if self.policy is not None:
                # Perform the prediction
                start_time = time.perf_counter()
                with torch.inference_mode():
                    action = self.policy.select_action(observation)
                end_time = time.perf_counter()
                inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
                logger.info(f"Inference time: {inference_time_ms:.2f}ms")
                logger.info(f"Used observations: {list(observation.keys())}")

                prediction = action.squeeze(0).to("cpu")
                logger.info(f"Prediction shape: {prediction.shape}")
            else:
                # Fallback to placeholder prediction
                prediction = torch.ones(7)
                
            
            # Create and return response
            response = policy_pb2.PredictResponse(
                prediction=prediction.tolist(),
                inference_time_ms=inference_time_ms
            )
            return response
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return policy_pb2.PredictResponse()
    
    def HealthCheck(self, request, context):
        """Handle health check requests"""
        return policy_pb2.HealthCheckResponse(status="healthy")
    
    def GetModelInfo(self, request, context):
        """Handle model info requests"""
        if self.policy is not None:
            # Check if the policy supports dual camera input by looking at the config
            supports_dual_camera = False
            if hasattr(self.policy.config, "input_features"):
                if "observation.images.cam_head" in self.policy.config.input_features:
                    supports_dual_camera = True
            
            camera_info = "Dual camera mode (wrist+head)" if supports_dual_camera else "Single camera mode (wrist only)"
            
            return policy_pb2.ModelInfoResponse(
                status="loaded",
                model_path=PRETRAINED_POLICY_PATH,
                device=device,
                input_features=str(self.policy.config.input_features) if hasattr(self.policy.config, "input_features") else "unknown",
                output_features=str(self.policy.config.output_features) if hasattr(self.policy.config, "output_features") else "unknown",
                message=f"Using image resolution: {TARGET_HEIGHT}x{TARGET_WIDTH}, {camera_info}"
            )
        else:
            return policy_pb2.ModelInfoResponse(
                status="not_loaded",
                model_path="",
                device=device,
                message="Using placeholder prediction (ones tensor)"
            )

    def SetPredictionEnabled(self, request, context):
        """Handle enabling/disabling prediction"""
        self.prediction_enabled = request.enabled
        return policy_pb2.SetPredictionEnabledResponse(status="success")


def serve():
    """Start the gRPC server"""
    # Create a gRPC server with 10 workers
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50MB
        ]
    )
    
    # Add the PolicyServicer to the server
    policy_pb2_grpc.add_PolicyServiceServicer_to_server(
        PolicyServicer(), server
    )
    
    # Listen on port 50051
    server.add_insecure_port('[::]:50051')
    server.start()
    
    logger.info("Policy gRPC server started on port 50051")
    logger.info(f"Using target resolution: {TARGET_HEIGHT}x{TARGET_WIDTH}")
    
    try:
        # Keep the server running until interrupted
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
