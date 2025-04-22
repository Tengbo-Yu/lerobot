from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import uvicorn
from typing import List, Dict, Any
import os
from lerobot.common.policies.act.modeling_act import ACTPolicy
import time
app = FastAPI(title="PyTorch Inference Server")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ACT policy model
# You can replace with your specific model path
PRETRAINED_POLICY_PATH = "/Users/yinzi/Downloads/Dummy_V2_workspace/dummy_ai/dummy_ctrl/checkpoints/train/cube_act_0326/100000/pretrained_model"

# Determine the device based on platform and availability
if torch.cuda.is_available():
    device = "cuda"  # Use CUDA for systems with NVIDIA GPUs
else:
    device = "cpu"  # Fallback to CPU

# gpu_id = os.environ.get('GPU_ID', '7')  # Default to GPU 0 if not specified
# torch.cuda.set_device(int(gpu_id))
# device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

try:
    policy = ACTPolicy.from_pretrained(PRETRAINED_POLICY_PATH)
    policy.to(device)
    print(f"Successfully loaded ACT policy from {PRETRAINED_POLICY_PATH}")
except Exception as e:
    policy = None
    raise e

class InferenceRequest(BaseModel):
    image: List[List[List[float]]]  # For 3D tensor: [3, h, w]
    state: List[float]  # For state vector: [7]

    class Config:
        schema_extra = {
            "example": {
                "image": [[[0.1] * 224] * 224] * 3,  # Example 3x224x224 tensor
                "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example state vector with 7 elements
            }
        }

class InferenceResponse(BaseModel):
    prediction: List[float]  # Predicted vector with shape [7]

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    try:
        # Convert the input data to PyTorch tensors
        image_tensor = torch.tensor(request.image, dtype=torch.float)
        state_tensor = torch.tensor([request.state], dtype=torch.float)  # Add batch dimension
        print(image_tensor.shape)
        print(state_tensor.shape)
        # Validate input shapes
        if image_tensor.shape[0] != 3:
            raise HTTPException(status_code=400, detail="Image must have 3 channels (RGB)")
        
        if state_tensor.shape != (1, 7):
            raise HTTPException(status_code=400, detail=f"State must have shape (1, 7), got {state_tensor.shape}")
        # print(policy)
        # Use the policy model for prediction if available
        if policy is not None:
            # Prepare tensors for the model
            # print("policy")
            image_tensor = image_tensor.to(device)
            state_tensor = state_tensor.to(device)
            
            # Create the policy input dictionary following the example in 2_evaluate_pretrained_policy.py
            observation = {
                "observation.state": state_tensor,
                "observation.images.cam_wrist": image_tensor.unsqueeze(0),  # Add batch dimension if needed
            }
            # print("observation")
            # Predict the next action with respect to the current observation
            # print("observation")
            # 添加时间计算
            start_time = time.perf_counter()
            action = policy.select_action(observation)
            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            print(f"推理时间: {inference_time:.2f}ms")
            # action = policy.select_action(observation)
            # Convert action to appropriate format
            prediction = action.squeeze(0).to("cpu")
            print("prediction: ", prediction)
        else:
            # Fallback to placeholder prediction
            prediction = torch.ones(1, 7)
        
        # Convert the prediction to a list for JSON serialization
        return InferenceResponse(prediction=prediction.tolist())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/model_info")
async def model_info():
    """Return information about the loaded model"""
    if policy is not None:
        return {
            "status": "loaded",
            "model": PRETRAINED_POLICY_PATH,
            "device": device,
            "input_features": str(policy.config.input_features) if hasattr(policy.config, "input_features") else "unknown",
            "output_features": str(policy.config.output_features) if hasattr(policy.config, "output_features") else "unknown"
        }
    else:
        return {
            "status": "not_loaded",
            "model": None,
            "device": device,
            "message": "Using placeholder prediction (ones tensor)"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
