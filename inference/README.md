# DummyAI Control - Inference Service

This directory contains implementation of the inference service for Dummy robot control, providing a gRPC interface for policy model inference.

## Overview

The inference service allows you to run a pre-trained policy model on a server and interact with it through gRPC. The system consists of two main components:

1. **gRPC Server**: Loads the policy model and processes prediction requests
2. **gRPC Client**: Connects to the server, sends observation data, and receives action predictions

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- gRPC
- fibre (for robot control)
- Additional dependencies in requirements.txt

## Getting Started

### 1. Start the gRPC Server

First, start the gRPC server which will load the policy model and listen for client requests:

```bash
cd dummy_ctrl
python inference/grpc/policy_grpc_server.py
```

The server will load the policy model from the specified checkpoint path and start listening on port 50051 (default).

### 2. Run the Client

Once the server is running, you can start the client to send prediction requests:

```bash
cd dummy_ctrl
python inference/grpc/policy_grpc_client.py --inference_time_s 60 --serial_number 396636713233 --control_rate 50
```

### Client Parameters

- `--inference_time_s`: How long to run the inference loop (default: 60 seconds)
- `--serial_number`: Serial number of the follower arm (default: 396636713233)
- `--control_rate`: Control rate in Hz (default: 50)
- `--server`: Server address (default: localhost:50051)
- `--camera_url`: Camera URL for streaming (default: http://192.168.65.124:8080/?action=stream)

## Features

- Real-time policy inference
- Support for compressed image transmission
- Health check and model information endpoints
- Configurable control rate
- Automatic connection to robot hardware

## Troubleshooting

- If the client can't connect to the server, ensure the server is running and check for firewall issues
- For robot connection problems, verify the serial number is correct and the robot is powered on
- For camera streaming issues, check the camera URL and ensure the stream is accessible
