# gRPC Protocol Documentation

## Compiling Protocol Buffers

When you make changes to the file transfer format in the `.proto` files, you need to recompile the protocol buffers using the following command:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. policy.proto
```

## Dual Camera Support

The system now supports dual camera input:
- `cam_wrist`: Primary camera mounted on the robot wrist
- `cam_head`: Secondary camera providing an additional perspective

### Server Usage

Run the server with target resolution for both cameras:

```bash
python policy_grpc_server.py --model_path /path/to/model --policy act --target_resolution 640x480
```

### Client Usage

Run the client with both cameras:

```bash
python policy_grpc_client.py --camera_wrist http://camera1:8080/?action=stream \
                            --camera_head http://camera2:8080/?action=stream \
                            --wrist_resolution 640x480 \
                            --head_resolution 640x480
```

Note: The model must support dual camera input. Check the model info response for dual camera support.
