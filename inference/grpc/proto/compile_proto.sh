#!/bin/bash

# Compile the protobuf file to generate Python code
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. policy.proto

echo "Protocol buffer files compiled successfully."
echo "You may need to restart your client and server applications." 