# syntax=docker/dockerfile:1

# Base image with CUDA, PyTorch, and Deepspeed support
#FROM nvcr.io/nvidia/pytorch:23.10-py3
FROM nvcr.io/nvidia/pytorch:22.04-py3

# Install unzip and git (still useful for pulling repos or debugging)
RUN apt-get update && apt-get install -y unzip git && rm -rf /var/lib/apt/lists/*

# RUN DS_BUILD_OPS=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_CCL=0
# Upgrade pip and install required Python packages
RUN pip install --upgrade pip \
    && pip install deepspeed \
    && pip install tensorflow-datasets transformers onnx

# Set working directory
WORKDIR /workspace

# Copy the entire source directory directly from your GitHub repo
COPY src/ /workspace/src/

# Install the hnet package, which is a dependency for the training script
RUN pip install /workspace/src/hnet

# Remove any .git folders to keep image clean
RUN find src -name ".git" -type d -exec rm -rf {} +

# Ensure the entry script is executable
RUN chmod +x /workspace/src/train_entry.sh

# Default entrypoint: run the training script
ENTRYPOINT ["/bin/bash", "/workspace/src/train_entry.sh"]
