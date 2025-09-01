# syntax=docker/dockerfile:1

# Base image with CUDA, PyTorch, and Deepspeed support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install unzip and git (still useful for pulling repos or debugging)
RUN apt-get update && apt-get install -y unzip git && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip \
    && pip install deepspeed tensorflow-datasets

# Set working directory
WORKDIR /workspace

# Copy the entire source directory directly from your GitHub repo
COPY src/ /workspace/src/

# Remove any .git folders to keep image clean
RUN find src -name ".git" -type d -exec rm -rf {} +

# Ensure the entry script is executable
RUN chmod +x /workspace/src/train_entry.sh

# Default entrypoint: run the training script
ENTRYPOINT ["/bin/bash", "/workspace/src/train_entry.sh"]
