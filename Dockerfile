# syntax=docker/dockerfile:1

# Base image with CUDA, PyTorch, and Deepspeed support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install unzip and clean up to reduce image size
RUN apt-get update && apt-get install -y unzip git && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip \
    && pip install deepspeed tensorflow-datasets

# Set working directory
WORKDIR /workspace

# Copy the zipped project
COPY src.zip /workspace/

# Make unzip step rerunnable:
#  - Remove any old extracted folder before unzipping
#  - Remove any .git folders if they exist
RUN rm -rf src \
    && unzip -o src.zip \
    && find src -name ".git" -type d -exec rm -rf {} + \
    && rm -f src.zip

# Ensure the entry script is executable
RUN chmod +x /workspace/src/train_entry.sh

# Default entrypoint: run the training script
ENTRYPOINT ["/bin/bash", "/workspace/src/train_entry.sh"]

