#!/bin/bash
set -e

# Default: use 1 GPU (A100)
NUM_GPUS=1

# Detect how many NVIDIA GPUs are visible
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# If more than 1 GPU, limit to 2
if [ "$GPU_COUNT" -gt 1 ]; then
    NUM_GPUS=2
fi

echo "Detected $GPU_COUNT GPUs available — launching with $NUM_GPUS GPU(s)."

# Launch training with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS scripts/train_hnet-deepspeed.py \
    --hnet_config configs/hnet-tiny_config.json \
    --steps 12000 --print_every 200 --save_dir checkpoints_tiny

