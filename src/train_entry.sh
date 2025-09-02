#!/bin/bash
set -e

NUM_GPUS=1

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
else
  GPU_COUNT=$(python -c "import sys
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)" 2>/dev/null || echo 0)
fi

GPU_COUNT=${GPU_COUNT:-0}
GPU_COUNT=$(echo "$GPU_COUNT" | tr -dc '0-9')

if [ "$GPU_COUNT" -gt 1 ]; then
  NUM_GPUS=2
fi

echo "Detected $GPU_COUNT GPU(s) — launching with $NUM_GPUS GPU(s)."

exec deepspeed --num_gpus="$NUM_GPUS" src/scripts/train_hnet-deepspeed.py \
  --hnet_config configs/hnet-tiny_config.json \
  --steps 12000 --print_every 200 --save_dir checkpoints_tiny \
  --use_gptneox_inside_hnet
