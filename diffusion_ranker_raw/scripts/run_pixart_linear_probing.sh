#!/bin/bash
# PixArt Linear Probing 实验
# 8 prompts × 200 samples = 1600 样本，全 28 层
# guidance_scale=1 节省显存
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} conda run -n diffusion python3 pixart_linear_probing.py \
    --model_dir /data/yxfeng/models/pixart_sigma \
    --output_dir pixart_linear_probing_output \
    --samples_per_prompt 200 \
    --num_inference_steps 28 \
    --height 256 --width 256 \
    --guidance_scale 1.0 \
    --cpu_offload
