#!/bin/bash
# PixArt t-SNE 实验
# guidance_scale=1 节省显存（不编码 negative prompt）
# 6 prompts × 100 samples × 5 layers
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} conda run -n diffusion --no-capture-output python3 -u pixart_tsne_visualization.py \
    --model_dir /data/yxfeng/models/pixart_sigma \
    --output_dir pixart_tsne_output \
    --samples_per_prompt 100 \
    --layer_indices $(seq 0 1 27) \
    --num_inference_steps 28 \
    --height 256 --width 256 \
    --guidance_scale 1.0 \
    --max_tsne_samples 8000 \
    --pca_dims 50
