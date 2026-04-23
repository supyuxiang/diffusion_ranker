#!/bin/bash
# DiT 250-step baseline: 生成 50k 张图 + 计算 FID/IS
cd /data/home/yxfeng/diffusion_ranker

CUDA_VISIBLE_DEVICES=5 python3 scripts/eval/eval_dit_scaling_laws_rollout_level.py \
    --dit_path /data/yxfeng/models/dit_model \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --output_dir /data/yxfeng/eval_dit_250step_baseline \
    --num_samples 50000 \
    --batch_size 16 \
    --deno_steps 250 \
    --guidance_scale 1.5 \
    --fp16 \
    --seed 42
