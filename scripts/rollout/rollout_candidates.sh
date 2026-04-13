#!/usr/bin/env bash
# 多卡：每个 GPU 跑一个 --candidate_idx（与 search 的 num_candidates=K 一致）

set -euo pipefail


CUDA_VISIBLE_DEVICES=0 python3 /root/diffusion_ranker/search/rollout.py \
    --candidate_idx 0 \
    --deno_steps 10 \
    --guidance_scale 1.0 \
    --model_name dit-xl-2-256 \
    --num_samples 50000 \
    --batch_size 512 \
    --dtype bf16





