#!/usr/bin/env bash
set -euo pipefail

# Generate ranker training data for:
# layer_ratio=0.6
# step_ratio in {0.0, 0.25, 0.5, 0.75}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

python3 /root/diffusion_ranker/class-condition/build_ranker_training_data_mc.py \
  --layer_ratio 0.6 \
  --step_ratios "0,0.25,0.5,0.75" \
  --base_output_dir /root/autodl-tmp/ranker_training_data_mc_class_condition \
  --num_inference_steps 10 \
  --guidance_scale 1.0 \
  --batch_size 256 \
  --mc 4 \
  --seed 42 \
  --dtype bf16 \
  "$@"
