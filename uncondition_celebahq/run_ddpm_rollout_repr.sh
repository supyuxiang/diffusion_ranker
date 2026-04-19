#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition_celebahq/ddpm_rollout_repr.py \
  --model_name "${MODEL_NAME:-google_ddpm-celebahq-256}" \
  --num_candidates "${NUM_CAND:-16}" \
  --num_samples "${NUM_SAMPLES:-50000}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --num_inference_steps "${DENO_STEPS:-10}" \
  --layer_ratios "${LAYER_RATIOS:-0.1,0.3,0.6,0.9,1.0}" \
  --step_ratios "${STEP_RATIOS:-0.0,0.25,0.5,0.75,1.0}" \
  --output_image_root "${OUT_IMG_ROOT:-/root/autodl-tmp/ddpm_candidates_50k_deno10}" \
  --output_repr_root "${OUT_REPR_ROOT:-/root/autodl-tmp/ddpm_repr_candidates_deno10}" \
  --seed "${SEED:-42}" \
  --dtype "${DTYPE:-fp16}" \
  --store_dtype "${STORE_DTYPE:-float16}"
