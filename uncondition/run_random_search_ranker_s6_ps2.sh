#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition/random_search_ranker_reward.py \
  --ranker_ckpt "${RANKER_CKPT:-/root/autodl-tmp/ranker_model/s6_l19/hd128_lr1e-4_ps2/ranker_best.pt}" \
  --output_dir "${OUT_DIR:-/root/autodl-tmp/uncond_random_search_ranker_s6_l19_hd128_lr1e-4_ps2}" \
  --model_path "${MODEL_PATH:-/root/autodl-tmp/google_ddpm-celebahq-256}" \
  --real_dir "${REAL_DIR:-/root/autodl-tmp/img_align_celeba}" \
  --num_samples "${NUM_SAMPLES:-50000}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --parallel_size "${PARALLEL_SIZE:-2}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-25}" \
  --layer_idx "${LAYER_IDX:-19}" \
  --step_idx "${STEP_IDX:-6}" \
  --pool_kernel "${POOL_KERNEL:-2}" \
  --pool_stride "${POOL_STRIDE:-2}" \
  --seed "${SEED:-42}" \
  --dtype "${DTYPE:-fp16}" \
  --num_layers "${NUM_LAYERS:-2}" \
  --hidden_dim "${HIDDEN_DIM:-128}" \
  --input_channels "${INPUT_CHANNELS:-256}" \
  --dropout "${DROPOUT:-0.1}" \
  --clip_model_dir "${CLIP_MODEL_DIR:-/root/autodl-tmp/openai_clip-vit-large-patch14}" \
  --clip_batch_size "${CLIP_BATCH_SIZE:-64}"
