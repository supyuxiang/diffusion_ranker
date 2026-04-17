#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition_cifar10/ddpm_cifar10_fid_protocol.py \
  --model_path "${MODEL_PATH:-/root/autodl-tmp/ddpm_ema_cifar10}" \
  --deno_steps "${DENO_STEPS:-1000}" \
  --num_gen "${NUM_GEN:-50000}" \
  --real_split "${REAL_SPLIT:-train}" \
  --batch_size "${BATCH_SIZE:-1024}" \
  --save_root "${SAVE_ROOT:-/root/autodl-tmp/outputs/ddpm_cifar10_fid_protocol}" \
  --data_root "${DATA_ROOT:-/root/autodl-tmp}" \
  --dtype "${DTYPE:-fp16}" \
  --seed "${SEED:-42}"
