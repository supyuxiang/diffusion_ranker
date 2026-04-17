#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash /root/diffusion_ranker/uncondition_mnist/run_random_search_classifier_ps8.sh

CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-/root/autodl-tmp/mnist_classifier/resnet18/classifier_best.pt}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/1aurent_ddpm-mnist}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/mnist_random_search_classifier_ps8}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-fp16}"

python /root/diffusion_ranker/uncondition_mnist/random_search_classifier_reward.py \
  --classifier_ckpt "${CLASSIFIER_CKPT}" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --parallel_size 8 \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --seed "${SEED}" \
  --dtype "${DTYPE}"
