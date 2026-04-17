#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition_mnist/mnist_ddpm_rollout.py \
  --model_path "${MODEL_PATH:-/root/autodl-tmp/1aurent_ddpm-mnist}" \
  --classifier_ckpt "${CLASSIFIER_CKPT:-/root/autodl-tmp/mnist_classifier/resnet18/classifier_best.pt}" \
  --save_path "${SAVE_PATH:-/root/autodl-tmp/mnist_ddpm_hiddens_and_rewards.pth}" \
  --num_samples "${NUM_SAMPLES:-2000}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-100}" \
  --mc_samples "${MC_SAMPLES:-5}" \
  --seed "${SEED:-42}"
