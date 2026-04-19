#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition_cifar10/cifar10_ddpm_rollout.py \
  --model_path "${MODEL_PATH:-/root/autodl-tmp/google_ddpm-cifar10-32}" \
  --classifier_ckpt "${CLASSIFIER_CKPT:-/root/autodl-tmp/cifar10_classifier/resnet18/classifier_best.pt}" \
  --save_path "${SAVE_PATH:-/root/autodl-tmp/cifar10_ddpm_hiddens_and_rewards.pth}" \
  --num_samples "${NUM_SAMPLES:-1500}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
  --mc_samples "${MC_SAMPLES:-5}" \
  --seed "${SEED:-42}"

