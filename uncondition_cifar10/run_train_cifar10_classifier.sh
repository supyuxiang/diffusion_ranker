#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition_cifar10/train_cifar10_classifier.py \
  --data_root "${DATA_ROOT:-/root/autodl-tmp}" \
  --save_dir "${SAVE_DIR:-/root/autodl-tmp/cifar10_classifier/resnet18}" \
  --num_epochs "${NUM_EPOCHS:-100}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --learning_rate "${LEARNING_RATE:-0.1}" \
  --weight_decay "${WEIGHT_DECAY:-5e-4}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --seed "${SEED:-42}"

