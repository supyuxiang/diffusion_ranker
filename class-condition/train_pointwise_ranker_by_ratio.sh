#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash class-condition/train_pointwise_ranker_by_ratio.sh 0.6 0.25

LAYER_RATIO="${1:-0.6}"
STEP_RATIO="${2:-0.25}"

python3 /root/diffusion_ranker/class-condition/train.py \
  --layer_ratio "${LAYER_RATIO}" \
  --step_ratio "${STEP_RATIO}" \
  --data_root /root/autodl-tmp/ranker_training_data_mc_class_condition
