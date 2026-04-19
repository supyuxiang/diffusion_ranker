#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_PATH="${DATA_PATH:-/root/autodl-tmp/ddpm_hiddens_and_rewards_shards}"
STEP_IDX="${STEP_IDX:-12}"
LAYER_IDX="${LAYER_IDX:-19}"
NUM_CANDIDATES="${NUM_CANDIDATES:-10}"
GROUP_SEED="${GROUP_SEED:-0}"

NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
INPUT_CHANNELS="${INPUT_CHANNELS:-256}"
DROPOUT="${DROPOUT:-0.1}"

ALPHA_KL="${ALPHA_KL:-0.3}"
ALPHA_MSE="${ALPHA_MSE:-0.7}"
TAU="${TAU:-1.0}"

NUM_EPOCHS="${NUM_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-cosine}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/trans_model/s12_l19/hd128_h8_lr1e-4_wd1e-4}"

python /root/diffusion_ranker/uncondition_celebahq/train_trans.py \
  --data_path "${DATA_PATH}" \
  --step_idx "${STEP_IDX}" \
  --layer_idx "${LAYER_IDX}" \
  --num_candidates "${NUM_CANDIDATES}" \
  --group_seed "${GROUP_SEED}" \
  --num_layers "${NUM_LAYERS}" \
  --num_heads "${NUM_HEADS}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --input_channels "${INPUT_CHANNELS}" \
  --dropout "${DROPOUT}" \
  --alpha_kl "${ALPHA_KL}" \
  --alpha_mse "${ALPHA_MSE}" \
  --tau "${TAU}" \
  --num_epochs "${NUM_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --optimizer "${OPTIMIZER}" \
  --scheduler "${SCHEDULER}" \
  --num_workers "${NUM_WORKERS}" \
  --train_ratio "${TRAIN_RATIO}" \
  --split_seed "${SPLIT_SEED}" \
  --save_interval "${SAVE_INTERVAL}" \
  --save_dir "${SAVE_DIR}" \
  "$@"
