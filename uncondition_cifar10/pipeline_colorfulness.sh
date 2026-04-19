#!/usr/bin/env bash
# Colorfulness-reward end-to-end pipeline wrapper.
# It reuses pipeline.sh, but overrides defaults to colorfulness-specific paths/names.
#
# Usage:
#   bash /root/diffusion_ranker/uncondition_cifar10/pipeline_colorfulness.sh
#
# Optional overrides:
#   STEP_IDX=900 EPOCHS=500 LR=1e-3 NUM_SAMPLES_RS=500 BS_RS=20 \
#   bash /root/diffusion_ranker/uncondition_cifar10/pipeline_colorfulness.sh

set -euo pipefail

export REPO="${REPO:-/root/diffusion_ranker}"
BASE="${REPO}/uncondition_cifar10/pipeline.sh"

export REWARD_TYPE="${REWARD_TYPE:-colorfulness}"
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/ddpm_ema_cifar10}"
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/cifar-10_from_autodl_pub}"

# rollout defaults
export NUM_SAMPLES_ROLLOUT="${NUM_SAMPLES_ROLLOUT:-1500}"
export BS_ROLLOUT="${BS_ROLLOUT:-256}"
export NUM_STEPS="${NUM_STEPS:-1000}"
export MC_SAMPLES="${MC_SAMPLES:-5}"
export SEED="${SEED:-42}"
export ROLLOUT_SAVE="${ROLLOUT_SAVE:-/root/autodl-tmp/cifar10_ddpm_colorfulness_hiddens_and_rewards_ema.pth}"
export ROLLOUT_SHARDS="${ROLLOUT_SHARDS:-/root/autodl-tmp/cifar10_ddpm_colorfulness_hiddens_and_rewards_ema_shards}"

# training defaults
export STEP_IDX="${STEP_IDX:-900}"
export LAYER_IDX="${LAYER_IDX:-12}"
export PS="${PS:-8}"
export EPOCHS="${EPOCHS:-100}"
export BS_TRAIN="${BS_TRAIN:-32}"
export LR="${LR:-1e-3}"
export WD="${WD:-1e-4}"
export HIDDEN_DIM="${HIDDEN_DIM:-64}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_LAYERS="${NUM_LAYERS:-2}"
export DROPOUT="${DROPOUT:-0.1}"

export TRAIN_SAVE_DIR="${TRAIN_SAVE_DIR:-/root/autodl-tmp/cifar10_ranker_trans_colorfulness_ema/s${STEP_IDX}_l${LAYER_IDX}_ps${PS}_hd${HIDDEN_DIM}_lr${LR}_ep${EPOCHS}}"
export RANKER_CKPT="${RANKER_CKPT:-${TRAIN_SAVE_DIR}/ranker_best.pt}"

# random search defaults
export NUM_SAMPLES_RS="${NUM_SAMPLES_RS:-10000}"
export BS_RS="${BS_RS:-2}"
export PARALLEL_SIZE="${PARALLEL_SIZE:-8}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/cifar10_random_search_ranker_trans_colorfulness_ema/s${STEP_IDX}_l${LAYER_IDX}_ps${PS}_hd${HIDDEN_DIM}_lr${LR}_ep${EPOCHS}_${NUM_SAMPLES_RS}_bs${BS_RS}}"

bash "${BASE}"
