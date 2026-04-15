#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_pointwise_ranker_by_ratio.sh 0.6 0.25
#
# Reads data from:
#   /root/autodl-tmp/ranker_training_data_mc/l{layer}_s{step_idx}/
# and writes checkpoint to:
#   /root/autodl-tmp/ranker_ckpt_pointwise_l{layer}_s{step_idx}/

LAYER_RATIO="${1:-0.6}"
STEP_RATIO="${2:-0.25}"

read -r STEP_IDX LAYER_TAG <<EOF
$(LAYER_RATIO="${LAYER_RATIO}" STEP_RATIO="${STEP_RATIO}" python3 - <<'PY'
import os
layer=float(os.environ["LAYER_RATIO"])
step=float(os.environ["STEP_RATIO"])
T=10
idx=min(T-1,max(0,round(step*(T-1))))
print(idx, f"{layer:g}".replace(".","p"))
PY
)
EOF

DATA_ROOT="/root/autodl-tmp/ranker_training_data_mc"
CASE_DIR="${DATA_ROOT}/l${LAYER_TAG}_s${STEP_IDX}"
REPR_PT="${CASE_DIR}/repr_reward.pt"
REWARD_JSON="${CASE_DIR}/reward_ls.json"
OUT_DIR="/root/autodl-tmp/ranker_ckpt_pointwise_l${LAYER_TAG}_s${STEP_IDX}"

if [[ ! -f "${REPR_PT}" ]]; then
  echo "missing ${REPR_PT}"
  exit 1
fi
if [[ ! -f "${REWARD_JSON}" ]]; then
  echo "missing ${REWARD_JSON}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python3 /root/diffusion_ranker/ranker/train.py \
  --num_epochs 10 \
  --output_dir "${OUT_DIR}" \
  --image_dir "${CASE_DIR}/images" \
  --candidate_repr_pt "${REPR_PT}" \
  --prompt_repr_pt /root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt \
  --reward_path "${REWARD_JSON}" \
  --batch_size 32 \
  --num_cand 10 \
  --input_dim_candidate 1152 \
  --input_dim_prompt 3584 \
  --hidden_dim 128 \
  --num_attention_heads 4 \
  --num_layers 8 \
  --dropout 0.1 \
  --optimizer adamw \
  --lr_scheduler cosine \
  --warmup_ratio 0.03 \
  --lr_min_ratio 0.01 \
  --alpha 0.8 \
  --tau 1.0 \
  --seed 42 \
  --num_candidate_groups 8 \
  --ranker_type pointwise \
  --has_prompt_hstates

echo "trained ckpt dir: ${OUT_DIR}"

