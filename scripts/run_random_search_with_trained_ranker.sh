#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_random_search_with_trained_ranker.sh 0.6 0.25
#   bash scripts/run_random_search_with_trained_ranker.sh 0.6 0.25 /path/to/ckpt.pt

LAYER_RATIO="${1:-0.6}"
STEP_RATIO="${2:-0.25}"
CKPT_PATH="${3:-}"

read -r STEP_IDX LAYER_TAG REPR_KEY <<EOF
$(LAYER_RATIO="${LAYER_RATIO}" STEP_RATIO="${STEP_RATIO}" python3 - <<'PY'
import os
layer=float(os.environ["LAYER_RATIO"])
step=float(os.environ["STEP_RATIO"])
T=10
idx=min(T-1,max(0,round(step*(T-1))))
layer_s=f"{layer:g}"
layer_tag=layer_s.replace(".","p")
print(idx, layer_tag, f"s{idx}_r{layer_s}")
PY
)
EOF

if [[ -z "${CKPT_PATH}" ]]; then
  CKPT_PATH="/root/autodl-tmp/ranker_ckpt_pointwise_l${LAYER_TAG}_s${STEP_IDX}/ranker_epoch10.pt"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "missing ckpt: ${CKPT_PATH}"
  exit 1
fi

OUT_DIR="/root/autodl-tmp/eval_random_search_ranker_l${LAYER_TAG}_s${STEP_IDX}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 /root/diffusion_ranker/search/dit_random_search_ranker.py \
  --ranker_ckpt "${CKPT_PATH}" \
  --prompt_repr_pt /root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt \
  --repr_key "${REPR_KEY}" \
  --num_samples 50000 \
  --batch_size 40 \
  --num_candidates 10 \
  --precomputed_suite_image_root /root/autodl-tmp/dit_candidates_50k_cfg1.0_deno10 \
  --precomputed_suite_repr_dir /root/autodl-tmp/repr_candidate_deno10_cfg1.0/mean \
  --output_dir "${OUT_DIR}"

echo "random search done -> ${OUT_DIR}"

