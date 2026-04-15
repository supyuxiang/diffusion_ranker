#!/usr/bin/env bash
set -euo pipefail

# FK-Steering + ImageReward + DIFFERENCE potential, fixed parallel size=10

PY_ENTRY="/root/diffusion_ranker/fk_search/dit_imagereward.py"
if [[ ! -f "${PY_ENTRY}" ]]; then
  echo "missing entry: ${PY_ENTRY}"
  exit 1
fi

OUT_DIR="${1:-/root/autodl-tmp/fk_search_imagereward_difference/num_candidates_10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 "${PY_ENTRY}" \
  --output_dir "${OUT_DIR}" \
  --num_samples 50000 \
  --search_batch_size 32 \
  --num_candidates 10 \
  --tau 0.05 \
  --potential difference \
  --fk_resample \
  --pick sample

echo "done -> ${OUT_DIR}"

