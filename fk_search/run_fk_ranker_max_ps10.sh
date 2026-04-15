#!/usr/bin/env bash
set -euo pipefail

# FK-Steering + ranker reward + MAX potential, fixed parallel size=10
#
# Usage:
#   bash fk_search/run_fk_ranker_max_ps10.sh <ranker_ckpt> <suite_image_root> <suite_repr_dir> [output_dir]

PY_ENTRY="/root/diffusion_ranker/fk_search/dit_ranker.py"
if [[ ! -f "${PY_ENTRY}" ]]; then
  echo "missing entry: ${PY_ENTRY}"
  exit 1
fi

RANKER_CKPT="${1:-}"
SUITE_IMAGE_ROOT="${2:-}"
SUITE_REPR_DIR="${3:-}"
OUT_DIR="${4:-/root/autodl-tmp/fk_search_ranker_max/num_candidates_10}"

if [[ -z "${RANKER_CKPT}" || -z "${SUITE_IMAGE_ROOT}" || -z "${SUITE_REPR_DIR}" ]]; then
  echo "usage: bash fk_search/run_fk_ranker_max_ps10.sh <ranker_ckpt> <suite_image_root> <suite_repr_dir> [output_dir]"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 "${PY_ENTRY}" \
  --output_dir "${OUT_DIR}" \
  --suite_image_root "${SUITE_IMAGE_ROOT}" \
  --suite_repr_dir "${SUITE_REPR_DIR}" \
  --ranker_ckpt "${RANKER_CKPT}" \
  --num_candidates 10 \
  --batch_size 32 \
  --num_samples 50000 \
  --n_init 1024 \
  --tau 0.5 \
  --potential max \
  --fk_resample \
  --pick sample

echo "done -> ${OUT_DIR}"

