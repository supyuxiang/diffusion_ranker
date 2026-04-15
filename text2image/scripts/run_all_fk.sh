#!/usr/bin/env bash
set -euo pipefail

PROMPTS_JSONL="${PROMPTS_JSONL:-/root/diffusion_ranker/text2image/data/prompts/example_prompts.jsonl}"
POOL_ROOT="${POOL_ROOT:-/root/autodl-tmp/t2i_runs/candidate_pool}"
SCORES_CLIP="${SCORES_CLIP:-/root/autodl-tmp/t2i_runs/scores_clip}"
SCORES_IR="${SCORES_IR:-/root/autodl-tmp/t2i_runs/scores_imagereward}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/t2i_runs/fk}"

for reward in clip imagereward; do
  if [[ "${reward}" == "clip" ]]; then
    SCORE_DIR="${SCORES_CLIP}"
  else
    SCORE_DIR="${SCORES_IR}"
  fi

  for potential in difference max; do
    python /root/diffusion_ranker/text2image/search/fk_steering.py \
      --prompts_jsonl "${PROMPTS_JSONL}" \
      --pool_root "${POOL_ROOT}" \
      --score_dir "${SCORE_DIR}" \
      --output_dir "${OUT_ROOT}/${reward}_${potential}_ps10" \
      --parallel_size 10 \
      --tau 0.1 \
      --potential "${potential}" \
      --pick argmax
  done
done
