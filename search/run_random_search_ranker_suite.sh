#!/usr/bin/env bash
# 候选目录约定与 dit_random_search_clipscore / rollout.py 一致；未传 suite 时由 Python 自动填默认路径。
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GS="${GS:-1.0}"
DENO="${DENO:-10}"

python3 "$(dirname "$0")/dit_random_search_ranker.py" \
  --ranker_ckpt "${RANKER_CKPT:?set RANKER_CKPT}" \
  --prompt_repr_pt "${PROMPT_REPR_PT:?set PROMPT_REPR_PT}" \
  --repr_key "${REPR_KEY:?set REPR_KEY}" \
  --guidance_scale "${GS}" \
  --deno_steps "${DENO}" \
  --step_ratios "${STEP_RATIOS:-0.1111111111111111}" \
  --layer_ratios "${LAYER_RATIOS:-0.6}" \
  --output_dir "${OUTPUT_DIR:-/root/autodl-tmp/rs_ranker_out}" \
  --num_samples "${NUM_SAMPLES:-5000}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --num_candidates "${NUM_CANDIDATES:-8}" \
  --precomputed_suite_num_candidates "${PRECOMPUTED_SUITE_NUM_CANDIDATES:-16}" \
  ${INIT_RANDOM_SAMPLES:+--init_random_samples "${INIT_RANDOM_SAMPLES}"} \
  "$@"
