#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PROMPTS_JSONL=/path/to/prompts.jsonl \
#   MODEL_PATH=/root/autodl-tmp/stable-diffusion-v1-5 \
#   OUT_ROOT=/root/autodl-tmp/t2i_runs \
#   bash text2image/scripts/run_all_baselines.sh

PROMPTS_JSONL="${PROMPTS_JSONL:-/root/diffusion_ranker/text2image/data/prompts/example_prompts.jsonl}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/stable-diffusion-v1-5}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/t2i_runs}"
POOL_ROOT="${POOL_ROOT:-${OUT_ROOT}/candidate_pool}"
SCORES_CLIP="${SCORES_CLIP:-${OUT_ROOT}/scores_clip}"
SCORES_IR="${SCORES_IR:-${OUT_ROOT}/scores_imagereward}"
BEST_ROOT="${BEST_ROOT:-${OUT_ROOT}/best_of_n}"

python /root/diffusion_ranker/text2image/generate/build_candidate_pool.py \
  --prompts_jsonl "${PROMPTS_JSONL}" \
  --output_root "${POOL_ROOT}" \
  --model_name_or_path "${MODEL_PATH}" \
  --num_candidates 64 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --batch_size 8

python /root/diffusion_ranker/text2image/reward/score_clip.py \
  --pool_root "${POOL_ROOT}" \
  --output_dir "${SCORES_CLIP}"

python /root/diffusion_ranker/text2image/reward/score_imagereward.py \
  --pool_root "${POOL_ROOT}" \
  --output_dir "${SCORES_IR}"

python /root/diffusion_ranker/text2image/search/best_of_n.py \
  --prompts_jsonl "${PROMPTS_JSONL}" \
  --pool_root "${POOL_ROOT}" \
  --score_dir "${SCORES_CLIP}" \
  --output_dir "${BEST_ROOT}/clip" \
  --n_values "1,2,4,8,10"

python /root/diffusion_ranker/text2image/search/best_of_n.py \
  --prompts_jsonl "${PROMPTS_JSONL}" \
  --pool_root "${POOL_ROOT}" \
  --score_dir "${SCORES_IR}" \
  --output_dir "${BEST_ROOT}/imagereward" \
  --n_values "1,2,4,8,10"
