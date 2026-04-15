#!/usr/bin/env bash
set -euo pipefail

PROMPTS_JSONL="${PROMPTS_JSONL:-/root/diffusion_ranker/text2image/data/prompts/example_prompts.jsonl}"
POOL_ROOT="${POOL_ROOT:-/root/autodl-tmp/t2i_runs/candidate_pool}"
TEACHER_SCORE_DIR="${TEACHER_SCORE_DIR:-/root/autodl-tmp/t2i_runs/scores_imagereward}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/t2i_runs/ranker}"

TRAINSET_PT="${TRAINSET_PT:-${OUT_ROOT}/trainset.pt}"
CKPT_DIR="${CKPT_DIR:-${OUT_ROOT}/ckpt}"
EVAL_OUT="${EVAL_OUT:-${OUT_ROOT}/selected_by_ranker}"

python /root/diffusion_ranker/text2image/ranker/build_ranker_trainset.py \
  --pool_root "${POOL_ROOT}" \
  --teacher_score_dir "${TEACHER_SCORE_DIR}" \
  --output_pt "${TRAINSET_PT}"

python /root/diffusion_ranker/text2image/ranker/train_ranker.py \
  --train_pt "${TRAINSET_PT}" \
  --output_dir "${CKPT_DIR}" \
  --num_epochs 10 \
  --batch_size 256 \
  --lr 1e-4

python /root/diffusion_ranker/text2image/ranker/eval_ranker.py \
  --prompts_jsonl "${PROMPTS_JSONL}" \
  --pool_root "${POOL_ROOT}" \
  --ranker_ckpt "${CKPT_DIR}/ranker_final.pt" \
  --prompt_table_pt "${CKPT_DIR}/prompt_table.pt" \
  --output_dir "${EVAL_OUT}"
