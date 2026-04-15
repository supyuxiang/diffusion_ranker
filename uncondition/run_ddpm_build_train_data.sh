#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/difusion_ranker/uncondition/ddpm_build_train_data.py \
  --candidates_root "${CAND_ROOT:-/root/autodl-tmp/ddpm_candidates_50k_deno10}" \
  --num_candidates "${NUM_CAND:-16}" \
  --max_per_candidate "${MAX_PER_CAND:-5000}" \
  --teacher "${TEACHER:-imagereward}" \
  --teacher_prompt "${TEACHER_PROMPT:-a photo of a human face}" \
  --output_pt "${TRAIN_PT:-/root/autodl-tmp/ddpm_ranker_train_data_deno10.pt}"
