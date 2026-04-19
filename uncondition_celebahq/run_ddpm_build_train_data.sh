#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition_celebahq/ddpm_build_train_data.py \
  --candidates_root "${CAND_ROOT:-/root/autodl-tmp/ddpm_candidates_50k_deno10}" \
  --num_candidates "${NUM_CAND:-16}" \
  --max_per_candidate "${MAX_PER_CAND:-5000}" \
  --teacher "${TEACHER:-imagereward}" \
  --teacher_prompt "${TEACHER_PROMPT:-a photo of a human face}" \
  ${REPR_ROOT:+--repr_root "${REPR_ROOT}"} \
  ${REPR_KEY:+--repr_key "${REPR_KEY}"} \
  --output_pt "${TRAIN_PT:-/root/autodl-tmp/ddpm_ranker_train_data_deno10.pt}"
