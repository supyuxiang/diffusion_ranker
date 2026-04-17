#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition/ddpm_random_search_ranker.py \
  --ranker_ckpt "${RANKER_CKPT:-/root/autodl-tmp/ddpm_ranker_ckpt_deno10/ranker_final.pt}" \
  --output_dir "${OUT_DIR:-/root/autodl-tmp/ddpm_random_search_ranker_deno10}" \
  --candidates_root "${CAND_ROOT:-/root/autodl-tmp/ddpm_candidates_50k_deno10}" \
  --num_candidates "${NUM_CAND:-16}" \
  --num_samples "${NUM_SAMPLES:-50000}" \
  --search_batch_size "${SEARCH_BS:-32}" \
  --real_dir "${REAL_DIR:-/root/autodl-tmp/celebahq256_png}" \
  --compute_clipscore \
  --clip_prompt "${CLIP_PROMPT:-a photo of a human face}"
