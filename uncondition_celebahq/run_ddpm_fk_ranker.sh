#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition_celebahq/ddpm_fk_ranker.py \
  --ranker_ckpt "${RANKER_CKPT:-/root/autodl-tmp/ddpm_ranker_ckpt_deno10/ranker_final.pt}" \
  --output_dir "${OUT_DIR:-/root/autodl-tmp/ddpm_fk_ranker_difference_ps10_deno10}" \
  --candidates_root "${CAND_ROOT:-/root/autodl-tmp/ddpm_candidates_50k_deno10}" \
  --num_candidates "${NUM_CAND:-10}" \
  --num_samples "${NUM_SAMPLES:-50000}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --potential "${POTENTIAL:-difference}" \
  --tau "${TAU:-0.1}" \
  --pick "${PICK:-argmax}" \
  --real_dir "${REAL_DIR:-/root/autodl-tmp/celebahq256_png_30k_flat}" \
  --compute_clipscore \
  --clip_prompt "${CLIP_PROMPT:-A photo of a human face.}"
