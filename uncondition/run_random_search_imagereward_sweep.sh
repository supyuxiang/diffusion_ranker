#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/uncond_random_search_imagereward_sweep}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/google_ddpm-celebahq-256}"
REAL_DIR="${REAL_DIR:-/root/autodl-tmp/img_align_celeba}"
CLIP_MODEL_DIR="${CLIP_MODEL_DIR:-/root/autodl-tmp/openai_clip-vit-large-patch14}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-fp16}"

mkdir -p "${OUT_ROOT}"

echo "[info] out_root=${OUT_ROOT}"

for PS in 2 4 8; do
  OUT_DIR="${OUT_ROOT}/ps${PS}"
  echo "[run] ImageReward parallel_size=${PS} -> ${OUT_DIR}"
  python /root/diffusion_ranker/uncondition/random_search_imagereward_reward.py \
    --output_dir "${OUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --real_dir "${REAL_DIR}" \
    --clip_model_dir "${CLIP_MODEL_DIR}" \
    --clip_batch_size "${CLIP_BATCH_SIZE}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --parallel_size "${PS}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}"
  echo "[done] ImageReward parallel_size=${PS}"
done

echo "[all done] ImageReward sweep results under ${OUT_ROOT}"
