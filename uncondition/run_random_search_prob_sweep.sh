#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROBE_CKPT="${PROBE_CKPT:-/root/autodl-tmp/probe_model/s24_l19/hd128_lr1e-4_wd1e-4/probe_best.pt}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/uncond_random_search_probe_sweep_final}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/google-ddpm-ema-celebahq-256}"
REAL_DIR="${REAL_DIR:-/root/autodl-tmp/celebahq256_png}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"
LAYER_IDX="${LAYER_IDX:-19}"
STEP_IDX="${STEP_IDX:-24}"
POOL_KERNEL="${POOL_KERNEL:-2}"
POOL_STRIDE="${POOL_STRIDE:-2}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-fp16}"
NUM_LAYERS="${NUM_LAYERS:-2}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
INPUT_CHANNELS="${INPUT_CHANNELS:-256}"
DROPOUT="${DROPOUT:-0.1}"
CLIP_MODEL_DIR="${CLIP_MODEL_DIR:-/root/autodl-tmp/openai_clip-vit-large-patch14}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-64}"

mkdir -p "${OUT_ROOT}"

echo "[info] probe_ckpt=${PROBE_CKPT}"
echo "[info] out_root=${OUT_ROOT}"

for PS in 2 4 8; do
  OUT_DIR="${OUT_ROOT}/ps${PS}"
  echo "[run] parallel_size=${PS} -> ${OUT_DIR}"
  python /root/diffusion_ranker/uncondition/random_search_prob_reward.py \
    --probe_ckpt "${PROBE_CKPT}" \
    --output_dir "${OUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --real_dir "${REAL_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --parallel_size "${PS}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --layer_idx "${LAYER_IDX}" \
    --step_idx "${STEP_IDX}" \
    --pool_kernel "${POOL_KERNEL}" \
    --pool_stride "${POOL_STRIDE}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --num_layers "${NUM_LAYERS}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --input_channels "${INPUT_CHANNELS}" \
    --dropout "${DROPOUT}" \
    --clip_model_dir "${CLIP_MODEL_DIR}" \
    --clip_batch_size "${CLIP_BATCH_SIZE}" \
    "$@"
  echo "[done] parallel_size=${PS}"
done

echo "[all done] results saved under ${OUT_ROOT}"
