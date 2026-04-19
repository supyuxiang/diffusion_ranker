#!/usr/bin/env bash
# End-to-end pipeline for CIFAR10 saturation-reward experiments:
# 1) rollout hidden+reward shards (EMA DDPM, saturation reward)
# 2) train ranker (ps=8)
# 3) random search generation
# 4) compute generated-image quality metrics + read FID/IS
#
# Usage:
#   bash /root/diffusion_ranker/uncondition_cifar10/pipeline.sh
#
# Optional overrides (examples):
#   STEP_IDX=600 NUM_SAMPLES_ROLLOUT=1500 EPOCHS=100 LR=1e-3 \
#   NUM_SAMPLES_RS=10000 BS_RS=2 PARALLEL_SIZE=8 \
#   bash /root/diffusion_ranker/uncondition_cifar10/pipeline.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO="${REPO:-/root/diffusion_ranker}"
SCRIPT_DIR="${REPO}/uncondition_cifar10"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/ddpm_ema_cifar10}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/cifar-10_from_autodl_pub}"

# ---------- rollout ----------
NUM_SAMPLES_ROLLOUT="${NUM_SAMPLES_ROLLOUT:-1500}"
BS_ROLLOUT="${BS_ROLLOUT:-256}"
NUM_STEPS="${NUM_STEPS:-1000}"
MC_SAMPLES="${MC_SAMPLES:-5}"
SEED="${SEED:-42}"
REWARD_TYPE="${REWARD_TYPE:-saturation}"

ROLLOUT_SAVE="${ROLLOUT_SAVE:-/root/autodl-tmp/cifar10_ddpm_saturation_hiddens_and_rewards_ema.pth}"
ROLLOUT_SHARDS="${ROLLOUT_SHARDS:-/root/autodl-tmp/cifar10_ddpm_saturation_hiddens_and_rewards_ema_shards}"

# ---------- train ranker ----------
STEP_IDX="${STEP_IDX:-600}"
LAYER_IDX="${LAYER_IDX:-12}"
PS="${PS:-8}"
EPOCHS="${EPOCHS:-500}"
BS_TRAIN="${BS_TRAIN:-32}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"

TRAIN_SAVE_DIR="${TRAIN_SAVE_DIR:-/root/autodl-tmp/cifar10_ranker_trans_saturation_ema/s${STEP_IDX}_l${LAYER_IDX}_ps${PS}_hd${HIDDEN_DIM}_lr${LR}_ep${EPOCHS}}"
RANKER_CKPT="${RANKER_CKPT:-${TRAIN_SAVE_DIR}/ranker_best.pt}"

# ---------- random search ----------
NUM_SAMPLES_RS="${NUM_SAMPLES_RS:-100}"
BS_RS="${BS_RS:-2}"
PARALLEL_SIZE="${PARALLEL_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/cifar10_random_search_ranker_trans_saturation_ema/s${STEP_IDX}_l${LAYER_IDX}_ps${PS}_hd${HIDDEN_DIM}_lr${LR}_ep${EPOCHS}_${NUM_SAMPLES_RS}_bs${BS_RS}}"

echo "[1/4] Ensure CIFAR-10 local root exists: ${DATA_ROOT}"
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "/root/diffusion_ranker")
from uncondition_cifar10.cifar10_scaling_with_different_steps import _resolve_cifar10_torchvision_root

root = _resolve_cifar10_torchvision_root(
    Path("/root/autodl-pub/cifar-10"),
    Path("/root/autodl-tmp/cifar-10_from_autodl_pub"),
)
print(f"[info] CIFAR root ready: {root}")
PY

echo "[2/4] Rollout with reward_type=${REWARD_TYPE}"
python "${SCRIPT_DIR}/cifar10_ddpm_rollout.py" \
  --model_path "${MODEL_PATH}" \
  --reward_type "${REWARD_TYPE}" \
  --save_path "${ROLLOUT_SAVE}" \
  --num_samples "${NUM_SAMPLES_ROLLOUT}" \
  --batch_size "${BS_ROLLOUT}" \
  --num_inference_steps "${NUM_STEPS}" \
  --mc_samples "${MC_SAMPLES}" \
  --seed "${SEED}"

echo "[3/4] Train ranker (ps=${PS}, step_idx=${STEP_IDX}, layer_idx=${LAYER_IDX})"
python "${SCRIPT_DIR}/train_ranker_trans.py" \
  --data_path "${ROLLOUT_SHARDS}" \
  --step_idx "${STEP_IDX}" \
  --layer_idx "${LAYER_IDX}" \
  --num_candidates "${PS}" \
  --num_epochs "${EPOCHS}" \
  --batch_size "${BS_TRAIN}" \
  --learning_rate "${LR}" \
  --weight_decay "${WD}" \
  --optimizer adamw \
  --scheduler cosine \
  --num_workers 4 \
  --train_ratio 0.9 \
  --split_seed 42 \
  --save_interval 10 \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_heads "${NUM_HEADS}" \
  --num_layers "${NUM_LAYERS}" \
  --input_channels 256 \
  --dropout "${DROPOUT}" \
  --save_dir "${TRAIN_SAVE_DIR}"

echo "[4/4] Random search generation"
python "${SCRIPT_DIR}/random_search_ranker_trans_reward.py" \
  --ranker_ckpt "${RANKER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --num_samples "${NUM_SAMPLES_RS}" \
  --batch_size "${BS_RS}" \
  --parallel_size "${PARALLEL_SIZE}" \
  --num_inference_steps "${NUM_STEPS}" \
  --layer_idx "${LAYER_IDX}" \
  --step_idx "${STEP_IDX}" \
  --seed "${SEED}" \
  --dtype fp16 \
  --num_layers "${NUM_LAYERS}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_heads "${NUM_HEADS}" \
  --input_channels 256 \
  --dropout "${DROPOUT}"

echo "[post] Compute image metrics on generated images"
python - <<PY
from pathlib import Path
import numpy as np
from PIL import Image
import json

img_dir = Path("${OUTPUT_DIR}")
paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".webp",".bmp"}])
if not paths:
    raise SystemExit(f"no images under {img_dir}")

brightness, contrast, saturation, colorfulness = [], [], [], []
for p in paths:
    rgb = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    y = 0.299*r + 0.587*g + 0.114*b
    brightness.append(float(y.mean()))
    contrast.append(float(y.std()))
    hsv = np.array(Image.fromarray(rgb.astype(np.uint8)).convert("HSV"), dtype=np.float32)
    saturation.append(float(hsv[...,1].mean()))
    rg = np.abs(r-g)
    yb = np.abs(0.5*(r+g)-b)
    colorfulness.append(float(np.sqrt(rg.std()**2 + yb.std()**2) + 0.3*np.sqrt(rg.mean()**2 + yb.mean()**2)))

def stats(a):
    a = np.array(a, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}

out = {
    "image_dir": str(img_dir),
    "num_images": len(paths),
    "metrics": {
        "brightness_luma_0_255": stats(brightness),
        "contrast_luma_std_0_255": stats(contrast),
        "saturation_hsv_s_0_255": stats(saturation),
        "colorfulness_hasler": stats(colorfulness),
    },
}
metrics_json = img_dir / "metrics.json"
if metrics_json.is_file():
    out["fid_is"] = json.loads(metrics_json.read_text())
save_path = img_dir / "image_quality_metrics.json"
save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
print(f"[done] image quality saved to: {save_path}")
PY

echo "[done] Pipeline finished."
echo "  rollout: ${ROLLOUT_SAVE}"
echo "  ranker_dir: ${TRAIN_SAVE_DIR}"
echo "  generated_dir: ${OUTPUT_DIR}"
