#!/usr/bin/env bash
set -euo pipefail

# One-click sweep:
# Run random search for ALL MLP ranker ckpts under /root/autodl-tmp/mnist_ranker_mlp
# with parallel_size=8 and batch_size=32.
#
# Optional overrides:
#   CKPT_ROOT=/root/autodl-tmp/mnist_ranker_mlp
#   OUT_ROOT=/root/autodl-tmp/mnist_random_search_ranker_mlp_ps8
#   NUM_SAMPLES=10000
#   NUM_INFERENCE_STEPS=100
#   DTYPE=fp16
#   SEED=42

CKPT_ROOT="${CKPT_ROOT:-/root/autodl-tmp/mnist_ranker_mlp}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/mnist_random_search_ranker_mlp_ps8}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/1aurent_ddpm-mnist}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
DTYPE="${DTYPE:-fp16}"
SEED="${SEED:-42}"

PARALLEL_SIZE=8
BATCH_SIZE=32

mkdir -p "${OUT_ROOT}"

shopt -s nullglob globstar
ckpts=( "${CKPT_ROOT}"/**/ranker_best.pt )
shopt -u globstar

if [ "${#ckpts[@]}" -eq 0 ]; then
  echo "[error] no ranker_best.pt found under ${CKPT_ROOT}"
  exit 1
fi

for ckpt in "${ckpts[@]}"; do
  ckpt_dir="$(dirname "${ckpt}")"
  rel="${ckpt_dir#${CKPT_ROOT}/}"

  # Parse step/layer from path like s0_l9/...
  step_idx="$(python - <<'PY' "${rel}"
import re,sys
s=sys.argv[1]
m=re.search(r's(\d+)_l(\d+)', s)
print(m.group(1) if m else '0')
PY
)"
  layer_idx="$(python - <<'PY' "${rel}"
import re,sys
s=sys.argv[1]
m=re.search(r's(\d+)_l(\d+)', s)
print(m.group(2) if m else '9')
PY
)"

  # Read model hyper-params from config.json if available.
  cfg_path="${ckpt_dir}/config.json"
  read -r num_layers hidden_dim input_channels dropout <<<"$(python - <<'PY' "${cfg_path}"
import json,sys,os
cfg_path=sys.argv[1]
defaults=(2,8,64,0.1)
if os.path.isfile(cfg_path):
    with open(cfg_path,'r',encoding='utf-8') as f:
        cfg=json.load(f)
    vals=(
      int(cfg.get('num_layers', defaults[0])),
      int(cfg.get('hidden_dim', defaults[1])),
      int(cfg.get('input_channels', defaults[2])),
      float(cfg.get('dropout', defaults[3])),
    )
else:
    vals=defaults
print(*vals)
PY
)"

  out_dir="${OUT_ROOT}/${rel}_ps8"
  mkdir -p "${out_dir}"

  echo "[run] ckpt=${ckpt}"
  echo "      step_idx=${step_idx} layer_idx=${layer_idx} num_layers=${num_layers} hidden_dim=${hidden_dim} input_channels=${input_channels} dropout=${dropout}"
  echo "      out_dir=${out_dir}"

  python /root/diffusion_ranker/uncondition_mnist/random_search_ranker_mlp_reward.py \
    --ranker_ckpt "${ckpt}" \
    --output_dir "${out_dir}" \
    --model_path "${MODEL_PATH}" \
    --data_root "${DATA_ROOT}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --parallel_size "${PARALLEL_SIZE}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --layer_idx "${layer_idx}" \
    --step_idx "${step_idx}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --num_layers "${num_layers}" \
    --hidden_dim "${hidden_dim}" \
    --input_channels "${input_channels}" \
    --dropout "${dropout}"
done

echo "[all done] finished ${#ckpts[@]} checkpoints"
