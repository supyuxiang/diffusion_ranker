#!/usr/bin/env bash
# 依次运行 dit_random_search_ranker.py，num_candidates = 2..8。
#
# 用法:
#   cd /data/home/yxfeng/diffusion_ranker/search_algoriths
#   CUDA_VISIBLE_DEVICES=0 ./run_ranker_num_candidates_sweep.sh
#
# 额外参数会传给每一次 python 调用，例如:
#   ./run_ranker_num_candidates_sweep.sh --num_samples 1000 --deno_steps 10 --batch_size 16
#
# 环境变量（可选）:
#   RANKER_DIR, DIT_PATH, IMAGENET_VAL, BASE_OUT, SEED

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=6

RANKER_DIR='/data/home/yxfeng/diffusion_ranker/outputs/ranker_mse_kl/g32_alpha0.3_layer_ratio0.6_lr1e-4_cosine_0328'
DIT_PATH="${DIT_PATH:-/data/yxfeng/models/dit_model}"
IMAGENET_VAL="${IMAGENET_VAL:-/data/yxfeng/datasets/imagenet/val}"
BASE_OUT="${BASE_OUT:-/data/yxfeng/eval_dit_random_search_ranker_nc_sweep}"
SEED="${SEED:-42}"

mkdir -p "$BASE_OUT"

SUMMARY="$BASE_OUT/sweep_summary.jsonl"
: >"$SUMMARY"

for k in 2 4 8 16; do
  out="${BASE_OUT}/num_candidates_${k}_ckpt_0045_cfg1.0"
  echo "========== num_candidates=${k} -> ${out} =========="
  python3 "${SCRIPT_DIR}/dit_random_search_ranker.py" \
    --ranker_dir "$RANKER_DIR" \
    --ckpt_name "ranker_diffusion_epoch_0045.pt" \
    --dit_path "$DIT_PATH" \
    --imagenet_val "$IMAGENET_VAL" \
    --output_dir "$out" \
    --num_candidates "$k" \
    --seed "$SEED" \
    --fp16 \
    --guidance_scale 1.0 \
    --batch_size 32 \
    --num_samples 50000 \
    --init_random_samples 32 \
    --deno_steps 10 
    "$@"

  if [[ -f "${out}/final_metrics.json" ]]; then
    python3 -c "
import json
from pathlib import Path
p = Path('$out') / 'final_metrics.json'
row = {'num_candidates': $k, 'output_dir': '$out'}
row.update(json.loads(p.read_text()))
print(json.dumps(row, ensure_ascii=False))
" >>"$SUMMARY"
  fi
done

echo "全部完成。汇总行 JSONL: $SUMMARY"
