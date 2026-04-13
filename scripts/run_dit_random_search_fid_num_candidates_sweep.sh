#!/usr/bin/env bash
# Sweep num_candidates for dit_random_search_fid_reward.py (init 固定 N_INIT=1024，见脚本内常量).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for nc in 2 4 8 16; do
  echo "========== num_candidates=${nc} =========="
  python3 -u "${ROOT}/search/dit_random_search_fid_reward.py" \
    --imagenet_val /root/autodl-tmp/imagenet/val \
    --output_dir "/root/autodl-tmp/eval_dit_random_search_fid/num_candidates_${nc}_cfg1.0_steps10" \
    --num_samples 50000 \
    --batch_size 512 \
    --num_candidates "${nc}" \
    --deno_steps 10 \
    --guidance_scale 1.0
done

echo "Sweep finished."
