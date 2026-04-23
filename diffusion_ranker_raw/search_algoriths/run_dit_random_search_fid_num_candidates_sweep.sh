#!/usr/bin/env bash
# Sweep num_candidates for dit_random_search_fid_reward.py (same settings as docstring example).
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

for nc in 2 3 4 5 6 7 8; do
  echo "========== num_candidates=${nc} =========="
  python3 -u dit_random_search_fid_reward.py \
    --dit_path /data/yxfeng/models/dit_model \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --output_dir "/data/yxfeng/eval_dit_random_search_fid/num_candidates_${nc}_cfg1.5" \
    --num_samples 50000 \
    --batch_size 24 \
    --num_candidates "${nc}" \
    --init_random_samples 1024 \
    --deno_steps 10 \
    --guidance_scale 1.25
done

echo "Sweep finished (num_candidates 2..8)."
