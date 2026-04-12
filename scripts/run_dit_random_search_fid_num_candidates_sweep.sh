#!/usr/bin/env bash
# Sweep num_candidates for dit_random_search_fid_reward.py (same settings as docstring example).
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for nc in 2 4 8 16; do
  echo "========== num_candidates=${nc} =========="
  python3 -u /root/diffusion_ranker_main/search_algorithms/dit_random_search_fid_reward.py \
    --model_name dit-xl-2-256 \
    --imagenet_val /root/autodl-tmp/imagenet/val \
    --output_dir "/root/autodl-tmp/eval_dit_random_search_fid/num_candidates_${nc}_cfg1.0_steps25" \
    --num_samples 50000 \
    --batch_size 512 \
    --num_candidates ${nc} \
    --init_random_samples 1024 \
    --deno_steps 25 \
    --guidance_scale 1.0 \
    --dtype bf16 \
    --compile
done

echo "Sweep finished (num_candidates 2..8)."
