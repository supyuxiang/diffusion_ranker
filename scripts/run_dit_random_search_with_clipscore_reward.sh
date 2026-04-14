#!/usr/bin/env bash
# 依次运行 dit_random_search_clipscore.py，num_candidates = 2..8。
#
# 用法:
#   cd /data/home/yxfeng/diffusion_ranker/search_algoriths
#   CUDA_VISIBLE_DEVICES=0 ./run_clipscore_num_candidates_sweep.sh
#
# 额外参数会传给每一次 python 调用，例如:
#   ./run_clipscore_num_candidates_sweep.sh --num_samples 1000 --deno_steps 10 --batch_size 16
#
# 环境变量（可选）:
#   DIT_PATH, IMAGENET_VAL, CLIP_MODEL_PATH, BASE_OUT, SEED

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0

for k in 2 4 8 16; do
  python /root/diffusion_ranker/search/dit_random_search_clipscore.py \
    --batch_size 512 \
    --num_candidates $k \
    --deno_steps 10 \
    --guidance_scale 1.0 \
    --num_samples 50000 \
    --prompt_template "a photo of a {name}" \
    --seed 42 \
    --output_dir /root/autodl-tmp/random_search_clipscore_50k_cfg1.0_deno10_reward_nc_$k
done