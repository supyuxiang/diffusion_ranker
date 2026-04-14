#!/usr/bin/env bash
# 依次运行 dit_random_search_imagereward.py，num_candidates = 2,4,6,…,16。
#
# 依赖（与脚本内 deno_steps / guidance_scale 一致）::
#   /root/autodl-tmp/dit_candidates_50k_cfg1.0_deno10/candidate_0000 …
#   /root/autodl-tmp/dit_candidates_50k_cfg1.0_deno10/init1024/  （固定 1024 张）
# 每个 candidate_* 目录须有不少于 --num_samples 张 rollout 图。
#
# 用法::
#   bash /root/diffusion_ranker/scripts/run_imagereward_search.sh
#
# 额外参数会传给每一次 python，例如::
#   bash .../run_imagereward_search.sh --metrics_batch_size 128 --imagereward_ckpt ImageReward-v1.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0

for k in 2 4 6 8 10 12 14 16; do
  python3 /root/diffusion_ranker/search/dit_random_search_imagereward.py \
    --batch_size 512 \
    --num_candidates "$k" \
    --deno_steps 10 \
    --guidance_scale 1.0 \
    --num_samples 50000 \
    --prompt_template "a photo of a {name}" \
    --seed 42 \
    --output_dir "/root/autodl-tmp/random_search_imagereward_50k_cfg1.0_deno10/num_candidates_$k" \
    "$@"
done
