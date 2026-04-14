#!/usr/bin/env bash
# 依次运行 dit_random_search_imagereward.py，num_candidates ∈ {2,4,6,…,16}
#
# 依赖: /root/autodl-tmp/dit_candidates_50k_cfg{gs}_deno{steps}/candidate_0000 …
#
# 用法:
#   bash /root/diffusion_ranker/scripts/run_imagereward_search.sh
#
# 可选: CUDA_VISIBLE_DEVICES=1 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for k in 2 4 6 8 10 12 14 16; do
  echo "========== num_candidates=${k} =========="
  python3 -u /root/diffusion_ranker/search/dit_random_search_imagereward.py \
    --search_batch_size 32 \
    --num_candidates "$k" \
    --deno_steps 10 \
    --guidance_scale 1.0 \
    --num_samples 50000 \
    --seed 42 \
    --output_dir "/root/autodl-tmp/random_search_imagereward_50k_cfg1.0_deno10_search32/num_candidates_$k"
done

echo "Sweep finished."
