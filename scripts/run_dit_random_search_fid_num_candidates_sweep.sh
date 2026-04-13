#!/usr/bin/env bash
# 对 dit_random_search_fid_reward.py 扫 num_candidates：nc ∈ {2, 4, 8, 16}
# （init 固定 N_INIT=1024，见 Python 脚本内常量）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${ROOT}/search/dit_random_search_fid_reward.py"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# NC_VALUES=(2 4 8 16)

NC_VALUES=(2 4 6 8 10 12 14 16)

for nc in "${NC_VALUES[@]}"; do
  echo "========== num_candidates=${nc} =========="
  python3 -u "${SCRIPT}" \
    --output_dir "/root/autodl-tmp/eval_dit_random_search_fid/num_candidates_${nc}_cfg1.0_steps10" \
    --num_samples 50000 \
    --batch_size 512 \
    --num_candidates "${nc}" \
    --deno_steps 10 \
    --guidance_scale 1.0
done

echo "Sweep finished (nc=${NC_VALUES[*]})."
