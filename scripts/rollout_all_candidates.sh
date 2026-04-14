#!/usr/bin/env bash
# Rollout all candidates for deno_steps=10 only (see rollout.py output dirs).
#
# 若更新了 search/rollout.py 的表征抓取（与 ranker 训练 repr 键对齐），须重新跑本脚本生成 idx*.pt。
#
# 输出布局（与 search/dit_random_search_ranker.py 的 suite 预计算模式一致）::
#   /root/autodl-tmp/dit_candidates_50k_cfg{cfg}_deno{steps}/candidate_{idx:04d}/*.png
#   /root/autodl-tmp/repr_candidate_deno{steps}_cfg{cfg}/idx{idx}.pt
#
# Random search（search / init 均不复用 DiT：init 从 candidate_0000 复制前 1024 张）示例::
#
#   python3 /root/diffusion_ranker/search/dit_random_search_ranker.py \
#     --verifier ranker \
#     --ranker_ckpt /path/to/ranker.pt \
#     --precomputed_suite_image_root "/root/autodl-tmp/dit_candidates_50k_cfg1.0_deno10" \
#     --precomputed_suite_repr_dir "/root/autodl-tmp/repr_candidate_deno10_cfg1.0" \
#     --precomputed_suite_num_candidates 16 \
#     --batch_size 512 \
#     --num_candidates 16 \
#     --num_samples 2048 \
#     --init_random_samples 1024 \
#     --init_suite_candidate_idx 0 \
#     --output_dir /path/to/rs_out

set -euo pipefail

DENO_STEPS=10
cfg=1.0
idx_ls=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

for idx in "${idx_ls[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python3 /root/diffusion_ranker/search/rollout.py \
    --candidate_idx "${idx}" \
    --deno_steps "${DENO_STEPS}" \
    --guidance_scale "${cfg}" \
    --num_samples 50000 \
    --batch_size 512 \
    --dtype bf16 \
    --repr_pt "/root/autodl-tmp/repr_candidate_deno${DENO_STEPS}_cfg${cfg}/idx${idx}.pt"
done

# compute metrics: fid, is, clipscore
# TODO
