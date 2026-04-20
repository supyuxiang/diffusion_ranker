#!/usr/bin/env bash
set -euo pipefail

# example:
# bash /root/diffusion_ranker/class_condition/scripts/rs_ranker.sh

ps_ls=(2 4 8)
deno_steps_ls=(10 25)

for deno_steps in "${deno_steps_ls[@]}"; do
  rollout_root="/root/autodl-tmp/rollout_candidate_step${deno_steps}_cfg1"
  if [[ ! -d "${rollout_root}" ]]; then
    mkdir -p "${rollout_root}"
    echo "created missing rollout_root: ${rollout_root}"
  fi
  for ps in "${ps_ls[@]}"; do
    python /root/diffusion_ranker/class_condition/dit_random_search_ranker.py \
      --output_dir "/root/autodl-tmp/class_condition/dit_random_search_ranker_steps${deno_steps}_ps${ps}" \
      --num_samples 50000 \
      --deno_steps "${deno_steps}" \
      --guidance_scale 1 \
      --num_candidates "${ps}" \
      --seed 42 \
      --search_batch_size 2
  done
done