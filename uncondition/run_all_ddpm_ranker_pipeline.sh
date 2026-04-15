#!/usr/bin/env bash
set -euo pipefail

# Optional defaults (can be overridden by caller)
OUT_IMG_ROOT="${OUT_IMG_ROOT:-/root/autodl-tmp/ddpm_candidates_50k_deno10}"
OUT_REPR_ROOT="${OUT_REPR_ROOT:-/root/autodl-tmp/ddpm_repr_candidates_deno10}"
REPR_KEY="${REPR_KEY:-s9_r0.6}"
SKIP_ROLLOUT="${SKIP_ROLLOUT:-0}"

# 0) rollout images + layer/step representations
if [[ "${SKIP_ROLLOUT}" != "1" ]]; then
  OUT_IMG_ROOT="${OUT_IMG_ROOT}" OUT_REPR_ROOT="${OUT_REPR_ROOT}" \
  bash /root/diffusion_ranker/uncondition/run_ddpm_rollout_repr.sh
fi

# 1) build train data (use rollout representations by default)
REPR_ROOT="${REPR_ROOT:-${OUT_REPR_ROOT}}" REPR_KEY="${REPR_KEY}" \
bash /root/diffusion_ranker/uncondition/run_ddpm_build_train_data.sh

# 2) train ranker
bash /root/diffusion_ranker/uncondition/run_ddpm_train_ranker.sh

# 3) random search with ranker
bash /root/diffusion_ranker/uncondition/run_ddpm_random_search_ranker.sh

# 4) FK with ranker
bash /root/diffusion_ranker/uncondition/run_ddpm_fk_ranker.sh
