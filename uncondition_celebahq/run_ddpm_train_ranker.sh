#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python /root/diffusion_ranker/uncondition_celebahq/ddpm_train_ranker.py \
  --train_data_pt "${TRAIN_PT:-/root/autodl-tmp/ddpm_ranker_train_data_deno10.pt}" \
  --output_dir "${CKPT_DIR:-/root/autodl-tmp/ddpm_ranker_ckpt_deno10}" \
  --batch_size "${BATCH_SIZE:-512}" \
  --num_epochs "${NUM_EPOCHS:-12}" \
  --lr "${LR:-1e-4}" \
  --weight_decay "${WD:-1e-4}" \
  --hidden_dim "${HIDDEN_DIM:-256}" \
  --num_layers "${NUM_LAYERS:-4}" \
  --dropout "${DROPOUT:-0.1}"
