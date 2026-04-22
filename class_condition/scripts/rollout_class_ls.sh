#!/usr/bin/env bash
set -euo pipefail

class_ls=(153 207 281 404 429 530 817 844 948)

MODEL_NAME="dit-xl-2-256"
NUM_SAMPLES=5000
BATCH_SIZE=32
NUM_INFERENCE_STEPS=10
GUIDANCE_SCALE=1.0
SEED=42
DTYPE="bf16"
TARGET_STEP_INDICES="1,6,9"
TARGET_LAYER_IDX=16
MC=8
LOG_DIR="/root/autodl-tmp/class_condition_rollout_logs"

mkdir -p "${LOG_DIR}"

for class_idx in "${class_ls[@]}"; do
    echo "Rolling out class ${class_idx}"
    # dit_rollout_mc_is_single.py writes one file per target step, e.g. *_s1_l16.pt
    save_path="/root/autodl-tmp/dit_imagenet_hiddens_and_is_rewards_s1_6_9_l16_class${class_idx}.pt"
    log_path="${LOG_DIR}/class_${class_idx}.log"

    python /root/diffusion_ranker/class_condition/dit_rollout_mc_is_single.py \
        --model_name "${MODEL_NAME}" \
        --class_idx "${class_idx}" \
        --num_samples "${NUM_SAMPLES}" \
        --batch_size "${BATCH_SIZE}" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" \
        --guidance_scale "${GUIDANCE_SCALE}" \
        --seed "${SEED}" \
        --dtype "${DTYPE}" \
        --target_step_indices "${TARGET_STEP_INDICES}" \
        --target_layer_idx "${TARGET_LAYER_IDX}" \
        --mc "${MC}" \
        --save_path "${save_path}" \
        2>&1 | tee "${log_path}"
done