#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/1aurent_ddpm-mnist}"
BASE_SAVE_ROOT="${BASE_SAVE_ROOT:-/root/autodl-tmp}"
NUM_SAMPLES_ROLLOUT="${NUM_SAMPLES_ROLLOUT:-2000}"
NUM_SAMPLES_SEARCH="${NUM_SAMPLES_SEARCH:-10000}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"

REWARDS=("brightness" "sharpness" "image_entropy")
SUMMARY_JSON="${BASE_SAVE_ROOT}/mnist_three_rewards_trans_ps8_summary.json"

python - <<'PY' "${SUMMARY_JSON}"
import json,sys
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps({"runs": []}, indent=2))
PY

for REWARD in "${REWARDS[@]}"; do
  echo "=============================="
  echo "[run] reward=${REWARD}"
  echo "=============================="

  SAVE_PATH="${BASE_SAVE_ROOT}/mnist_ddpm_hiddens_and_rewards_${REWARD}.pth"
  SHARDS_PATH="${BASE_SAVE_ROOT}/mnist_ddpm_hiddens_and_rewards_${REWARD}_shards"
  RANKER_DIR="${BASE_SAVE_ROOT}/mnist_ranker_trans_${REWARD}/s0_l9/hd8_h8_lr1e-4_n8"
  SEARCH_DIR="${BASE_SAVE_ROOT}/mnist_random_search_ranker_trans_${REWARD}/s0_l9/hd8_h8_lr1e-4_n8_ps8"
  EVAL_JSON="${SEARCH_DIR}/three_rewards_metrics.json"

  # 1) rollout data with current reward
  REWARD_TYPE="${REWARD}" \
  MODEL_PATH="${MODEL_PATH}" \
  SAVE_PATH="${SAVE_PATH}" \
  NUM_SAMPLES="${NUM_SAMPLES_ROLLOUT}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  bash /root/diffusion_ranker/uncondition_mnist/run_mnist_ddpm_rollout.sh

  # 2) train trans ranker (n8 setting)
  python /root/diffusion_ranker/uncondition_mnist/train_ranker_trans.py \
    --data_path "${SHARDS_PATH}" \
    --step_idx 0 \
    --layer_idx 9 \
    --num_candidates 8 \
    --group_seed 0 \
    --num_layers 2 \
    --hidden_dim 8 \
    --num_heads 8 \
    --input_channels 64 \
    --dropout 0.1 \
    --alpha_kl 0.3 \
    --alpha_mse 0.7 \
    --tau 1.0 \
    --num_epochs 30 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --num_workers 4 \
    --train_ratio 0.9 \
    --split_seed 42 \
    --save_interval 1 \
    --save_dir "${RANKER_DIR}"

  # 3) random search with ps=8
  python /root/diffusion_ranker/uncondition_mnist/random_search_ranker_trans_reward.py \
    --ranker_ckpt "${RANKER_DIR}/ranker_best.pt" \
    --output_dir "${SEARCH_DIR}" \
    --model_path "${MODEL_PATH}" \
    --num_samples "${NUM_SAMPLES_SEARCH}" \
    --parallel_size 8 \
    --num_layers 2 \
    --hidden_dim 8 \
    --num_heads 8 \
    --input_channels 64 \
    --layer_idx 9 \
    --step_idx 0

  # 4) evaluate three image rewards on generated images
  python /root/diffusion_ranker/uncondition_mnist/eval_three_image_rewards.py \
    --gen_dir "${SEARCH_DIR}" \
    --output_json "${EVAL_JSON}"

  # 5) append summary
  python - <<'PY' "${SUMMARY_JSON}" "${REWARD}" "${RANKER_DIR}" "${SEARCH_DIR}" "${EVAL_JSON}"
import json,sys
summary_path,reward,ranker_dir,search_dir,eval_json = sys.argv[1:]
summary = json.load(open(summary_path, "r", encoding="utf-8"))
eval_data = json.load(open(eval_json, "r", encoding="utf-8"))
summary["runs"].append(
    {
        "reward_type": reward,
        "ranker_dir": ranker_dir,
        "search_dir": search_dir,
        "metrics": eval_data,
    }
)
open(summary_path, "w", encoding="utf-8").write(json.dumps(summary, indent=2))
PY
done

echo "[done] summary saved to ${SUMMARY_JSON}"
