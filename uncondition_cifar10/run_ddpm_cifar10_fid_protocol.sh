#!/usr/bin/env bash
# 对齐 HF google/ddpm-cifar10-32 讨论 #10：EMA + 1000 步 DDPM + 50k 生成 vs 50k train FID。
# 无外网时：请先把 EMA 的 diffusers 快照放到 MODEL_PATH（例如从已有机器拷贝
# VainF/Diff-Pruning 的 ddpm_ema_cifar10 转换结果）。
#
# 关于 diffusers#2585：该 issue 针对 DDIM 时间步离散方式；本脚本使用 DDPMPipeline
# 的 num_inference_steps=1000（DDPM），与 #2585 不是同一条代码路径。

set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/ddpm_ema_cifar10}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/outputs/ddpm_cifar10_fid_protocol}"

python /root/diffusion_ranker/uncondition_cifar10/ddpm_cifar10_fid_protocol.py \
  --model_path "${MODEL_PATH}" \
  --deno_steps "${DENO_STEPS:-1000}" \
  --num_gen "${NUM_GEN:-50000}" \
  --batch_size "${BATCH_SIZE:-2048}" \
  --real_split "${REAL_SPLIT:-train}" \
  --save_dir "${SAVE_DIR}"
