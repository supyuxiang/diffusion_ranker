#!/usr/bin/env bash
# 依赖: hf（huggingface_hub CLI）。国内可配合 hf-mirror。
# 需要访问 gated 资源时: export HF_TOKEN=...  （勿把 token 写进仓库）

set -euo pipefail

# autodl 加速（非 autodl 环境若无此文件可注释掉下一行）
if [[ -f /etc/network_turbo ]]; then
  # shellcheck source=/dev/null
  source /etc/network_turbo
fi

ROOT="${DIFFUSION_RANKER_DATA:-/root/autodl-tmp/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---------- DiT ----------
mkdir -p "${ROOT}/DiT-XL-2-256"
hf download facebook/DiT-XL-2-256 --local-dir "${ROOT}/DiT-XL-2-256"

# ---------- ImageReward ----------
mkdir -p "${ROOT}/ImageReward"
hf download yuxng/ImageReward --local-dir "${ROOT}/ImageReward"

# ---------- DDPM CelebA-HQ 256（与 utils.model_name2path / eval 脚本本地路径一致可建 symlink）----------
mkdir -p "${ROOT}/google_ddpm-celebahq-256"
hf download google/ddpm-celebahq-256 --local-dir "${ROOT}/google_ddpm-celebahq-256"

# ---------- CLIP ViT-B/32（torchmetrics CLIPScore 默认路径常指向含 openai 的目录名）----------
mkdir -p "${ROOT}/openai_clip-vit-base-patch32"
hf download openai/clip-vit-base-patch32 --local-dir "${ROOT}/openai_clip-vit-base-patch32"

# ---------- CLIP ViT-L/14（dit_random_search_clipscore 等）----------
mkdir -p "${ROOT}/openai_clip-vit-large-patch14"
hf download openai/clip-vit-large-patch14 --local-dir "${ROOT}/openai_clip-vit-large-patch14"

# imagenet 和 celeba 在 autodl-pub 上有，可自行解压到 ${ROOT}
