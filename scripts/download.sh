#!/usr/bin/env bash
# 依赖: huggingface-cli（huggingface_hub）。国内可配合 hf-mirror。
# 需要访问 gated 资源时: export HF_TOKEN=...  （勿把 token 写进仓库）

set -euo pipefail

# autodl 加速
source /etc/network_turbo


ROOT="${DIFFUSION_RANKER_DATA:-/autodl-tmp/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---------- DiT ----------
mkdir -p "${ROOT}/DiT-XL-2-256"
huggingface-cli download facebook/DiT-XL-2-256 \
  --local-dir "${ROOT}/DiT-XL-2-256" \
  --local-dir-use-symlinks False

# ---------- ImageReward ----------
mkdir -p "${ROOT}/ImageReward"
huggingface-cli download yuxng/ImageReward \
  --local-dir "${ROOT}/ImageReward" \
  --local-dir-use-symlinks False

# ---------- DDPM CelebA-HQ 256（与 utils.model_name2path / eval 脚本本地路径一致可建 symlink）----------
mkdir -p "${ROOT}/google_ddpm-celebahq-256"
huggingface-cli download google/ddpm-celebahq-256 \
  --local-dir "${ROOT}/google_ddpm-celebahq-256" \
  --local-dir-use-symlinks False

# ---------- CLIP ViT-B/32（CLIPScore / 多模态打分）----------
mkdir -p "${ROOT}/openai_clip-vit-base-patch32"
huggingface-cli download openai/clip-vit-base-patch32 \
  --local-dir "${ROOT}/openai_clip-vit-base-patch32" \
  --local-dir-use-symlinks False



# imagenet和celeba 在autodl-pub上有，可以自行解压到/root/autodl-tmp/中

