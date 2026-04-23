#!/usr/bin/env python3
"""
下载 DiT (Diffusion Transformer) 模型到本地
使用 hf-mirror，适合无法直连 huggingface.co 的环境
下载后可用离线加载
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from diffusers import DiTPipeline

# facebook/DiT-XL-2-256: 256x256 类别条件图像生成，ImageNet 1000 类
# facebook/DiT-XL-2-512: 512x512，更大
MODEL_ID = "facebook/DiT-XL-2-256"
SAVE_DIR = Path("/data/home/yxfeng/data/models/dit_model")

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"正在从 hf-mirror 下载 {MODEL_ID} 到 {SAVE_DIR} ...")
    pipe = DiTPipeline.from_pretrained(MODEL_ID)
    pipe.save_pretrained(SAVE_DIR)
    print(f"下载完成！模型已保存到 {SAVE_DIR}")
    print("离线加载示例:")
    print('  pipe = DiTPipeline.from_pretrained(SAVE_DIR, local_files_only=True)')

if __name__ == "__main__":
    main()
