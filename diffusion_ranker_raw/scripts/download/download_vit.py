#!/usr/bin/env python3
"""
下载 ViT 模型到本地（使用 hf-mirror，适合无法直连 huggingface.co 的环境）
下载后可用 local_files_only=True 离线加载
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 可选模型（选一个小的先玩）:
# - google/vit-base-patch16-224: 86M 参数，ImageNet 预训练
# - microsoft/swin-tiny-patch4-window7-224: 更小
MODEL_ID = "google/vit-base-patch16-224"
SAVE_DIR = Path("/data/home/yxfeng/data/models/vit_model")

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"正在从 hf-mirror 下载 {MODEL_ID} 到 {SAVE_DIR} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=str(SAVE_DIR))
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, cache_dir=str(SAVE_DIR))
    # 额外保存一份到明确路径，方便离线加载
    processor.save_pretrained(SAVE_DIR / "processor")
    model.save_pretrained(SAVE_DIR / "model")
    print(f"下载完成！模型已保存到 {SAVE_DIR}")
    print("离线加载示例:")
    print('  processor = AutoImageProcessor.from_pretrained(SAVE_DIR / "processor", local_files_only=True)')
    print('  model = AutoModelForImageClassification.from_pretrained(SAVE_DIR / "model", local_files_only=True)')

if __name__ == "__main__":
    main()
