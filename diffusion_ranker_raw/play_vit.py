#!/usr/bin/env python3
"""
ViT 模型简单演示：加载模型并对图像做分类
支持离线加载（local_files_only=True）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

SAVE_DIR = Path("/data/home/yxfeng/data/models/vit_model")
OFFLINE = True  # 设为 True 则只用本地缓存，不联网

def load_vit(offline: bool = True):
    processor_path = SAVE_DIR / "processor"
    model_path = SAVE_DIR / "model"
    if not processor_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"模型未找到。请先运行: python download_vit.py\n"
            f"或从有网络的机器下载后，将 {SAVE_DIR} 目录拷贝到本机"
        )
    processor = AutoImageProcessor.from_pretrained(
        str(processor_path), local_files_only=offline
    )
    model = AutoModelForImageClassification.from_pretrained(
        str(model_path), local_files_only=offline
    )
    return processor, model

def predict(image_path: str, processor, model, device, top_k: int = 5):
    """对单张图像做分类预测"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    print('-'*100)
    print(outputs)
    print(outputs.keys())

    print('-'*100)
    hidden = outputs.hidden_states
    print(hidden[0].shape)
    print('-'*100)
    probs = torch.softmax(outputs.logits, dim=-1)
    top_probs, top_indices = torch.topk(probs[0], top_k)
    return [
        (model.config.id2label[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default='/data/home/yxfeng/image.jpg', help="图像路径，不传则用随机噪声测试")
    parser.add_argument("--offline", action="store_true", default=True, help="离线加载")
    args = parser.parse_args()

    print("加载 ViT 模型...")
    processor, model = load_vit(offline=args.offline)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    if args.image and Path(args.image).exists():
        print(f"\n对图像 {args.image} 做分类:")
        results = predict(args.image, processor, model, device)
        for label, prob in results:
            print(f"  {prob:.2%}  {label}")
    else:
        # 无图像时，用随机噪声测试模型能跑通
        print("\n无图像输入，用随机 224x224 图像测试前向...")
        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(dummy)
        print(f"输出 logits shape: {out.logits.shape}, 类别数: {out.logits.shape[-1]}")

if __name__ == "__main__":
    main()
