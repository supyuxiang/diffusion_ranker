#!/usr/bin/env python3
"""
下载 ImageNet-1k 验证集的部分图像，用于 DiT 的 FID 评估
使用 hf-mirror，需先在 https://huggingface.co/datasets/ILSVRC/imagenet-1k 同意 ImageNet 使用条款

用法:
  python download_imagenet_subset.py --num_samples 10000 --output_dir /data/home/yxfeng/data/imagenet_val_subset
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="下载的验证集图像数量（最多 50000）")
    parser.add_argument("--output_dir", default="/data/home/yxfeng/data/imagenet_val_subset",
                        help="保存目录，将导出为 PNG 供 FID 使用")
    parser.add_argument("--split", default="validation", choices=["validation", "train"],
                        help="validation=5万张, train=约128万张")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载 ImageNet-1k 数据集（需已同意 HF 上的使用条款）...")
    from datasets import load_dataset

    # 只加载需要的数量，避免全量下载（注意：可能仍会下载完整 split 的元数据）
    split_spec = f"{args.split}[:{args.num_samples}]"
    ds = load_dataset("ILSVRC/imagenet-1k", split=split_spec)

    print(f"导出 {len(ds)} 张图像到 {output_dir} ...")
    for i in tqdm(range(len(ds)), desc="export"):
        item = ds[i]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB").resize((256, 256), Image.LANCZOS)
        img.save(output_dir / f"{i:06d}.png")

    print(f"完成！共 {len(ds)} 张，保存到 {output_dir}")
    print("DiT 评估时使用:")
    print(f"  --real_dataset {output_dir}")


if __name__ == "__main__":
    main()
