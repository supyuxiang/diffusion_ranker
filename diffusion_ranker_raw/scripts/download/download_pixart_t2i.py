#!/usr/bin/env python3
"""
下载 PixArt-Sigma（DiT 架构的 text-to-image 模型）到本地

PixArt-Sigma: DiT-based T2I, 支持 1024/2K/4K 分辨率，约 0.6B 参数

用法:
    python3 scripts/download_pixart_t2i.py --save_dir /data/home/yxfeng/data/models/pixart_sigma
    # 若 HF 镜像 404，可加 --no_mirror 使用官方源
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="/data/home/yxfeng/data/models/pixart_sigma",
        help="模型保存目录",
    )
    parser.add_argument(
        "--no_mirror",
        action="store_true",
        help="使用官方 huggingface.co（镜像 404 时用）",
    )
    args = parser.parse_args()

    if args.no_mirror:
        os.environ.pop("HF_ENDPOINT", None)  # 使用官方源
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 仅 1024-MS 在 HF 上可用，512-MS 可能不存在
    model_id = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    print(f"正在下载 {model_id} 到 {save_dir}")
    print(f"源: {'官方 huggingface.co' if args.no_mirror else '镜像 hf-mirror.com'}")
    print("(首次下载需从 Hugging Face 拉取，可能较慢)\n")

    from diffusers import PixArtSigmaPipeline
    import torch

    pipe = PixArtSigmaPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe.save_pretrained(str(save_dir))
    print(f"\n已保存到: {save_dir}")

    # 快速测试
    print("\n快速测试生成...")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipe("a cute cat", num_inference_steps=20).images[0]
    test_path = save_dir / "test_sample.png"
    image.save(test_path)
    print(f"测试图像已保存: {test_path}")


if __name__ == "__main__":
    main()
