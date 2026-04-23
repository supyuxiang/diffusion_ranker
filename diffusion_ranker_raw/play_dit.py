#!/usr/bin/env python3
"""
DiT (Diffusion Transformer) 演示：类别条件图像生成
支持离线加载（local_files_only=True）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

SAVE_DIR = Path("/data/home/yxfeng/data/models/dit_model")

# ImageNet 1000 类中的一些有趣类别（示例）
CLASS_NAMES = [
    "goldfish", "golden retriever", "tabby cat", "pizza", "pizza pie",
    "strawberry", "orange", "broccoli", "teddy bear", "laptop",
]
CLASS_IDS = [1, 207, 281, 963, 963, 949, 950, 937, 850, 620]  # 对应 id

def load_dit(offline: bool = True):
    pipe = DiTPipeline.from_pretrained(
        str(SAVE_DIR), local_files_only=offline, torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_id", type=int, default=281, help="ImageNet 类别 ID (0-999)，如 207=金毛")
    parser.add_argument("--num_images", type=int, default=4, help="生成图像数量")
    parser.add_argument("--num_inference_steps", type=int, default=999, help="推理步数（DPMSolver 推荐 20-50，1000 会报错）")
    parser.add_argument("--output_dir", default="/data/home/yxfeng/project1/idm/dit_outputs")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("加载 DiT 模型...")
    pipe = load_dit(offline=args.offline)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # 类别标签
    labels = [args.class_id] * args.num_images
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"生成 {args.num_images} 张图像，类别 ID={args.class_id}，步数={args.num_inference_steps} ...")
    images = pipe(
        labels,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).images
    print('-'*100)
    print(images)
    print(')'*100)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = out_dir / f"dit_class{args.class_id}_{i:02d}.png"
        img.save(path)
        print(f"  保存: {path}")
    print(f"\n完成！输出目录: {out_dir}")

if __name__ == "__main__":
    main()
