#!/usr/bin/env python3
"""
无条件扩散模型图像生成 Demo
使用 HF-Mirror 下载 google/ddpm-celebahq-256 模型，生成 256x256 人脸图像
"""

import os

# 使用 HF-Mirror 加速国内下载（必须在 import diffusers 之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import DDPMPipeline
from pathlib import Path


def main():
    # 配置
    model_id = "google/ddpm-celebahq-256"
    output_dir = Path(__file__).parent / "outputs_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_images = 4  # 生成图像数量
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型（首次运行会自动从 hf-mirror 下载）
    pipe = DDPMPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    
    # generate images
    gen = torch.Generator(device=device).manual_seed(42)
    # DDPMPipeline 为无条件模型，无 CFG，不能传 guidance_scale
    result = pipe(
        batch_size=num_images,
        generator=gen,
        num_inference_steps=100,
        output_type="pil",
    )
    
    images = result.images
    
    # save images
    for i, image in enumerate(images):
        save_path = output_dir / f"generated_{i+1}.png"
        image.save(save_path)
        print(f"saved: {save_path}")
    


if __name__ == "__main__":
    main()
