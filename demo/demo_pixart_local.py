#!/usr/bin/env python3
"""
Minimal local PixArt demo.

Example:
  python /root/diffusion_ranker/demo/demo_pixart_local.py \
    --model_path /root/autodl-tmp/PixArt-XL-2-1024-MS \
    --prompt "a cinematic photo of a panda astronaut, ultra detailed"
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local PixArt model for quick generation.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/PixArt-XL-2-1024-MS",
        help="Local diffusers model directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A fantasy city floating above clouds, concept art, highly detailed, 8k",
        help="Text prompt.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, watermark, text, deformed",
        help="Negative prompt.",
    )
    parser.add_argument("--num_images", type=int, default=2, help="Number of images to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="CFG scale.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Computation dtype on GPU.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/diffusion_ranker/demo/outputs_pixart",
        help="Directory to save generated images.",
    )
    return parser.parse_args()


def _resolve_dtype(dtype_name: str, use_cuda: bool) -> torch.dtype:
    if not use_cuda:
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    torch_dtype = _resolve_dtype(args.dtype, use_cuda)

    print(f"[load] model={model_path}")
    print(f"[load] device={device}, dtype={torch_dtype}")

    pipe = PixArtAlphaPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    pipe = pipe.to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("[run] generating images...")
    for idx in range(args.num_images):
        cur_seed = args.seed + idx
        generator = torch.Generator(device=device).manual_seed(cur_seed)
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]

        file_name = f"pixart_{ts}_seed{cur_seed}_{idx:02d}.png"
        save_path = out_dir / file_name
        image.save(save_path)
        print(f"[save] {save_path}")

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
