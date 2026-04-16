#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline


def _dtype_from_str(dtype: str) -> torch.dtype:
    d = dtype.strip().lower()
    if d == "fp16":
        return torch.float16
    if d == "bf16":
        return torch.bfloat16
    if d == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype: {dtype}")


def _build_pipe(model_path: str, dtype: torch.dtype):
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser("Minimal SDXL local inference demo")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/sdxl-base-1.0")
    p.add_argument("--prompt", type=str, default="a cinematic portrait of a young woman, 85mm lens, soft light")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/sdxl_demo_outputs")
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--enable_attention_slicing", action="store_true")
    args = p.parse_args()

    dtype = _dtype_from_str(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pipe = _build_pipe(args.model_path, dtype=dtype)
    pipe = pipe.to(device)
    if args.enable_attention_slicing:
        pipe.enable_attention_slicing()
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    gens = [torch.Generator(device=device).manual_seed(int(args.seed) + i) for i in range(int(args.num_images))]
    prompts = [str(args.prompt)] * int(args.num_images)
    negatives = [str(args.negative_prompt)] * int(args.num_images) if str(args.negative_prompt) else None

    out = pipe(
        prompt=prompts,
        negative_prompt=negatives,
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        height=int(args.height),
        width=int(args.width),
        generator=gens,
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths: list[str] = []
    for i, img in enumerate(out.images):
        save_p = out_dir / f"sdxl_{ts}_{i:02d}_seed{int(args.seed)+i}.png"
        img.save(save_p)
        saved_paths.append(str(save_p))

    meta = {
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "prompt": str(args.prompt),
        "negative_prompt": str(args.negative_prompt),
        "num_images": int(args.num_images),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "height": int(args.height),
        "width": int(args.width),
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "device": str(device),
        "saved_paths": saved_paths,
    }
    (out_dir / f"run_{ts}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
