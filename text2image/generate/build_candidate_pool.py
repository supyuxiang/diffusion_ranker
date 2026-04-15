#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from text2image.common import (
    dump_json,
    ensure_dir,
    load_prompts_jsonl,
    safe_image_name,
    set_seed,
    torch_dtype_from_str,
)


def _load_t2i_pipeline(model_name_or_path: str, dtype: torch.dtype, device: torch.device):
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    return pipe


def _save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", compress_level=1)


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser("Build offline candidate pool for text-to-image")
    p.add_argument("--prompts_jsonl", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, default="/root/autodl-tmp/stable-diffusion-v1-5")
    p.add_argument("--num_candidates", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", type=str, default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    set_seed(args.seed)
    prompts = load_prompts_jsonl(args.prompts_jsonl)
    out_root = ensure_dir(args.output_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype_from_str(args.dtype)
    pipe = _load_t2i_pipeline(args.model_name_or_path, dtype=dtype, device=device)

    run_meta = {
        "model_name_or_path": args.model_name_or_path,
        "num_candidates": int(args.num_candidates),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "height": int(args.height),
        "width": int(args.width),
        "batch_size": int(args.batch_size),
        "dtype": str(args.dtype),
        "seed": int(args.seed),
        "device": str(device),
    }
    dump_json(run_meta, out_root / "run_config.json")

    for pi, item in enumerate(tqdm(prompts, desc="prompts")):
        pdir = out_root / item.benchmark / item.prompt_id
        img_dir = pdir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        all_seeds: list[int] = []
        remaining = int(args.num_candidates)
        offset = 0
        while remaining > 0:
            n_this = min(remaining, int(args.batch_size))
            seeds = [
                int(args.seed) + pi * int(args.num_candidates) + offset + k
                for k in range(n_this)
            ]
            gens = [torch.Generator(device=device).manual_seed(s) for s in seeds]
            out = pipe(
                prompt=[item.prompt_text] * n_this,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.guidance_scale),
                height=int(args.height),
                width=int(args.width),
                generator=gens,
            )
            for j, img in enumerate(out.images):
                global_idx = offset + j
                _save_png(img, img_dir / safe_image_name(global_idx))
            all_seeds.extend(seeds)
            remaining -= n_this
            offset += n_this

        dump_json(
            {
                "prompt_id": item.prompt_id,
                "prompt_text": item.prompt_text,
                "benchmark": item.benchmark,
                "tags": item.tags or [],
                "image_count": int(args.num_candidates),
                "seeds": all_seeds,
            },
            pdir / "meta.json",
        )


if __name__ == "__main__":
    main()
