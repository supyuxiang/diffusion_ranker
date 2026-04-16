"""
example:

    CUDA_VISIBLE_DEVICES=0 python /root/diffusion_ranker/uncondition/unet_scaling_with_different_steps.py \
        --model_name google_ddpm-celebahq-256 \
        --batch_size 128 \
        --num_gen 50000 \
        --save_dir /root/autodl-tmp/outputs/unet_scaling_with_different_steps \
        --real_dir /root/autodl-tmp/img_align_celeba 
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT_ = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(_ROOT_))

import torch

from utils import (
    model_loader,
    rollout_ddpm_with_steps,
    set_device,
    set_seed,
    compute_fid_is_clipscore_for_uncondition,
)

_DENO_STEP_LS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name",
        type=str,
        default="google_ddpm-celebahq-256",
        help='utils里面有写好的model_name2path'
    )
    p.add_argument("--batch_size", type=int, default=128) # 128需要大概40G显存
    p.add_argument("--num_gen", type=int, default=50000)
    p.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/outputs/unet_scaling_with_different_steps",
    )
    p.add_argument(
        "--real_dir",
        type=str,
        default="/root/autodl-tmp/img_align_celeba",
        help="真实图像根目录",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device()

    real_dir = Path(args.real_dir).expanduser().resolve()
    if not real_dir.is_dir():
        raise FileNotFoundError(f"--real_dir is not a directory: {real_dir}")

    dtype = args.dtype
    if dtype == "bf16":
        # diffusers DDPMPipeline converts tensors to numpy internally; numpy does not support bf16.
        print("[warn] bf16 is not supported in DDPMPipeline numpy conversion, fallback to fp16.")
        dtype = "fp16"
    pipe = model_loader(args.model_name, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    base_save = Path(args.save_dir).expanduser().resolve()

    for deno_step in _DENO_STEP_LS:
        out_dir = base_save / f"deno_step_{deno_step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        rollout_ddpm_with_steps(
            pipe,
            out_dir,
            deno_step,
            args.num_gen,
            args.batch_size,
            device,
            seed=args.seed,
        )

        clip_model_dir = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")

        metrics = compute_fid_is_clipscore_for_uncondition(
            str(out_dir),
            str(real_dir),
            clip_model_dir=str(clip_model_dir),
            clip_batch_size=64,
        )

        metrics_path = out_dir / f"metrics_deno_step_{deno_step}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Deno step {deno_step} metrics: {metrics}")
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
