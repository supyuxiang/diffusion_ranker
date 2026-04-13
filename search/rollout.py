#!/usr/bin/env python3
"""
多卡：每进程 ``--candidate_idx``，写满一个目录（默认 50k 张），供后续 search 只读盘。

输出目录（固定）::

    /root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}/candidate_{idx:04d}/

示例::

  CUDA_VISIBLE_DEVICES=0 python3 search/rollout.py --candidate_idx 0 --deno_steps 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import DiTPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import model_loader, rollout_dit_with_steps, set_device

BASE_SEED = 42


def _output_dir(guidance_scale: float, deno_steps: int, candidate_idx: int) -> Path:
    root = Path(
        f"/root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}"
    )
    return root / f"candidate_{candidate_idx:04d}"


def rollout_init_1024(
    pipe: DiTPipeline,
    guidance_scale: float,
    deno_steps: int,
) -> int:

    # save_dir = Path(
    #     f"/root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}/init1024"
    # )

    init1024_dir: Path=Path("/root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}/init1024")
    init1024_dir.mkdir(parents=True, exist_ok=True)

    seed = 521
    num_samples = 1024
    batch_size = 512
    device = set_device()

    
    return rollout_dit_with_steps(
        pipe,
        init1024_dir,
        num_steps,
        num_samples,
        batch_size,
        device,
        seed=seed,
        guidance_scale=guidance_scale,
    )


@torch.inference_mode()
def rollout_candidate(
    pipe: DiTPipeline,
    candidate_dir: Path,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    device: torch.device,
    candidate_idx: int,
    guidance_scale: float,
) -> int:
    seed = BASE_SEED + candidate_idx * 520
    return rollout_dit_with_steps(
        pipe,
        candidate_dir,
        num_steps,
        num_samples,
        batch_size,
        device,
        seed=seed,
        guidance_scale=guidance_scale,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="dit-xl-2-256")
    p.add_argument("--candidate_idx", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=("fp32", "fp16", "bf16"))
    args = p.parse_args()

    device = set_device()
    out_dir = _output_dir(args.guidance_scale, args.deno_steps, args.candidate_idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = model_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    if device.type == "cuda":
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")

    n = rollout_candidate(
        pipe,
        out_dir,
        args.num_samples,
        args.batch_size,
        args.deno_steps,
        device,
        args.candidate_idx,
        args.guidance_scale,
    )
    if n != args.num_samples:
        print(f"warning: wrote {n}, expected {args.num_samples}", file=sys.stderr)
    print(f"done -> {out_dir} ({n} images)")


if __name__ == "__main__":
    main()
