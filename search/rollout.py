#!/usr/bin/env python3
"""
多卡：每进程 ``--candidate_idx``，写满一个目录（默认 50k 张），供后续 search 只读盘。

输出目录（固定）::

    /root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}/candidate_{idx:04d}/

示例::

  CUDA_VISIBLE_DEVICES=0 python3 /root/diffusion_ranker/search/rollout.py \
  --candidate_idx 0 \
  --deno_steps 10 \
  --guidance_scale 1.0 \
  --num_samples 50000 \
  --batch_size 512 \
  --dtype bf16 \
  --repr_pt "/root/autodl-tmp/repr_candidate_0_deno10.pt"
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
from diffusers import DiTPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import DitRolloutReprCapture, model_loader, rollout_dit_with_steps, set_device,refresh_memory

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

    init1024_dir = Path(
        f"/root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}/init1024"
    )
    init1024_dir.mkdir(parents=True, exist_ok=True)

    seed = 521
    num_samples = 1024
    batch_size = 512
    device = set_device()

    return rollout_dit_with_steps(
        pipe,
        init1024_dir,
        deno_steps,
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
    *,
    repr_capture: DitRolloutReprCapture | None = None,
    repr_max_batches: int | None = None,
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
        repr_capture=repr_capture,
        repr_max_batches=repr_max_batches,
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
    p.add_argument(
        "--repr_pt",
        default="",
        help="若非空：抓取去噪步 {0,T/4,T/2,3T/4,T} 上若干层的 token mean/last 并存此 .pt（与 compile 不兼容，将自动关闭 compile）",
    )
    p.add_argument(
        "--repr_max_batches",
        type=int,
        default=-1,
        help="仅记录前 K 个 batch 的表征；-1 表示每个 batch 都记（占用大）",
    )
    p.add_argument(
        "--repr_store_dtype",
        default="fp16",
        choices=("fp16", "bf16"),
        help="表征落盘前的精度（fp16 更省内存；与 sample_hidden 思路一致：尽快缩小 CPU 侧占用）",
    )
    args = p.parse_args()

    device = set_device()
    out_dir = _output_dir(args.guidance_scale, args.deno_steps, args.candidate_idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = model_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    repr_capture: DitRolloutReprCapture | None = None
    repr_max_batches: int | None = None
    if args.repr_pt: # 带表征提取的rollout
        store_dtype = (
            torch.bfloat16 if args.repr_store_dtype == "bf16" else torch.float16
        )
        repr_capture = DitRolloutReprCapture(
            pipe,
            num_inference_steps=args.deno_steps,
            guidance_scale=args.guidance_scale,
            store_dtype=store_dtype,
            rollout_base_seed=BASE_SEED + args.candidate_idx * 520,
        )
        repr_capture.install()
        repr_max_batches = None if args.repr_max_batches < 0 else args.repr_max_batches
    elif device.type == "cuda":
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
        repr_capture=repr_capture,
        repr_max_batches=repr_max_batches,
    )

    if repr_capture is not None:
        repr_capture.remove()
        repr_path = Path(args.repr_pt).expanduser().resolve()
        refresh_memory()
        repr_capture.save_pt(repr_path)
        refresh_memory()
        print(f"representations saved -> {repr_path}")
    if n != args.num_samples:
        print(f"warning: wrote {n}, expected {args.num_samples}", file=sys.stderr)
    print(f"done -> {out_dir} ({n} images)")


if __name__ == "__main__":
    # device = set_device()
    # pipe = model_loader("dit-xl-2-256", device=device, dtype="bf16")
    # pipe.set_progress_bar_config(disable=True)
    # rollout_init_1024(pipe, guidance_scale=1.0, deno_steps=10)
    main()
