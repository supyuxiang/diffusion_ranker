#!/usr/bin/env python3
"""
CIFAR-10 DDPM scaling with different denoising steps (FID/IS only).

Example:
  CUDA_VISIBLE_DEVICES=0 python /root/diffusion_ranker/uncondition_cifar10/cifar10_scaling_with_different_steps.py \
      --model_name google_ddpm-cifar10-32 \
      --batch_size 512 \
      --num_gen 50000 \
      --save_dir /root/autodl-tmp/outputs/cifar10_scaling_with_different_steps

CUDA_VISIBLE_DEVICES=0 python /root/diffusion_ranker/uncondition_cifar10/cifar10_scaling_with_different_steps.py \
  --model_name google_ddpm-cifar10-32 \
  --batch_size 512 \
  --num_gen 50000 \
  --save_dir /root/autodl-tmp/outputs/cifar10_scaling_with_different_steps

  
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10

_ROOT_ = Path(__file__).resolve().parent.parent
if str(_ROOT_) not in sys.path:
    sys.path.insert(0, str(_ROOT_))

from utils import (
    compute_fid,
    compute_is,
    model_loader,
    rollout_ddpm_with_steps,
    set_device,
    set_seed,
)

_DENO_STEP_LS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _load_cifar10_test_images(data_root: str | Path) -> torch.Tensor:
    ds = CIFAR10(root=str(Path(data_root).expanduser().resolve()), train=False, download=False)
    # ds.data: (N, H, W, C), uint8
    arr = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    return arr.to(torch.uint8)


def _load_images_from_dir_uint8(img_dir: str | Path) -> torch.Tensor:
    d = Path(img_dir).expanduser().resolve()
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)
    imgs: list[torch.Tensor] = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)  # HWC
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        imgs.append(t)
    if not imgs:
        raise FileNotFoundError(f"no images found under {d}")
    return torch.stack(imgs, dim=0).to(torch.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="google_ddpm-cifar10-32")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_gen", type=int, default=50000)
    p.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/outputs/cifar10_scaling_with_different_steps",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp",
        help="CIFAR-10 root containing cifar-10-batches-py",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device()

    data_root = Path(args.data_root).expanduser().resolve()
    real_images = _load_cifar10_test_images(data_root)
    print(f"[info] loaded CIFAR10 test images: {tuple(real_images.shape)}")

    dtype = args.dtype
    if dtype == "bf16":
        # diffusers DDPMPipeline converts tensors to numpy internally; numpy does not support bf16.
        print("[warn] bf16 is not supported in DDPMPipeline numpy conversion, fallback to fp16.")
        dtype = "fp16"
    pipe = model_loader(args.model_name, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    base_save = Path(args.save_dir).expanduser().resolve()
    base_save.mkdir(parents=True, exist_ok=True)

    summary: dict[str, list[float | int]] = {
        "deno_step": [],
        "fid": [],
        "is": [],
        "is_std": [],
    }

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

        gen_images = _load_images_from_dir_uint8(out_dir)
        metrics = {
            "fid": float(compute_fid(real_images, gen_images, device=device)),
        }
        is_mean, is_std = compute_is(gen_images, device=device)
        metrics["is"] = float(is_mean)
        metrics["is_std"] = float(is_std)

        metrics_path = out_dir / f"metrics_deno_step_{deno_step}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        summary["deno_step"].append(int(deno_step))
        summary["fid"].append(float(metrics["fid"]))
        summary["is"].append(float(metrics["is"]))
        summary["is_std"].append(float(metrics["is_std"]))

        print(f"[step={deno_step}] metrics={metrics}")

    summary_path = base_save / "metrics_deno_step_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()

