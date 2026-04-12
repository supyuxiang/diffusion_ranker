"""
DDPM（UNet，如 ``google_ddpm-celebahq-256``）在不同 ``num_inference_steps`` 下生成样本，
对每个步数目录用 ``utils.compute_metrics`` 算 FID / IS（需本地真实人脸参考图目录）。

与 ``dit_scaling_with_different_steps.py`` 不同：DiT 用 ImageNet val + CLIP 类名；此处为**无条件**
256×256，应对齐 CelebA-HQ（或同类）真实集路径。

example::

    CUDA_VISIBLE_DEVICES=0 python /root/diffusion_ranker/unet_scaling_with_different_steps.py \
        --model_name google_ddpm-celebahq-256 \
        --batch_size 32 \
        --num_gen 30000 \
        --save_dir /root/autodl-tmp/outputs/unet_scaling_with_different_steps \
        --real_dir /root/autodl-tmp/celebahq_reference_png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from utils import (
    compute_metrics,
    model_loader,
    rollout_ddpm_with_steps,
    set_device,
    set_seed,
)

_DENO_STEP_LS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name",
        type=str,
        default="google_ddpm-celebahq-256",
        help="见 utils.model_name2path（需本地权重）",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_gen", type=int, default=30000)
    p.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/outputs/unet_scaling_with_different_steps",
    )
    p.add_argument(
        "--real_dir",
        type=str,
        default="/root/autodl-tmp/celebahq_reference_png",
        help="真实图像根目录（递归 .png/.jpg；需自行准备或与 CelebA-HQ 对齐）",
    )
    p.add_argument(
        "--fid_real_max_samples",
        type=int,
        default=30_000,
        help="参考集最多使用张数；设很大时实际受目录内文件数限制",
    )
    p.add_argument("--fid_real_subset_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(pipe.unet)，仅 CUDA 建议尝试",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device_t = set_device()
    device = str(device_t)

    real_dir = Path(args.real_dir).expanduser().resolve()
    if not real_dir.is_dir():
        raise FileNotFoundError(f"--real_dir is not a directory: {real_dir}")

    pipe = model_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    if args.compile and device == "cuda":
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
        print("torch.compile enabled (unet)")

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

        metrics = compute_metrics(
            out_dir,
            real_dir,
            device,
            real_cache_name="celebahq_hq256_ref",
            fid=True,
            isc=True,
            kid=False,
            prc=False,
            real_max_samples=args.fid_real_max_samples,
            real_subset_seed=args.fid_real_subset_seed,
        )
        metrics_serializable = {k: float(v) for k, v in metrics.items()}
        print(f"Done! Saved to {out_dir}")
        print(f"Metrics: {metrics_serializable}")

        metrics_path = out_dir / f"metrics_deno_step_{deno_step}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
