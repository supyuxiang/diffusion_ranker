"""
DDPM（UNet，如 ``google_ddpm-celebahq-256``）在不同 ``num_inference_steps`` 下生成样本，
对每个步数目录用 ``utils.compute_fid_is_from_dirs``（``torch_fidelity``）算 FID / IS（需本地真实人脸参考图目录）。

与 ``dit_scaling_with_different_steps.py`` 不同：DiT 用 ImageNet val + CLIP 类名；此处为**无条件**
256×256，应对齐人脸真实集（CelebA-HQ、CelebA 对齐图等；递归读 ``.png`` / ``.jpg``）。

CelebA 对齐图可从机内镜像解压到数据盘（示例）::

    unzip -q -o /root/autodl-pub/CelebA/Img/img_align_celeba.zip -d /root/autodl-tmp

从全量目录随机抽 5 万张作固定 FID/IS 参考集（默认输出目录与脚本 ``--real_dir`` 一致）::

    python3 /root/diffusion_ranker/scripts/build_real_image_subset.py \
        --src /root/autodl-tmp/img_align_celeba \
        --out /root/autodl-tmp/img_align_celeba_fid_ref_50k \
        --n 50000 --seed 42

example::

    CUDA_VISIBLE_DEVICES=2 python /root/diffusion_ranker/unet_scaling_with_different_steps.py \
        --model_name google_ddpm-celebahq-256 \
        --batch_size 512 \
        --num_gen 50000 \
        --save_dir /root/autodl-tmp/outputs/unet_scaling_with_different_steps \
        --real_dir /root/autodl-tmp/img_align_celeba_fid_ref_50k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from utils import (
    compute_fid_is_from_dirs,
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
        default="/root/autodl-tmp/img_align_celeba_fid_ref_50k",
        help="真实图像根目录（默认 5 万张子集；见脚本 scripts/build_real_image_subset.py）",
    )
    p.add_argument(
        "--fid_real_max_samples",
        type=int,
        default=50_000,
        help="参考集最多使用张数（与 5 万子集目录一致；勿超过目录内文件数）",
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
    device = set_device()

    real_dir = Path(args.real_dir).expanduser().resolve()
    if not real_dir.is_dir():
        raise FileNotFoundError(f"--real_dir is not a directory: {real_dir}")

    pipe = model_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    if args.compile and device.type == "cuda":
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

        metrics = compute_fid_is_from_dirs(
            out_dir,
            real_dir,
            device,
            batch_size=64,
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
