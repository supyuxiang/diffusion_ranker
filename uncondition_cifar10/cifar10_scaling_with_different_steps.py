#!/usr/bin/env python3
"""
CIFAR-10 DDPM scaling with different denoising steps (FID/IS only).

默认从 ``/root/autodl-pub/cifar-10`` 读本地 ``cifar-10-python.tar.gz``（或已解压的
``cifar-10-batches-py``），``CIFAR10(..., download=False)`` 不联网。pub 盘只读时会解压到
``--extract_cache``（默认 ``/root/autodl-tmp/cifar-10_from_autodl_pub``）。

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
import errno
import json
import sys
import tarfile
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

_DENO_STEP_LS = [400, 500, 600, 700, 800, 900, 950]

_CIFAR10_PYTHON_TAR = "cifar-10-python.tar.gz"


def _resolve_cifar10_torchvision_root(data_root: Path, extract_cache: Path) -> Path:
    """仅使用本地数据：不触发 torchvision 联网下载。

    ``data_root`` 下若有 ``cifar-10-batches-py`` 则直接使用；否则从同目录的
    ``cifar-10-python.tar.gz`` 解压。若 ``data_root`` 只读（常见 autodl-pub），则解压到
    ``extract_cache``。
    """
    data_root = data_root.expanduser().resolve()
    extract_cache = extract_cache.expanduser().resolve()

    def _has_batches(root: Path) -> bool:
        return (root / "cifar-10-batches-py" / "batches.meta").is_file()

    if _has_batches(data_root):
        return data_root
    if _has_batches(extract_cache):
        return extract_cache

    tarball = data_root / _CIFAR10_PYTHON_TAR
    if not tarball.is_file():
        raise FileNotFoundError(
            f"CIFAR-10 not under {data_root}: need cifar-10-batches-py/ or {_CIFAR10_PYTHON_TAR}"
        )

    def _extract(to_dir: Path) -> None:
        to_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] extracting local {tarball} -> {to_dir}")
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(to_dir)

    try:
        _extract(data_root)
    except OSError as e:
        if e.errno not in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise
        print(f"[info] cannot write under {data_root} ({e}), using extract cache")
        _extract(extract_cache)
        if not _has_batches(extract_cache):
            raise RuntimeError(f"extracted to {extract_cache} but cifar-10-batches-py is incomplete")
        return extract_cache

    if not _has_batches(data_root):
        raise RuntimeError(f"extracted {tarball} to {data_root} but cifar-10-batches-py is incomplete")
    return data_root


def _load_cifar10_test_images(data_root: str | Path) -> torch.Tensor:
    root = Path(data_root).expanduser().resolve()
    ds = CIFAR10(root=str(root), train=False, download=False)
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
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_gen", type=int, default=10000)
    p.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/outputs/cifar10_scaling_with_different_steps",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-pub/cifar-10",
        help="本地 CIFAR-10：含 cifar-10-batches-py/ 或 cifar-10-python.tar.gz（不联网）",
    )
    p.add_argument(
        "--extract_cache",
        type=str,
        default="/root/autodl-tmp/cifar-10_from_autodl_pub",
        help="data_root 只读时，将 cifar-10-python.tar.gz 解压到此目录供 torchvision 读取",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device()

    data_root = Path(args.data_root).expanduser().resolve()
    extract_cache = Path(args.extract_cache).expanduser().resolve()
    cifar_root = _resolve_cifar10_torchvision_root(data_root, extract_cache)
    real_images = _load_cifar10_test_images(cifar_root)
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
            image_size=32,
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

