#!/usr/bin/env python3
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

from utils import compute_fid, compute_is, ddpm_loader, rollout_ddpm_with_steps, set_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--deno_steps", type=int, default=1000)
    p.add_argument("--num_gen", type=int, default=50000)
    p.add_argument("--real_split", type=str, default="train", choices=("train", "test"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save_root",
        type=str,
        default="/root/autodl-tmp/outputs/ddpm_cifar10_fid_protocol",
    )
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp")
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    return p.parse_args()


def _load_cifar10_images(data_root: str | Path, split: str) -> torch.Tensor:
    is_train = split == "train"
    ds = CIFAR10(root=str(Path(data_root).expanduser().resolve()), train=is_train, download=False)
    arr = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    return arr.to(torch.uint8)


def _load_images_from_dir_uint8(img_dir: str | Path) -> torch.Tensor:
    d = Path(img_dir).expanduser().resolve()
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)
    imgs: list[torch.Tensor] = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        imgs.append(t)
    if not imgs:
        raise FileNotFoundError(f"no generated images found under {d}")
    return torch.stack(imgs, dim=0).to(torch.uint8)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device()

    save_root = Path(args.save_root).expanduser().resolve()
    out_dir = save_root / f"ddpm_steps{args.deno_steps}"
    gen_dir = out_dir / "generated_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    real_images = _load_cifar10_images(args.data_root, args.real_split)
    if args.num_gen < real_images.shape[0]:
        real_images = real_images[: args.num_gen]
    print(f"[info] loaded CIFAR10 {args.real_split} images: {tuple(real_images.shape)}")

    dtype = args.dtype
    if dtype == "bf16":
        # DDPMPipeline contains numpy conversion that does not support bf16.
        print("[warn] bf16 fallback to fp16 for DDPMPipeline.")
        dtype = "fp16"

    pipe = ddpm_loader(Path(args.model_path), device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    n_saved = rollout_ddpm_with_steps(
        pipe=pipe,
        save_dir=gen_dir,
        num_steps=args.deno_steps,
        num_samples=args.num_gen,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        image_size=32,
    )
    print(f"[info] generated images saved: {n_saved} -> {gen_dir}")

    gen_images = _load_images_from_dir_uint8(gen_dir)
    if gen_images.shape[0] > real_images.shape[0]:
        gen_images = gen_images[: real_images.shape[0]]
    elif gen_images.shape[0] < real_images.shape[0]:
        real_images = real_images[: gen_images.shape[0]]

    fid = float(compute_fid(real_images, gen_images, device=device))
    is_mean, is_std = compute_is(gen_images, device=device)
    metrics = {
        "fid": fid,
        "is": float(is_mean),
        "is_std": float(is_std),
        "num_gen": int(args.num_gen),
        "num_eval": int(gen_images.shape[0]),
        "deno_steps": int(args.deno_steps),
        "real_split": args.real_split,
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "generated_dir": str(gen_dir),
    }

    metrics_path = out_dir / "metrics_fid_protocol.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[done] metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
