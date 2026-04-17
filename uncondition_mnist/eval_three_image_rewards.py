#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uncondition_mnist.mnist_classifier_reward import (
    reward_brightness_mnist,
    reward_image_entropy_mnist,
    reward_sharpness_mnist,
)

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate brightness/sharpness/image-entropy rewards on image directory")
    p.add_argument("--gen_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--output_json", type=str, default="")
    return p.parse_args()


def _load_paths(gen_dir: Path) -> list[Path]:
    return sorted([p for p in gen_dir.iterdir() if p.is_file() and p.suffix.lower() in _EXTS])


def _stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def main() -> None:
    args = parse_args()
    gen_dir = Path(args.gen_dir).expanduser().resolve()
    if not gen_dir.is_dir():
        raise NotADirectoryError(f"gen_dir does not exist: {gen_dir}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}")

    paths = _load_paths(gen_dir)
    if not paths:
        raise FileNotFoundError(f"no images found under {gen_dir}")

    bright_all: list[torch.Tensor] = []
    sharp_all: list[torch.Tensor] = []
    entr_all: list[torch.Tensor] = []

    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i : i + args.batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        bright_all.append(reward_brightness_mnist(imgs))
        sharp_all.append(reward_sharpness_mnist(imgs))
        entr_all.append(reward_image_entropy_mnist(imgs))

    bright = torch.cat(bright_all, dim=0)
    sharp = torch.cat(sharp_all, dim=0)
    entr = torch.cat(entr_all, dim=0)

    out = {
        "gen_dir": str(gen_dir),
        "num_images": int(len(paths)),
        "brightness": _stats(bright),
        "sharpness": _stats(sharp),
        "image_entropy": _stats(entr),
    }

    out_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (gen_dir / "three_rewards_metrics.json")
    )
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
