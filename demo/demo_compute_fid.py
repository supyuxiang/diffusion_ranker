#!/usr/bin/env python3
"""
FID / Inception Score 小演示：基于 ``torchmetrics`` + ``torch-fidelity``（Inception v3）。

- **toy**：不读磁盘，用两团随机 ``uint8`` 图像张量估 FID，并对一团图算 IS（与官方文档一致）。
- **dirs**：从 ``--real_dir`` / ``--gen_dir`` 读取常见图片后缀，堆成 batch 后算 FID；仅对 ``gen`` 算 IS。

依赖见仓库 ``requirements.txt``（``torchmetrics``, ``torch-fidelity``, ``scipy`` 建议安装）。

示例::

    cd /root/diffusion_ranker
    python demo/demo_compute_fid.py --mode toy
    python demo/demo_compute_fid.py --mode dirs --real_dir demo/outputs_demo/class666 --gen_dir demo/outputs_demo/class666
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def _load_images_from_dir(folder: Path, max_images: int) -> torch.Tensor:
    """Return ``(N, 3, H, W)`` uint8 RGB. Resizes by loader to a common size if mixed."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()
    )[:max_images]
    if not paths:
        raise FileNotFoundError(f"No images in {folder} (supported: {exts})")
    tensors: list[torch.Tensor] = []
    to_t = PILToTensor()
    for p in paths:
        im = Image.open(p).convert("RGB")
        # if im.size != (299, 299):
        #     im = im.resize((299, 299), Image.Resampling.LANCZOS)
        t = to_t(im)
        tensors.append(t)
    batch = torch.stack(tensors, dim=0) # (N, 3, 299, 299)
    if not batch.dtype == torch.uint8:
        batch = batch.to(torch.uint8)
    return batch




def run_dirs(real_dir: Path, gen_dir: Path, max_images: int, fid_feature: int) -> None:
    real = _load_images_from_dir(real_dir, max_images)
    fake = _load_images_from_dir(gen_dir, max_images)
    print(f"Loaded real batch {tuple(real.shape)}, gen batch {tuple(fake.shape)}")

    fid = FrechetInceptionDistance(feature=fid_feature, normalize=False)
    fid.set_dtype(torch.float64)
    # 大 batch 可再拆成子 batch 调用 update，这里整批喂入
    fid.update(real, real=True)
    fid.update(fake, real=False)
    fid_val = float(fid.compute().item())
    print(f"[dirs] FID (feature={fid_feature}): {fid_val:.4f}")

    is_m = InceptionScore(normalize=False, splits=min(10, fake.shape[0]))
    is_m.update(fake)
    is_mean, is_std = is_m.compute()
    print(f"[dirs] IS mean: {float(is_mean):.4f}  std: {float(is_std):.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FID / IS demo (torchmetrics)")
    p.add_argument("--real_dir", type=Path, default=None)
    p.add_argument("--gen_dir", type=Path, default=None)
    p.add_argument("--max_images", type=int, default=128)
    p.add_argument("--fid_feature", type=int, default=2048, choices=(64, 192, 768, 2048))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_dir is None or args.gen_dir is None:
        raise SystemExit("requires --real_dir and --gen_dir")
    run_my_dirs(args.real_dir.resolve(), args.gen_dir.resolve(), args.max_images, args.fid_feature)


if __name__ == "__main__":
    main()
