#!/usr/bin/env python3
"""从图像目录随机抽取固定数量文件到输出目录（默认符号链接，省空间、快）。"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _collect_images(root: Path, *, recursive: bool) -> list[Path]:
    if recursive:
        paths = [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]
    else:
        paths = [
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]
    return paths


def main() -> None:
    p = argparse.ArgumentParser(
        description="Shuffle and take N images into output dir (symlinks or copy)."
    )
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="源目录（默认只扫一层；加 --recursive 则 rglob）",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="输出目录（若已存在会清空其中的文件后再写入）",
    )
    p.add_argument("--n", type=int, default=50_000, help="抽取数量")
    p.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归收集子目录中的图像",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="复制文件；默认用符号链接指向源文件",
    )
    args = p.parse_args()

    src = args.src.expanduser().resolve()
    if not src.is_dir():
        print(f"error: --src is not a directory: {src}", file=sys.stderr)
        sys.exit(1)

    paths = _collect_images(src, recursive=args.recursive)
    if len(paths) < args.n:
        print(
            f"error: only {len(paths)} images under {src}, need {args.n}",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(args.seed)
    paths = paths.copy()
    rng.shuffle(paths)
    chosen = paths[: args.n]

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for existing in out.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)

    src_abs = src.resolve()
    for img in chosen:
        dest = out / img.name
        if dest.exists():
            # 极小概率重名：用 stem 冲突时加后缀
            stem, suf = img.stem, img.suffix
            k = 0
            while dest.exists():
                k += 1
                dest = out / f"{stem}_{k}{suf}"
        if args.copy:
            shutil.copy2(img, dest)
        else:
            dest.symlink_to(img.resolve())

    mode = "copied" if args.copy else "symlinked"
    print(f"{mode} {len(chosen)} images from {src} -> {out} (seed={args.seed})")


if __name__ == "__main__":
    main()
