#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Convert local CelebA-HQ parquet shards to PNG directory")
    p.add_argument("--src_dir", type=str, default="/root/autodl-tmp/celebahq256_hf")
    p.add_argument("--out_dir", type=str, default="/root/autodl-tmp/celebahq256_png")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--max_images", type=int, default=0, help="0 means all images")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(src_dir.glob("*.parquet")) + sorted((src_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {src_dir}")

    ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in parquet_files],
        split="train",
        streaming=True,
    )
    image_col = "image" if "image" in ds.column_names else ds.column_names[0]
    if args.start_idx > 0:
        ds = ds.skip(args.start_idx)

    saved = 0
    cur_idx = args.start_idx
    pbar = tqdm(desc="convert parquet->png", unit="img")
    for row in ds:
        if args.max_images > 0 and saved >= args.max_images:
            break
        save_path = out_dir / f"{cur_idx:06d}.png"
        if (not args.overwrite) and save_path.exists():
            cur_idx += 1
            continue

        img = row[image_col]
        if isinstance(img, Image.Image):
            pil_img = img
        elif isinstance(img, dict):
            if img.get("bytes") is not None:
                pil_img = Image.open(io.BytesIO(img["bytes"]))
            elif img.get("path"):
                pil_img = Image.open(img["path"])
            else:
                raise ValueError("unsupported image dict format in parquet dataset")
        else:
            pil_img = Image.fromarray(img)
        pil_img.convert("RGB").save(save_path)
        saved += 1
        cur_idx += 1
        pbar.update(1)
    pbar.close()
    print(f"done. converted={saved}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
