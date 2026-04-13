#!/usr/bin/env python3
"""对 rollout 的 candidate 目录批量算 FID / IS / CLIPScore。

- FID/IS：``utils.compute_fid_is_from_dirs``（torch_fidelity，不整盘载入 val）。
- CLIPScore：``utils.compute_clipscore``；默认目录 ``utils.CLIP_HF_LOCAL_DIR``，可用 ``--clip_model_dir`` 覆盖。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils import (
    compute_clipscore,
    compute_fid_is_from_dirs,
    imagenet_label2text_for_clipscore,
    set_device,
    _load_images_from_dir,
)


def _labels_from_gen_dir(gen_dir: Path) -> list[int]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(
        p for p in gen_dir.iterdir() if p.suffix.lower() in exts and p.is_file()
    )
    return [int(p.stem.split("_")[-1]) for p in paths]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gen_dir_base",
        type=Path,
        default=Path("/root/autodl-tmp/dit_candidates_50k_cfg1.0_deno10"),
        help="rollout 根目录，其下为 candidate_0000, candidate_0001, …",
    )
    p.add_argument(
        "--real_dir",
        type=Path,
        default=Path("/root/autodl-tmp/imagenet/val"),
    )
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--skip_clip",
        action="store_true",
        help="不算 CLIPScore（避免依赖 CLIP 权重与较大内存）",
    )
    p.add_argument(
        "--clip_model_dir",
        type=Path,
        default=None,
        help="HuggingFace CLIP 目录；默认 utils.CLIP_HF_LOCAL_DIR",
    )
    p.add_argument(
        "--clip_batch_size",
        type=int,
        default=64,
        help="CLIP 前向 batch 大小（控制显存）",
    )
    args = p.parse_args()

    device = set_device()
    dev = "cuda" if device.type == "cuda" else "cpu"

    for i in range(args.num_candidates):
        gen_dir = args.gen_dir_base / f"candidate_{i:04d}"
        if not gen_dir.is_dir():
            print(f"skip (not a dir): {gen_dir}", file=sys.stderr)
            continue
        metrics = compute_fid_is_from_dirs(
            gen_dir=gen_dir,
            real_dir=args.real_dir,
            device=dev,
            batch_size=args.batch_size,
            samples_resize_and_crop=256,
            samples_find_deep=True,
        )
        out = {k: float(v) for k, v in metrics.items()}
        if not args.skip_clip:
            labels_ls = _labels_from_gen_dir(gen_dir)
            gen_images = _load_images_from_dir(gen_dir)
            if gen_images.shape[0] != len(labels_ls):
                raise ValueError(
                    f"{gen_dir}: image count {gen_images.shape[0]} vs labels {len(labels_ls)}"
                )
            prompts = imagenet_label2text_for_clipscore(labels_ls)
            out["CLIPScore"] = float(
                compute_clipscore(
                    gen_images,
                    prompts,
                    device=device,
                    clip_model_dir=args.clip_model_dir,
                    clip_batch_size=args.clip_batch_size,
                )
            )
        save_path = gen_dir / "final_metrics.json"
        save_path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        clip_s = ""
        if not args.skip_clip:
            clip_s = f" CLIPScore={out['CLIPScore']:.4f}"
        print(
            f"{gen_dir.name}: FID={out['FID']:.4f} IS={out['IS']:.4f}{clip_s}"
        )


if __name__ == "__main__":
    main()
