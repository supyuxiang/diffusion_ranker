#!/usr/bin/env python3
"""
DiT Random Search with ImageReward Verifier（仅预生成图）

须先用 ``search/rollout.py`` 生成 ``candidate_0000`` …（``image_{idx}_{label}.png``）与可选 ``init_1024``。
每轮从 K 个池的同一全局下标切片取 ``batch_size`` 张，算 batch 平均 ImageReward，选最高一批写入 ``output_dir``。
**不加载 DiT**。

默认 ``batch_size=32``、``num_candidates=16``（与 CLIP 版一致，可按需改）。

用法::

  python3 search/dit_random_search_imagereward.py \\
      --imagenet_val /root/autodl-tmp/imagenet/val \\
      --output_dir /root/autodl-tmp/eval_dit_rs_imagereward \\
      --deno_steps 10 --guidance_scale 1.0 \\
      --num_candidates 16 --batch_size 32 --init_random_samples 32
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import ensure_image_reward_on_path

ensure_image_reward_on_path()
import ImageReward as RM

from dit_random_search_common import load_imagenet_simple_labels, prompt_for_class

from utils import compute_fid_is_from_dirs, set_device, set_seed


def default_candidates_root(guidance_scale: float, deno_steps: int) -> Path:
    return Path(
        f"/root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}"
    )


def sorted_rollout_image_paths(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def class_id_from_rollout_path(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"expected image_*_*_label.png, got {path.name}")
    return int(parts[2])


def load_images_and_class_ids(paths: list[Path]) -> tuple[list[Image.Image], list[int]]:
    images: list[Image.Image] = []
    cids: list[int] = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB").copy())
        cids.append(class_id_from_rollout_path(p))
    return images, cids


def build_candidate_path_lists(root: Path, num_candidates: int) -> list[list[Path]]:
    lists: list[list[Path]] = []
    for i in range(num_candidates):
        d = root / f"candidate_{i:04d}"
        if not d.is_dir():
            raise FileNotFoundError(f"missing candidate dir: {d}")
        pl = sorted_rollout_image_paths(d)
        if not pl:
            raise FileNotFoundError(f"no image_*_*.png under {d}")
        lists.append(pl)
    return lists


def resolve_init_paths(
    candidates_root: Path,
    init_dir: Path | None,
    n_init: int,
    candidate_lists: list[list[Path]],
) -> list[Path]:
    if n_init <= 0:
        return []
    if init_dir is not None and init_dir.is_dir():
        pl = sorted_rollout_image_paths(init_dir)
        if len(pl) < n_init:
            raise RuntimeError(f"{init_dir}: need >= {n_init} images, got {len(pl)}")
        return pl[:n_init]
    pl0 = candidate_lists[0]
    if len(pl0) < n_init:
        raise RuntimeError(f"candidate_0000: need >= {n_init} images for init, got {len(pl0)}")
    return pl0[:n_init]


def init_from_precomputed(output_dir: Path, init_paths: list[Path]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(init_paths):
        with Image.open(p) as im:
            im.convert("RGB").save(output_dir / f"{i:06d}.png")
    return len(init_paths)


def mean_batch_imagereward(
    reward_model,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
) -> float:
    scores = []
    with torch.no_grad():
        for img, cid in zip(images, class_ids):
            text = prompt_for_class(labels_1k, cid, prompt_template)
            scores.append(float(reward_model.score(text, img)))
    return float(np.mean(scores)) if scores else float("-inf")


@torch.inference_mode()
def run_random_search_imagereward(
    reward_model,
    labels_1k: list[str],
    prompt_template: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    candidate_lists: list[list[Path]],
    n_init: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cursor = n_init
    img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for c in range(num_candidates):
        if len(candidate_lists[c]) < num_samples:
            raise RuntimeError(
                f"candidate_{c:04d}: need >= {num_samples} images, got {len(candidate_lists[c])}"
            )

    for _ in tqdm(range(num_rounds), desc="random search imagereward"):
        n_this = min(batch_size, num_samples - cursor)
        if n_this <= 0:
            break

        best_score = float("-inf")
        best_batch: list[Image.Image] | None = None

        for c in range(num_candidates):
            sub = candidate_lists[c][cursor : cursor + n_this]
            batch_images, batch_classes = load_images_and_class_ids(sub)
            score = mean_batch_imagereward(
                reward_model, batch_images, batch_classes, labels_1k, prompt_template
            )
            if score > best_score:
                best_score = score
                best_batch = batch_images

        assert best_batch is not None
        for img in best_batch:
            if img_counter >= num_samples:
                break
            img.save(output_dir / f"{img_counter:06d}.png")
            img_counter += 1
        cursor += n_this


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--imagenet_val", default="/root/autodl-tmp/imagenet/val")
    p.add_argument("--output_dir", default="/root/autodl-tmp/eval_dit_random_search_imagereward")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--init_random_samples", type=int, default=32)
    p.add_argument("--deno_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument(
        "--candidates_root",
        default="",
        help="预生成根目录；空则用 dit_candidates_50k_cfg{gs}_deno{steps}",
    )
    p.add_argument(
        "--init_dir",
        default="",
        help="init 目录；空则尝试 candidates_root/init_1024，否则 candidate_0000 前 N 张",
    )
    p.add_argument(
        "--prompt_template",
        default="a high-quality picture of a {name}",
        help="{name}、{id} 见 prompt_for_class",
    )
    p.add_argument(
        "--imagereward_ckpt",
        default="ImageReward-v1.0",
        help="传给 ImageReward.load",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()
    if not imagenet_val.is_dir():
        raise FileNotFoundError(f"--imagenet_val not a directory: {imagenet_val}")

    set_seed(args.seed)
    device = set_device()

    labels_1k = load_imagenet_simple_labels()

    print(f"load ImageReward: {args.imagereward_ckpt}")
    reward_model = RM.load(args.imagereward_ckpt)

    cr = (
        Path(args.candidates_root).expanduser().resolve()
        if args.candidates_root
        else default_candidates_root(args.guidance_scale, args.deno_steps)
    )
    if not cr.is_dir():
        raise FileNotFoundError(
            f"candidates root missing: {cr}\nRun search/rollout.py per candidate_idx."
        )

    init_dir_opt = Path(args.init_dir).expanduser().resolve() if args.init_dir else None
    auto_init = cr / "init_1024"
    if init_dir_opt is None and auto_init.is_dir() and sorted_rollout_image_paths(auto_init):
        init_dir_opt = auto_init

    lists = build_candidate_path_lists(cr, args.num_candidates)

    n_init = 0
    if args.init_random_samples > 0:
        init_paths = resolve_init_paths(cr, init_dir_opt, args.init_random_samples, lists)
        print(f"init from disk ({len(init_paths)} images) -> {output_dir}")
        n_init = init_from_precomputed(output_dir, init_paths)

    print(
        f"random search (imagereward): root={cr}, "
        f"num_candidates={args.num_candidates}, batch_size={args.batch_size}"
    )
    run_random_search_imagereward(
        reward_model,
        labels_1k,
        args.prompt_template,
        output_dir,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        lists,
        n_init=n_init,
    )

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    resolved_root = (
        str(Path(args.candidates_root).expanduser().resolve())
        if args.candidates_root
        else str(default_candidates_root(args.guidance_scale, args.deno_steps))
    )
    cfg = {
        "verifier": "imagereward",
        "imagereward_ckpt": args.imagereward_ckpt,
        "init_random_samples": args.init_random_samples,
        "deno_steps": args.deno_steps,
        "guidance_scale": args.guidance_scale,
        "num_candidates": args.num_candidates,
        "batch_size": args.batch_size,
        "prompt_template": args.prompt_template,
        "candidates_root": resolved_root,
        "init_dir": str(Path(args.init_dir).expanduser().resolve()) if args.init_dir else "",
    }
    log_path = output_dir / "config.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"config saved to {log_path}")

    print("\nfinal FID / IS (torch_fidelity)")
    metrics = compute_fid_is_from_dirs(
        output_dir,
        imagenet_val,
        device,
        batch_size=args.metrics_batch_size,
    )
    print(f"   FID: {metrics['FID']:.2f}, IS: {metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}")
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
