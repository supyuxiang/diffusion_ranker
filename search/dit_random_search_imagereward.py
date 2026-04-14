#!/usr/bin/env python3
"""
DiT Random Search + ImageReward verifier（仅读预生成图）。init 固定 N_INIT=1024（与 FID / CLIP 脚本一致）。

须先用 ``search/rollout.py`` 生成 ``candidate_0000`` …（``image_{idx}_{label}.png``）与 ``init1024/``。

每轮从 K 个池的同一位置切片取 ``batch_size`` 张，算 batch 平均 ImageReward，选最高一批；
胜出路径记入 ``best_path_ls``，再 ``shutil.copy2`` 到 ``best_images/``，文件名为
``image_{序}_{class_id}.png``（与 rollout 一致，便于后续按类评估）。

默认 ``--batch_size=32``、``--num_candidates=16``。

用法::

  python3 search/dit_random_search_imagereward.py \
      --output_dir /root/autodl-tmp/eval_dit_rs_imagereward \
      --deno_steps 10 --guidance_scale 1.0 \
      --num_candidates 16 --batch_size 32
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import ensure_image_reward_on_path

ensure_image_reward_on_path()
import ImageReward as RM

from dit_random_search_common import load_imagenet_simple_labels, prompt_for_class

from utils import compute_fid_is_from_dirs, set_device, set_seed

N_INIT = 1024


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


def save_init_images(init_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(init_paths):
        cid = class_id_from_rollout_path(path)
        dest = output_dir / f"image_{i:05d}_{cid:05d}.png"
        shutil.copy2(path, dest)


def save_best_images(best_path_ls: list[Path], output_dir: Path, *, start_idx: int = 0) -> None:
    for j, path in enumerate(best_path_ls):
        i = start_idx + j
        cid = class_id_from_rollout_path(path)
        dest = output_dir / f"image_{i:05d}_{cid:05d}.png"
        shutil.copy2(path, dest)


@torch.inference_mode()
def run_random_search_imagereward(
    reward_model,
    labels_1k: list[str],
    prompt_template: str,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    candidate_pathes_dict: dict[int, list[Path]],
    n_init: int,
) -> list[Path]:
    best_path_ls: list[Path] = []
    current_idx = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for c in range(num_candidates):
        if len(candidate_pathes_dict[c]) < num_samples:
            raise RuntimeError(
                f"candidate_{c:04d}: need >= {num_samples} images, got {len(candidate_pathes_dict[c])}"
            )

    for _ in tqdm(range(num_rounds), desc="random search imagereward"):
        n_this_batch = min(batch_size, num_samples - current_idx)
        if n_this_batch <= 0:
            break

        best_score = float("-inf")
        best_candidate_idx = -1

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            sub = candidate_path_ls[current_idx : current_idx + n_this_batch]
            batch_images, batch_classes = load_images_and_class_ids(sub)
            score = mean_batch_imagereward(
                reward_model,
                batch_images,
                batch_classes,
                labels_1k,
                prompt_template,
            )
            if score > best_score:
                best_score = score
                best_candidate_idx = candidate_idx

        assert best_candidate_idx != -1
        win_paths = candidate_pathes_dict[best_candidate_idx][
            current_idx : current_idx + n_this_batch
        ]
        best_path_ls.extend(win_paths)
        current_idx += n_this_batch
        if current_idx >= num_samples:
            break

    return best_path_ls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/root/autodl-tmp/eval_dit_random_search_imagereward")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument(
        "--prompt_template",
        default="a photo of a {name}",
        help="{name}、{id} 见 prompt_for_class",
    )
    p.add_argument(
        "--imagereward_ckpt",
        default="ImageReward-v1.0",
        help="传给 ImageReward.load",
    )
    return p.parse_args()


def main() -> None:
    imagenet_val = Path("/root/autodl-tmp/imagenet/val")
    cache_name = "imagenet_val"

    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    rollout_root = Path(
        f"/root/autodl-tmp/dit_candidates_50k_cfg{args.guidance_scale}_deno{args.deno_steps}"
    )
    candidate_dir_ls = [
        rollout_root / f"candidate_{i:04d}" for i in range(args.num_candidates)
    ]
    init_dir = rollout_root / "init1024"

    assert all(
        len(list(candidate_dir.glob("image_*_*.png"))) == args.num_samples
        for candidate_dir in candidate_dir_ls
    ) and len(list(init_dir.glob("image_*_*.png"))) == N_INIT, (
        f"expected each candidate to have {args.num_samples} images and init {N_INIT}; "
        f"check rollout_root={rollout_root}"
    )

    set_seed(args.seed)
    device = set_device()

    labels_1k = load_imagenet_simple_labels()

    print(f"load ImageReward: {args.imagereward_ckpt}")
    reward_model = RM.load(args.imagereward_ckpt)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        candidate_pathes_dict[i] = sorted(
            candidate_dir_ls[i].glob("image_*_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

    init_path_ls = sorted(
        init_dir.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    best_dir = output_dir / "best_images"
    print(f"init {N_INIT} -> {best_dir}")
    save_init_images(init_path_ls, best_dir)

    print(
        f"random search (imagereward): root={rollout_root}, "
        f"num_candidates={args.num_candidates}, batch_size={args.batch_size}"
    )
    best_path_ls = run_random_search_imagereward(
        reward_model,
        labels_1k,
        args.prompt_template,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        candidate_pathes_dict,
        N_INIT,
    )
    save_best_images(best_path_ls, best_dir, start_idx=N_INIT)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "verifier": "imagereward",
                "imagereward_ckpt": args.imagereward_ckpt,
                "imagenet_val": str(imagenet_val),
                "candidates_root": str(rollout_root),
                "init_dir": str(init_dir),
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "num_candidates": args.num_candidates,
                "n_init": N_INIT,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "ref_cache_name": cache_name,
                "prompt_template": args.prompt_template,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nfinal FID / IS (torch_fidelity)")
    metrics = compute_fid_is_from_dirs(
        best_dir,
        imagenet_val,
        device,
        batch_size=args.metrics_batch_size,
    )
    print(f"   FID: {metrics['FID']:.2f}, IS: {metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}")
    (best_dir / "final_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
