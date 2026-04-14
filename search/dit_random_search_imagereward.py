#!/usr/bin/env python3
"""
example:

  python3 search/dit_random_search_imagereward.py \
      --num_samples 50000 \
      --num_candidates 16 \
      --deno_steps 10 \
      --search_batch_size 32 \
      --guidance_scale 1.0 \
      --output_dir /root/autodl-tmp/random_search_imagereward_50k_cfg1.0_deno10_search32
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
from reward import load_imagereward, imagereward_reward



def save_best_images(best_path_ls: list[Path], output_dir: Path) -> None:
    for path in best_path_ls:
        shutil.copy2(path, output_dir / path.name)


@torch.inference_mode()
def run_random_search_imagereward(
    reward_model,
    num_samples: int,
    search_batch_size: int,
    num_candidates: int,
    candidate_pathes_dict: dict[int, list[Path]],
) -> list[Path]:
    best_path_ls: list[Path] = []
    current_idx = 0
    num_rounds = (num_samples + search_batch_size - 1) // search_batch_size

    for c in range(num_candidates):
        if len(candidate_pathes_dict[c]) < num_samples:
            raise RuntimeError(
                f"candidate_{c:04d}: need >= {num_samples} images, got {len(candidate_pathes_dict[c])}"
            )

    for _ in tqdm(range(num_rounds), desc="random search imagereward"):
        n_this_batch = min(search_batch_size, num_samples - current_idx)
        if n_this_batch <= 0:
            break

        best_score = float("-inf")
        best_candidate_idx = -1

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            sub = candidate_path_ls[current_idx : current_idx + n_this_batch]
            score = imagereward_reward(sub, reward_model)
            if score > best_score:
                best_score = score
                best_candidate_idx = candidate_idx

        assert best_candidate_idx != -1
        best_path_ls.extend(candidate_pathes_dict[best_candidate_idx][
            current_idx : current_idx + n_this_batch
        ])
        current_idx += n_this_batch
        if current_idx >= num_samples:
            break

    return best_path_ls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/root/autodl-tmp/eval_dit_random_search_imagereward")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--search_batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
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

    set_seed(args.seed)
    device = set_device()
    reward_model = load_imagereward(device)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        candidate_pathes_dict[i] = sorted(
            candidate_dir_ls[i].glob("*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

    print(
        f"random search (imagereward): root={rollout_root}, "
        f"num_candidates={args.num_candidates}, search_batch_size={args.search_batch_size}"
    )
    best_path_ls = run_random_search_imagereward(
        reward_model,
        args.num_samples,
        args.search_batch_size,
        args.num_candidates,
        candidate_pathes_dict,
    )
    best_dir = output_dir / "best_images"
    best_dir.mkdir(parents=True, exist_ok=True)
    save_best_images(best_path_ls, best_dir)

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "verifier": "imagereward",
                "imagenet_val": str(imagenet_val),
                "candidates_root": str(rollout_root),
                "num_samples": args.num_samples,
                "search_batch_size": args.search_batch_size,
                "num_candidates": args.num_candidates,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "ref_cache_name": cache_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
