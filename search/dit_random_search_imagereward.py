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
    start_idx = 0
    num_loops = (num_samples + search_batch_size - 1) // search_batch_size

    for _ in tqdm(range(num_loops), desc="random search imagereward"):
        n_this_batch = min(search_batch_size, num_samples - start_idx)
        if n_this_batch <= 0:
            break

        best_reward = float("-inf")
        best_candidate_idx = -1

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_candidate_paths = candidate_path_ls[start_idx : start_idx + n_this_batch]
            reward = imagereward_reward(batch_candidate_paths, reward_model)
            if reward > best_reward:
                best_reward = reward
                best_candidate_idx = candidate_idx

        assert best_candidate_idx != -1
        best_paths_this_loop = candidate_pathes_dict[best_candidate_idx][
            start_idx : start_idx + n_this_batch
        ]
        best_path_ls.extend(best_paths_this_loop)
        start_idx += n_this_batch

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
                "candidates_root": str(rollout_root),
                "num_samples": args.num_samples,
                "search_batch_size": args.search_batch_size,
                "num_candidates": args.num_candidates,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metrics = compute_imagereward_metrics(best_dir, device)
    with open(best_dir / "final_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(metrics)


if __name__ == "__main__":
    main()
