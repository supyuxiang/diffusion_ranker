#!/usr/bin/env python3
"""
example:

python3 /root/diffusion_ranker/search/dit_random_search_clipscore.py \
    --output_dir /root/autodl-tmp/dit_random_search_clipscore_reward \
    --num_samples 50000 \
    --deno_steps 10 \
    --guidance_scale 1.0 \
    --num_candidates 16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import compute_fid_is_clipscore, set_device, set_seed
from reward import load_clip, clipscore_reward, labels_1k

_DEFAULT_CLIP_DIR = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")


def save_best_images(best_path_ls: list[Path], output_dir: Path) -> None:
    for path in best_path_ls:
        copy_path = output_dir / path.name
        shutil.copy2(path,copy_path)
    print(f"saved {len(best_path_ls)} best images to {output_dir}")


@torch.inference_mode()
def run_random_search_clipscore(
    clip: CLIPModel,
    processor: CLIPProcessor,
    labels_1k: list[str],
    num_samples: int,
    search_batch_size: int,
    candidate_pathes_dict: dict[int, list[Path]],
) -> list[Path]:
    n_expected = num_samples
    start_idx = 0

    num_loops = (n_expected + search_batch_size - 1) // search_batch_size
    
    best_path_ls: list[Path] = [] # 以便后续save和计算metrics
    for _ in tqdm(range(num_loops), desc="random search clipscore"):
        n_this_batch = min(search_batch_size, n_expected - start_idx)
        if n_this_batch <= 0:
            break

        best_reward = float("-inf")
        best_candidate_idx = -1

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_candidate_paths = candidate_path_ls[start_idx : start_idx + n_this_batch]

            reward = clipscore_reward(
                batch_candidate_paths,
                clip,
                processor,
                labels_1k,
            )
            if reward > best_reward:
                best_reward = reward
                best_candidate_idx = candidate_idx

        assert best_candidate_idx != -1
        win_paths_this_loop = candidate_pathes_dict[best_candidate_idx][
            start_idx : start_idx + n_this_batch
        ]
        best_path_ls.extend(win_paths_this_loop)
        start_idx += n_this_batch

    return best_path_ls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/root/autodl-tmp/random_search_clipscore")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument(
        "--search_batch_size",
        type=int,
        default=32,
        help="每轮每个 candidate 池切片长度（search batch，默认 32）",
    )
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10, help="与 rollout 目录名一致")
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

    clip, processor = load_clip(device)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        candidate_pathes_dict[i] = sorted(
            candidate_dir_ls[i].glob("image_*_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

    best_path_ls = run_random_search_clipscore(
        clip,
        processor,
        labels_1k,
        args.num_samples,
        args.search_batch_size,
        candidate_pathes_dict,
    )

    best_dir = output_dir / "best_images"
    best_dir.mkdir(exist_ok=True,parents=True)
    save_best_images(best_path_ls, best_dir)


    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "verifier": "clipscore",
                "num_samples": args.num_samples,
                "search_batch_size": args.search_batch_size,
                "num_candidates": args.num_candidates,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # compute metrics
    from utils import compute_fid_is_clipscore
    import json
    metrics = compute_fid_is_clipscore(best_dir, device=device)
    with open(best_dir / "final_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(metrics)


if __name__ == "__main__":
    main()
