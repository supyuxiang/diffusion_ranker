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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import compute_fid_is_clipscore, set_device, set_seed
from class_condition.reward import load_clip, clipscore_reward
from class_condition.utils import _sorted_candidate_images, _reset_best_dir, save_best_images

_DEFAULT_CLIP_DIR = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")



# ------------------------------- I love Peking University -------------------------------



@torch.inference_mode()
def run_random_search_clipscore(
    clip: CLIPModel,
    processor: CLIPProcessor,
    num_samples: int,
    search_batch_size: int,
    candidate_pathes_dict: dict[int, list[Path]],
) -> list[Path]:

    start_idx = 0
    num_loops = (num_samples + search_batch_size - 1) // search_batch_size
    best_path_ls: list[Path] = [] # 以便后续save和计算metrics

    for _ in tqdm(range(num_loops), desc="random search clipscore"):
        n_this_batch = min(search_batch_size, num_samples - start_idx)
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
            )
            if reward > best_reward:
                best_reward = reward
                best_candidate_idx = candidate_idx

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
    p.add_argument("--guidance_scale", type=float, default=1)
    p.add_argument("--real_dir", type=Path, default=Path("/root/autodl-tmp/imagenet/val"))
    p.add_argument("--clip_model_dir", type=Path, default=_DEFAULT_CLIP_DIR)
    p.add_argument("--clip_batch_size", type=int, default=64)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    real_dir = args.real_dir.expanduser().resolve()
    clip_model_dir = args.clip_model_dir.expanduser().resolve()

    # Keep directory naming stable for integer-like scales: 1.0 -> "1"
    cfg_tag = f"{float(args.guidance_scale):g}"
    rollout_root = Path(
        f"/root/autodl-tmp/rollout_candidate_step{args.deno_steps}_cfg{cfg_tag}"
    )
    # /root/autodl-tmp/rollout_candidate_step10_cfg1
    
    candidate_dir_ls = [
        rollout_root / f'candidate_{seed}'
        for seed in range(int(args.num_candidates))
    ]

    set_seed(args.seed)
    device = set_device()

    clip, processor = load_clip(device)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for seed in range(int(args.num_candidates)):
        candidate_dir = candidate_dir_ls[seed]
        if not candidate_dir.is_dir():
            raise FileNotFoundError(f"missing candidate dir: {candidate_dir}")
        paths = _sorted_candidate_images(candidate_dir)
        if len(paths) < int(args.num_samples):
            raise ValueError(
                f"{candidate_dir} has only {len(paths)} images, expected >= {int(args.num_samples)}"
            )
        candidate_pathes_dict[seed] = paths

    best_path_ls = run_random_search_clipscore(
        clip,
        processor,
        args.num_samples,
        args.search_batch_size,
        candidate_pathes_dict,
    )

    best_dir = output_dir / "best_images"
    _reset_best_dir(best_dir)
    save_best_images(best_path_ls, best_dir)


    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "verifier": "clipscore",
                "candidates_root": str(rollout_root),
                "real_dir": str(real_dir),
                "num_samples": args.num_samples,
                "search_batch_size": args.search_batch_size,
                "num_candidates": args.num_candidates,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "clip_model_dir": str(clip_model_dir),
                "clip_batch_size": args.clip_batch_size,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Release the selection CLIP before metric computation, which loads another CLIP model.
    del clip, processor
    gc.collect()
    torch.cuda.empty_cache()

    metrics = compute_fid_is_clipscore(
        best_dir,
        real_dir,
        device=device,
        clip_model_dir=clip_model_dir,
        clip_batch_size=args.clip_batch_size,
    )

    metrics["selection_reward"] = "clipscore"
    metrics["num_selected"] = float(len(best_path_ls))
    metrics["num_candidates"] = float(args.num_candidates)
    with open(best_dir / "final_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
