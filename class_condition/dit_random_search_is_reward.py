#!/usr/bin/env python3
"""
example:

python3 /root/diffusion_ranker/search/dit_random_search_is_reward.py \
    --output_dir /root/autodl-tmp/dit_random_search_is_ps2_cfg1.0_deno10 \
    --num_samples 50000 \
    --search_batch_size 32 \
    --num_candidates 2 \
    --deno_steps 10 \
    --guidance_scale 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import CenterCrop, Compose, Resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import INCEPTION_WEIGHTS_PATH, compute_fid_is_clipscore, set_device, set_seed
from class_condition.utils import _sorted_candidate_images, _reset_best_dir, save_best_images
from class_condition.reward import is_reward, _build_inception_logits_extractor


# ------------------------------- I love Peking University -------------------------------


def _copy_selected(best_path_ls: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in best_path_ls:
        shutil.copy2(path, output_dir / path.name)




@torch.inference_mode()
def run_random_search_is_reward(
    feat_extractor,
    num_samples: int,
    search_batch_size: int,
    candidate_pathes_dict: dict[int, list[Path]],
    device: torch.device,
) -> tuple[list[Path], list[dict[str, float]]]:
    best_path_ls: list[Path] = []
    round_logs: list[dict[str, float]] = []
    start_idx = 0
    num_loops = (num_samples + search_batch_size - 1) // search_batch_size

    for loop_idx in tqdm(range(num_loops), desc="random search is-reward"):
        n_this_batch = min(search_batch_size, num_samples - start_idx)
        if n_this_batch <= 0:
            break

        best_reward = float("-inf")
        best_candidate_idx = -1
        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_candidate_paths = candidate_path_ls[start_idx : start_idx + n_this_batch]
            reward = is_reward(batch_candidate_paths, feat_extractor, device)
            if reward > best_reward:
                best_reward = reward
                best_candidate_idx = candidate_idx

        best_paths_this_loop = candidate_pathes_dict[best_candidate_idx][
            start_idx : start_idx + n_this_batch
        ]
        best_path_ls.extend(best_paths_this_loop)
        round_logs.append(
            {
                "loop_idx": float(loop_idx),
                "start_idx": float(start_idx),
                "batch_size": float(n_this_batch),
                "winner_candidate_idx": float(best_candidate_idx),
                "winner_reward": float(best_reward),
            }
        )
        start_idx += n_this_batch
    return best_path_ls, round_logs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/root/autodl-tmp/dit_random_search_is_reward")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--search_batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--real_dir", type=Path, default=Path("/root/autodl-tmp/imagenet/val"))
    p.add_argument("--clip_model_dir", type=Path, default=Path("/root/autodl-tmp/openai_clip-vit-large-patch14"))
    p.add_argument("--clip_batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    best_dir = output_dir / "best_images"
    # Keep directory naming stable for integer-like scales: 1.0 -> "1"
    cfg_tag = f"{float(args.guidance_scale):g}"
    rollout_root = Path(
        f"/root/autodl-tmp/rollout_candidate_step{args.deno_steps}_cfg{cfg_tag}"
    )
    candidate_dir_ls = [
        rollout_root / f'candidate_{seed}'
        for seed in range(int(args.num_candidates))
    ]

    set_seed(args.seed)
    device = set_device()
    feat_extractor = _build_inception_logits_extractor(device)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for seed, candidate_dir in enumerate(candidate_dir_ls):
        if not candidate_dir.is_dir():
            raise FileNotFoundError(f"missing candidate dir: {candidate_dir}")
        paths = _sorted_candidate_images(candidate_dir)
        if len(paths) < int(args.num_samples):
            raise ValueError(
                f"{candidate_dir} has only {len(paths)} images, expected >= {int(args.num_samples)}"
            )
        candidate_pathes_dict[seed] = paths

    best_path_ls, round_logs = run_random_search_is_reward(
        feat_extractor,
        args.num_samples,
        args.search_batch_size,
        candidate_pathes_dict,
        device,
    )
    _reset_best_dir(best_dir)
    _copy_selected(best_path_ls, best_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "verifier": "inception_max_softmax_confidence",
        "candidates_root": str(rollout_root),
        "real_dir": str(args.real_dir.expanduser().resolve()),
        "num_samples": args.num_samples,
        "search_batch_size": args.search_batch_size,
        "num_candidates": args.num_candidates,
        "deno_steps": args.deno_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "round_log.json").write_text(
        json.dumps(round_logs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics = compute_fid_is_clipscore(
        best_dir,
        args.real_dir.expanduser().resolve(),
        device=device,
        clip_model_dir=args.clip_model_dir.expanduser().resolve(),
        clip_batch_size=args.clip_batch_size,
    )
    metrics["selection_reward"] = "inception_max_softmax_confidence"
    metrics["num_selected"] = float(len(best_path_ls))
    metrics["num_candidates"] = float(args.num_candidates)
    (best_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
