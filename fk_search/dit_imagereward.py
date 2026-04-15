#!/usr/bin/env python3
r"""
多候选预生成图 + **ImageReward**，用 **FK steering** 选图（对应 ``search/dit_random_search_imagereward.py`` 的数据布局）。

与 ``fk_search/dit_clipscore.py`` 相同 FK 规则；verifier 换为 ``imagereward_reward``。

example::

    python3 fk_search/dit_imagereward.py \\
        --output_dir /root/autodl-tmp/fk_search_imagereward \\
        --num_samples 50000 --num_candidates 16 --deno_steps 10 \\
        --guidance_scale 1.0 --tau 0.05 --fk_resample
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_FK = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "search"))
sys.path.insert(0, str(_FK))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fk_utils import fk_update_state, pick_candidate, save_best_images_preserve_names
from reward import imagereward_reward, load_imagereward
from utils import compute_fid_is_clipscore, set_device, set_seed


@torch.inference_mode()
def run_fk_steering_imagereward(
    reward_model,
    num_samples: int,
    search_batch_size: int,
    candidate_pathes_dict: dict[int, list[Path]],
    *,
    tau: float,
    fk_resample: bool,
    pick: str,
) -> tuple[list[Path], list[dict]]:
    n_expected = num_samples
    start_idx = 0
    num_loops = (n_expected + search_batch_size - 1) // search_batch_size
    k_cand = len(candidate_pathes_dict)
    log_belief = torch.zeros(k_cand, dtype=torch.float64)

    best_path_ls: list[Path] = []
    trace: list[dict] = []

    for loop_i in tqdm(range(num_loops), desc="fk steer imagereward"):
        n_this_batch = min(search_batch_size, n_expected - start_idx)
        if n_this_batch <= 0:
            break

        scores = torch.empty(k_cand, dtype=torch.float64)
        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_paths = candidate_path_ls[start_idx : start_idx + n_this_batch]
            r = float(imagereward_reward(batch_paths, reward_model))
            scores[candidate_idx] = r

        log_belief, w = fk_update_state(
            log_belief, scores, tau=tau, do_resample=fk_resample
        )
        j = pick_candidate(w, pick=pick)
        win_paths = candidate_pathes_dict[j][start_idx : start_idx + n_this_batch]
        best_path_ls.extend(win_paths)
        trace.append(
            {
                "loop": loop_i,
                "start_idx": start_idx,
                "n_batch": n_this_batch,
                "scores": {int(i): float(scores[i]) for i in range(k_cand)},
                "weights": [float(x) for x in w.tolist()],
                "picked_candidate": j,
            }
        )
        start_idx += n_this_batch

    return best_path_ls, trace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FK steering + ImageReward (precomputed candidates)")
    p.add_argument("--output_dir", default="/root/autodl-tmp/fk_search_imagereward")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--search_batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--fk_resample", action="store_true")
    p.add_argument("--pick", type=str, default="sample", choices=("sample", "argmax"))
    p.add_argument("--rollout_root", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path("/root/autodl-tmp/imagenet/val")

    if args.rollout_root.strip():
        rollout_root = Path(args.rollout_root).expanduser().resolve()
    else:
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

    best_path_ls, trace = run_fk_steering_imagereward(
        reward_model,
        args.num_samples,
        args.search_batch_size,
        candidate_pathes_dict,
        tau=args.tau,
        fk_resample=args.fk_resample,
        pick=args.pick,
    )

    best_dir = output_dir / "best_images"
    best_dir.mkdir(parents=True, exist_ok=True)
    save_best_images_preserve_names(best_path_ls, best_dir)

    cfg = {
        "verifier": "imagereward_fk_steering",
        "candidates_root": str(rollout_root),
        "num_samples": args.num_samples,
        "search_batch_size": args.search_batch_size,
        "num_candidates": args.num_candidates,
        "deno_steps": args.deno_steps,
        "guidance_scale": args.guidance_scale,
        "tau": args.tau,
        "fk_resample": args.fk_resample,
        "pick": args.pick,
    }
    (output_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "fk_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics = compute_fid_is_clipscore(best_dir, imagenet_val, device=device)
    (best_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(metrics)


if __name__ == "__main__":
    main()
