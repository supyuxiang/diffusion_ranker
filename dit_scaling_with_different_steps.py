"""
example:
CUDA_VISIBLE_DEVICES=2 python /root/diffusion_ranker/dit_scaling_with_different_steps.py \
    --model_name dit-xl-2-256 \
    --cfg 1.0 \
    --batch_size 512 \
    --num_gen 50000 \
    --save_dir /root/autodl-tmp/outputs/dit_scaling_with_different_steps \
    --imagenet_val /root/autodl-tmp/imagenet/val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    compute_fid_is_clipscore,
    model_loader,
    rollout_dit_with_steps,
    set_device,
    set_seed,
)

_DENO_STEP_LS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_gen", type=int, default=50000)
    p.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/outputs/dit_scaling_with_different_steps",
    )
    p.add_argument(
        "--imagenet_val",
        type=str,
        default="/root/autodl-tmp/imagenet/val",
        help="解压后的 ImageNet val 根目录（与 compute_fid_is_clipscore 对齐）",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device()
    pipe = model_loader(args.model_name, device=device, dtype="fp16")
    pipe.set_progress_bar_config(disable=True)

    base_save = Path(args.save_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()

    out_dir_ls = []

    for deno_step in _DENO_STEP_LS:
        out_dir = base_save / f"deno_step_{deno_step}_rollout"
        out_dir_ls.append(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rollout_dit_with_steps(
            pipe,
            out_dir,
            deno_step,
            args.num_gen,
            args.batch_size,
            device,
            seed=args.seed,
            guidance_scale=args.cfg,
        )
    metrics_path = base_save / "metrics_deno_step_summary.json"
    metrics_dict: dict[str, list] = {
        "deno_step": list(_DENO_STEP_LS),
        "fid": [],
        "is": [],
        "is_std": [],
        "clipscore": [],
    }
    for out_dir in out_dir_ls:
        metrics = compute_fid_is_clipscore(out_dir, imagenet_val, device=device)
        metrics_serializable = {k: float(v) for k, v in metrics.items()}
        for k, v in metrics_serializable.items():
            if k in metrics_dict:
                metrics_dict[k].append(v)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
