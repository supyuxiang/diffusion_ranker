#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from search.reward import load_imagereward
from text2image.common import dump_json, ensure_dir, list_prompt_dirs


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser("Score candidate pool with ImageReward")
    p.add_argument("--pool_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--benchmark", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    # Keep behavior identical to your existing offline flow.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    out_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    reward_model = load_imagereward(device)
    pdirs = list_prompt_dirs(args.pool_root, args.benchmark or None)
    if not pdirs:
        raise FileNotFoundError(f"no prompt folders under {args.pool_root}")

    all_means: list[float] = []
    for pdir in tqdm(pdirs, desc="imagereward prompts"):
        meta_path = pdir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prompt = str(meta["prompt_text"])
        img_paths = sorted((pdir / "images").glob("image_*.png"))
        vals: list[float] = []
        for ip in img_paths:
            img = Image.open(ip).convert("RGB")
            vals.append(float(reward_model.score(prompt, img)))
        scores = torch.tensor(vals, dtype=torch.float32)
        all_means.append(float(scores.mean().item()) if scores.numel() > 0 else float("nan"))
        prompt_out_dir = out_dir / meta.get("benchmark", "custom")
        prompt_out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt_id": meta["prompt_id"],
                "benchmark": meta.get("benchmark", "custom"),
                "scores": scores,
                "best_idx": int(torch.argmax(scores).item()) if scores.numel() > 0 else -1,
                "mean_score": float(scores.mean().item()) if scores.numel() > 0 else float("nan"),
            },
            prompt_out_dir / f"{meta['prompt_id']}.pt",
        )

    dump_json(
        {
            "reward": "imagereward",
            "num_prompts": len(pdirs),
            "global_mean_of_prompt_means": float(torch.tensor(all_means).nanmean().item()) if all_means else float("nan"),
        },
        out_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
