#!/usr/bin/env python3

'''
cd /root/diffusion_ranker
python3 ranker/mid_rollout.py \
  --layer_ratios "0.1,0.3,0.6,0.9,1.0" \
  --step_ratios "0.0,0.25,0.5,0.75,1.0" \
  --base_output_dir /root/autodl-tmp/ranker_mid_training_data_mc \
  --mc 8

'''
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import rollout_dit_perclass32_and_save_repr


LAYER_RATIO_LS = [0.1, 0.3, 0.6, 0.9, 1.0]
STEP_RATIO_LS = [0.0, 0.25, 0.5, 0.75, 1.0]


def step_ratio_to_idx(step_ratio: float, num_inference_steps: int) -> int:
    t = int(num_inference_steps)
    if t <= 1:
        return 0
    return min(t - 1, max(0, round(float(step_ratio) * (t - 1))))


def safe_tag(x: float) -> str:
    return f"{float(x):g}".replace(".", "p")


def build_case_tag(layer_ratio: float, step_ratio: float, num_inference_steps: int) -> str:
    s_idx = step_ratio_to_idx(step_ratio, num_inference_steps)
    return f"l{safe_tag(layer_ratio)}_s{s_idx}"


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rollout ranker training data for (layer_ratio, step_ratio) grid."
    )
    p.add_argument(
        "--layer_ratios",
        type=str,
        default="0.1,0.3,0.6,0.9,1.0",
        help="comma-separated layer ratios",
    )
    p.add_argument(
        "--step_ratios",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="comma-separated step ratios",
    )
    p.add_argument("--base_output_dir", type=str, default="/root/autodl-tmp/ranker_mid_training_data_mc")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--mc", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--pooling_mode", type=str, default="mean", choices=("mean", "last"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    layer_ratios = parse_float_list(args.layer_ratios)
    step_ratios = parse_float_list(args.step_ratios)
    if not layer_ratios:
        layer_ratios = LAYER_RATIO_LS
    if not step_ratios:
        step_ratios = STEP_RATIO_LS

    base_output_dir = Path(args.base_output_dir).expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    total = len(layer_ratios) * len(step_ratios)
    case_i = 0
    for layer_ratio in layer_ratios:
        for step_ratio in step_ratios:
            case_i += 1
            case_tag = build_case_tag(layer_ratio, step_ratio, args.num_inference_steps)
            case_dir = base_output_dir / case_tag
            image_dir = case_dir / "images"
            repr_pt = case_dir / "repr_reward.pt"
            reward_json = case_dir / "reward_ls.json"

            if repr_pt.exists() and reward_json.exists() and not args.overwrite:
                print(f"[{case_i}/{total}] [skip] {case_tag}")
                payload = torch.load(repr_pt, map_location="cpu", weights_only=False)
                summary.append(
                    {
                        "case_tag": case_tag,
                        "layer_ratio": layer_ratio,
                        "step_ratio": step_ratio,
                        "repr_key": payload.get("repr_key"),
                        "dir": str(case_dir),
                        "repr_pt": str(repr_pt),
                        "reward_json": str(reward_json),
                    }
                )
                continue

            print(f"[{case_i}/{total}] [run] case={case_tag} layer={layer_ratio} step={step_ratio}")
            case_dir.mkdir(parents=True, exist_ok=True)
            tensor = rollout_dit_perclass32_and_save_repr(
                layer_ratio=layer_ratio,
                step_ratio=step_ratio,
                image_output_dir=image_dir,
                repr_output_pt=repr_pt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                batch_size=args.batch_size,
                seed=args.seed,
                dtype=args.dtype,
                pooling_mode=args.pooling_mode,
                mc=args.mc,
            )
            payload = torch.load(repr_pt, map_location="cpu", weights_only=False)
            reward_t = payload["reward"]
            if not isinstance(reward_t, torch.Tensor) or reward_t.dim() != 1:
                raise RuntimeError(f"{repr_pt}: reward 不是 1D Tensor")
            if reward_t.shape[0] != tensor.shape[0]:
                raise RuntimeError(
                    f"{repr_pt}: reward 行数 {reward_t.shape[0]} != tensor 行数 {tensor.shape[0]}"
                )

            reward_ls = [float(x) for x in reward_t.tolist()]
            reward_json.write_text(json.dumps(reward_ls, ensure_ascii=False), encoding="utf-8")

            case_meta = {
                "case_tag": case_tag,
                "layer_ratio": float(layer_ratio),
                "step_ratio": float(step_ratio),
                "repr_key": payload.get("repr_key"),
                "dir": str(case_dir),
                "image_dir": str(image_dir),
                "repr_pt": str(repr_pt),
                "reward_json": str(reward_json),
                "tensor_shape": list(tensor.shape),
                "reward_shape": list(reward_t.shape),
            }
            (case_dir / "meta.json").write_text(
                json.dumps(case_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            summary.append(case_meta)

    (base_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone. summary -> {base_output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

