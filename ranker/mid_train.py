#!/usr/bin/env python3
'''
python3 ranker/mid_train.py \
  --layer_ratio 0.6 \
  --step_ratio 0.25 \
  --data_root /root/autodl-tmp/ranker_mid_training_data_mc

'''
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def step_ratio_to_idx(step_ratio: float, num_inference_steps: int) -> int:
    t = int(num_inference_steps)
    if t <= 1:
        return 0
    return min(t - 1, max(0, round(float(step_ratio) * (t - 1))))


def safe_tag(x: float) -> str:
    return f"{float(x):g}".replace(".", "p")


def main() -> None:
    p = argparse.ArgumentParser(description="Train pointwise ranker for one (layer_ratio, step_ratio).")
    p.add_argument("--layer_ratio", type=float, required=True)
    p.add_argument("--step_ratio", type=float, required=True)
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp/ranker_mid_training_data_mc")
    p.add_argument("--prompt_repr_pt", type=str, default="/root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_cand", type=int, default=10)
    p.add_argument("--num_candidate_groups", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda_visible_devices", type=str, default="0")
    p.add_argument("--extra_args", type=str, default="", help="additional raw args to append to ranker/train.py")
    args = p.parse_args()

    step_idx = step_ratio_to_idx(args.step_ratio, args.num_inference_steps)
    layer_tag = safe_tag(args.layer_ratio)
    case_tag = f"l{layer_tag}_s{step_idx}"
    case_dir = Path(args.data_root).expanduser().resolve() / case_tag
    repr_pt = case_dir / "repr_reward.pt"
    reward_json = case_dir / "reward_ls.json"
    image_dir = case_dir / "images"
    out_dir = Path(f"/root/autodl-tmp/ranker_mid_ckpt_pointwise_l{layer_tag}_s{step_idx}")

    if not repr_pt.is_file():
        raise SystemExit(f"missing repr pt: {repr_pt}")
    if not reward_json.is_file():
        raise SystemExit(f"missing reward json: {reward_json}")

    cmd = [
        sys.executable,
        "/root/diffusion_ranker/ranker/train.py",
        "--num_epochs", str(args.num_epochs),
        "--output_dir", str(out_dir),
        "--image_dir", str(image_dir),
        "--candidate_repr_pt", str(repr_pt),
        "--prompt_repr_pt", str(Path(args.prompt_repr_pt).expanduser().resolve()),
        "--reward_path", str(reward_json),
        "--batch_size", str(args.batch_size),
        "--num_cand", str(args.num_cand),
        "--input_dim_candidate", "1152",
        "--input_dim_prompt", "3584",
        "--hidden_dim", str(args.hidden_dim),
        "--num_attention_heads", str(args.num_attention_heads),
        "--num_layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--optimizer", "adamw",
        "--lr_scheduler", "cosine",
        "--warmup_ratio", "0.03",
        "--lr_min_ratio", "0.01",
        "--alpha", "0.8",
        "--tau", "1.0",
        "--seed", str(args.seed),
        "--num_candidate_groups", str(args.num_candidate_groups),
        "--ranker_type", "pointwise",
        "--has_prompt_hstates",
    ]
    if args.extra_args.strip():
        cmd.extend(args.extra_args.strip().split())

    env = dict(**__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"Done. ckpt dir -> {out_dir}")


if __name__ == "__main__":
    main()

