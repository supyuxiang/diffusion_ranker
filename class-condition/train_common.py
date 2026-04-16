#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
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


def build_parser(
    *,
    description: str,
    default_num_layers: int,
    default_num_attention_heads: int,
    default_hidden_dim: int,
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--layer_ratio", type=float, required=True)
    p.add_argument("--step_ratio", type=float, required=True)
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp/ranker_training_data_mc_class_condition")
    p.add_argument(
        "--prompt_repr_pt",
        type=str,
        default="/root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt",
    )
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_cand", type=int, default=10)
    p.add_argument("--num_candidate_groups", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=default_hidden_dim)
    p.add_argument("--num_attention_heads", type=int, default=default_num_attention_heads)
    p.add_argument("--num_layers", type=int, default=default_num_layers)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda_visible_devices", type=str, default="0")
    p.add_argument("--input_dim_candidate", type=int, default=1152)
    p.add_argument("--input_dim_prompt", type=int, default=3584)
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_min_ratio", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--output_prefix", type=str, default="/root/autodl-tmp/ranker_ckpt")
    p.add_argument("--extra_args", type=str, default="", help="additional raw args to append to ranker/train.py")
    return p


def run_train_for_type(args: argparse.Namespace, *, ranker_type: str) -> None:
    step_idx = step_ratio_to_idx(args.step_ratio, args.num_inference_steps)
    layer_tag = safe_tag(args.layer_ratio)
    case_tag = f"l{layer_tag}_s{step_idx}"
    case_dir = Path(args.data_root).expanduser().resolve() / case_tag
    repr_pt = case_dir / "repr_reward.pt"
    reward_json = case_dir / "reward_ls.json"
    image_dir = case_dir / "images"
    out_dir = Path(f"{args.output_prefix}_{ranker_type}_l{layer_tag}_s{step_idx}")

    if not repr_pt.is_file():
        raise SystemExit(f"missing repr pt: {repr_pt}")
    if not reward_json.is_file():
        raise SystemExit(f"missing reward json: {reward_json}")

    cmd = [
        sys.executable,
        "/root/diffusion_ranker/ranker/train.py",
        "--num_epochs",
        str(args.num_epochs),
        "--output_dir",
        str(out_dir),
        "--image_dir",
        str(image_dir),
        "--candidate_repr_pt",
        str(repr_pt),
        "--prompt_repr_pt",
        str(Path(args.prompt_repr_pt).expanduser().resolve()),
        "--reward_path",
        str(reward_json),
        "--batch_size",
        str(args.batch_size),
        "--num_cand",
        str(args.num_cand),
        "--num_candidate_groups",
        str(args.num_candidate_groups),
        "--input_dim_candidate",
        str(args.input_dim_candidate),
        "--input_dim_prompt",
        str(args.input_dim_prompt),
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_attention_heads",
        str(args.num_attention_heads),
        "--num_layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--optimizer",
        str(args.optimizer),
        "--lr_scheduler",
        str(args.lr_scheduler),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--lr_min_ratio",
        str(args.lr_min_ratio),
        "--alpha",
        str(args.alpha),
        "--tau",
        str(args.tau),
        "--seed",
        str(args.seed),
        "--ranker_type",
        ranker_type,
        "--has_prompt_hstates",
    ]
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args.strip()))

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"Done. ckpt dir -> {out_dir}")
