#!/usr/bin/env python3
from __future__ import annotations

"""
Train a ranker directly from the merged rollout file produced by
`class-condition/dit_rollout_mc_imagenet.py`.

This script bridges:
1. rollout `.pth` with
   - `hidden_store_target["s{step}_l{layer}"]`
   - `reward_store["s{step}"]`
   - `label_store`
2. pooled candidate representations expected by `ranker/train.py`

Default example:

    CUDA_VISIBLE_DEVICES=0 python /root/diffusion_ranker/class-condition/train_from_rollout_mc_imagenet.py \
        --rollout_pt /root/autodl-tmp/dit_imagenet_hiddens_and_rewards_mc_s1_6_9_l16_per64.pth \
        --step_idx 6 \
        --layer_idx 16 \
        --pooling_mode mean \
        --ranker_type pointwise

Train from another captured step:

    CUDA_VISIBLE_DEVICES=0 python /root/diffusion_ranker/class-condition/train_from_rollout_mc_imagenet.py \
        --step_idx 1 \
        --layer_idx 16 \
        --ranker_type listwise

Use last-token pooling instead of mean pooling:

    CUDA_VISIBLE_DEVICES=0 python /root/diffusion_ranker/class-condition/train_from_rollout_mc_imagenet.py \
        --step_idx 9 \
        --pooling_mode last \
        --ranker_type cross_attention
"""


import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ranker.dataset import load_prompt_repr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare pooled tensors from ImageNet rollout .pth and launch ranker training."
    )
    p.add_argument(
        "--rollout_pt",
        type=str,
        default="/root/autodl-tmp/dit_imagenet_hiddens_and_rewards_mc_s1_6_9_l16_per64.pth",
    )
    p.add_argument("--step_idx", type=int, default=6)
    p.add_argument("--layer_idx", type=int, default=16)
    p.add_argument("--pooling_mode", type=str, default="mean", choices=("mean", "last"))
    p.add_argument(
        "--prompt_repr_pt",
        type=str,
        default="/root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt",
    )
    p.add_argument("--disable_prompt_hstates", action="store_true")
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="If empty, auto-generate a path under /root/autodl-tmp.",
    )
    p.add_argument("--ranker_type", type=str, default="pointwise", choices=("pointwise", "listwise", "cross_attention"))
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_cand", type=int, default=10)
    p.add_argument("--num_candidate_groups", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_min_ratio", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--cuda_visible_devices", type=str, default="0")
    p.add_argument("--extra_args", type=str, default="")
    return p.parse_args()


def _pool_hidden(hidden: torch.Tensor, pooling_mode: str) -> torch.Tensor:
    if hidden.dim() != 3:
        raise ValueError(f"expected hidden tensor with shape (N, T, D), got {tuple(hidden.shape)}")
    if pooling_mode == "mean":
        return hidden.float().mean(dim=1).contiguous()
    if pooling_mode == "last":
        return hidden.float()[:, -1, :].contiguous()
    raise ValueError(f"unknown pooling_mode: {pooling_mode}")


def _auto_output_dir(args: argparse.Namespace) -> Path:
    rollout_stem = Path(args.rollout_pt).expanduser().resolve().stem
    name = (
        f"/root/autodl-tmp/ranker_ckpt_imagenet_rollout_"
        f"{rollout_stem}_{args.ranker_type}_s{int(args.step_idx)}_l{int(args.layer_idx)}_{args.pooling_mode}"
    )
    return Path(name)


def main() -> None:
    args = parse_args()
    rollout_pt = Path(args.rollout_pt).expanduser().resolve()
    if not rollout_pt.is_file():
        raise SystemExit(f"missing rollout pt: {rollout_pt}")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _auto_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = out_dir / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(rollout_pt, map_location="cpu", weights_only=False)
    hidden_store = obj.get("hidden_store_target")
    reward_store = obj.get("reward_store")
    label_store = obj.get("label_store")
    meta = obj.get("meta", {})
    if not isinstance(hidden_store, dict) or not isinstance(reward_store, dict):
        raise SystemExit(f"{rollout_pt}: missing hidden_store_target/reward_store")
    if not isinstance(label_store, torch.Tensor):
        raise SystemExit(f"{rollout_pt}: missing label_store tensor")

    hidden_key = f"s{int(args.step_idx)}_l{int(args.layer_idx)}"
    reward_key = f"s{int(args.step_idx)}"
    if hidden_key not in hidden_store:
        raise SystemExit(f"{rollout_pt}: hidden key not found: {hidden_key}")
    if reward_key not in reward_store:
        raise SystemExit(f"{rollout_pt}: reward key not found: {reward_key}")

    hidden = hidden_store[hidden_key]
    reward = reward_store[reward_key]
    if not isinstance(hidden, torch.Tensor) or not isinstance(reward, torch.Tensor):
        raise SystemExit(f"{rollout_pt}: hidden/reward must be tensors")

    candidate_tensor = _pool_hidden(hidden, args.pooling_mode)
    reward_tensor = reward.float().view(-1).contiguous()
    label_tensor = label_store.long().view(-1).contiguous()
    if candidate_tensor.shape[0] != reward_tensor.shape[0]:
        raise SystemExit(
            f"row mismatch: candidate={candidate_tensor.shape[0]} reward={reward_tensor.shape[0]}"
        )
    if candidate_tensor.shape[0] != label_tensor.shape[0]:
        raise SystemExit(
            f"row mismatch: candidate={candidate_tensor.shape[0]} labels={label_tensor.shape[0]}"
        )

    candidate_repr_pt = prepared_dir / "candidate_repr.pt"
    reward_json = prepared_dir / "reward.json"
    image_dir = prepared_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "tensor": candidate_tensor,
            "hidden_key": hidden_key,
            "reward_key": reward_key,
            "pooling_mode": args.pooling_mode,
            "source_rollout_pt": str(rollout_pt),
        },
        candidate_repr_pt,
    )
    reward_json.write_text(
        json.dumps([float(x) for x in reward_tensor.tolist()], ensure_ascii=False),
        encoding="utf-8",
    )

    has_prompt_hstates = not bool(args.disable_prompt_hstates)
    input_dim_prompt = 1
    prompt_repr_path = ""
    aligned_prompt_pt = prepared_dir / "prompt_repr_aligned.pt"
    if has_prompt_hstates:
        prompt_table = load_prompt_repr(Path(args.prompt_repr_pt).expanduser().resolve()).float()
        if prompt_table.dim() != 2:
            raise SystemExit(f"prompt_repr must be 2D, got {tuple(prompt_table.shape)}")
        if int(label_tensor.min().item()) < 0 or int(label_tensor.max().item()) >= prompt_table.shape[0]:
            raise SystemExit(
                f"label range [{int(label_tensor.min().item())}, {int(label_tensor.max().item())}] "
                f"out of prompt table range [0, {prompt_table.shape[0] - 1}]"
            )
        aligned_prompt = prompt_table[label_tensor]
        torch.save({"tensor": aligned_prompt}, aligned_prompt_pt)
        prompt_repr_path = str(aligned_prompt_pt)
        input_dim_prompt = int(aligned_prompt.shape[1])

    config = {
        "source_rollout_pt": str(rollout_pt),
        "hidden_key": hidden_key,
        "reward_key": reward_key,
        "pooling_mode": args.pooling_mode,
        "candidate_shape": list(candidate_tensor.shape),
        "reward_shape": list(reward_tensor.shape),
        "label_shape": list(label_tensor.shape),
        "has_prompt_hstates": has_prompt_hstates,
        "prompt_repr_pt": str(Path(args.prompt_repr_pt).expanduser().resolve()) if has_prompt_hstates else None,
        "rollout_meta": meta,
    }
    (prepared_dir / "prepare_meta.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "/root/diffusion_ranker/ranker/train.py",
        "--candidate_repr_pt",
        str(candidate_repr_pt),
        "--reward_path",
        str(reward_json),
        "--image_dir",
        str(image_dir),
        "--output_dir",
        str(out_dir),
        "--ranker_type",
        str(args.ranker_type),
        "--num_epochs",
        str(args.num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_cand",
        str(args.num_cand),
        "--num_candidate_groups",
        str(args.num_candidate_groups),
        "--input_dim_candidate",
        str(int(candidate_tensor.shape[1])),
        "--input_dim_prompt",
        str(int(input_dim_prompt)),
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_attention_heads",
        str(args.num_attention_heads),
        "--num_layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
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
        "--save_freq",
        str(args.save_freq),
    ]
    if has_prompt_hstates:
        cmd.extend(["--prompt_repr_pt", prompt_repr_path, "--has_prompt_hstates"])
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args.strip()))

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"Done. ckpt dir -> {out_dir}")


if __name__ == "__main__":
    main()
