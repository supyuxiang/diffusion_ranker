#!/usr/bin/env python3
"""
用训练好的 CrossSampleDiffusionRanker 对 DiT 生成样本打分。

默认读取:
  - {ranker_dir}/run_config.json  （dit_path、deno_steps、guidance_scale、layer_ratio、ranker 结构等）
  - {ranker_dir}/ranker_diffusion.pt

流程与 train_ranker_imagenet 中 rollout 一致：同 class、group_size 条噪声轨迹 → 捕获指定 block 的 GAP → ranker 前向得到 logits。

示例:
  python3 score_dit_with_ranker.py \\
    --ranker_dir /data/home/yxfeng/diffusion_ranker/outputs/ranker_mse_kl \\
    --class_id 281 \\
    --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
import train_ranker_imagenet as tr  # CrossSampleDiffusionRanker, rollout, resolve_hidden_layer_idx

from utils import load_dit


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ranker_dir",
        type=str,
        default="/data/home/yxfeng/diffusion_ranker/outputs/ranker_mse_kl",
        help="含 run_config.json 与 ranker_diffusion.pt 的目录",
    )
    p.add_argument("--ckpt_name", type=str, default="ranker_diffusion.pt")
    p.add_argument("--class_id", type=int, default=281, help="DiT 条件类 [0,999]")
    p.add_argument("--group_size", type=int, default=None, help="默认用 run_config 里的 group_size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dit_path", type=str, default=None, help="覆盖 run_config 中的 dit_path")
    p.add_argument("--deno_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    args = p.parse_args()

    ranker_dir = Path(args.ranker_dir)
    cfg_path = ranker_dir / "run_config.json"
    ckpt_path = ranker_dir / args.ckpt_name
    if not cfg_path.is_file():
        raise SystemExit(f"缺少配置: {cfg_path}")
    if not ckpt_path.is_file():
        raise SystemExit(f"缺少权重: {ckpt_path}（请先完成训练或检查路径）")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    dit_path = args.dit_path or cfg.get("dit_path")
    if not dit_path:
        raise SystemExit("run_config 中无 dit_path，请传 --dit_path")
    deno_steps = args.deno_steps if args.deno_steps is not None else int(cfg["deno_steps"])
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else float(cfg["guidance_scale"])
    )
    group_size = args.group_size if args.group_size is not None else int(cfg["group_size"])

    if not (0 <= args.class_id < 1000):
        raise SystemExit("class_id 须在 [0, 999]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(cfg.get("fp16", True))

    pipe = load_dit(dit_path, str(device), fp16=fp16)
    n_blocks = len(pipe.transformer.transformer_blocks)
    hidden_layer_idx = tr.resolve_hidden_layer_idx(
        n_blocks,
        cfg.get("layer_ratio", 0.6),
        cfg.get("hidden_layer_idx"),
    )

    ranker = tr.CrossSampleDiffusionRanker(
        dit_hidden_dim=int(cfg["dit_hidden_dim"]),
        proj_dim=int(cfg["ranker_proj_dim"]),
        num_tf_layers=int(cfg["ranker_tf_layers"]),
        n_heads=int(cfg.get("ranker_heads", 4)),
        ff_mult=int(cfg.get("ranker_ff_mult", 4)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    ranker.load_state_dict(ckpt["ranker_state_dict"])
    ranker.eval()

    print(
        f"DiT: {dit_path} | block={hidden_layer_idx}/{n_blocks} | "
        f"group_size={group_size} class={args.class_id} steps={deno_steps} guidance={guidance_scale}"
    )

    with torch.no_grad():
        pil_list, pooled_h = tr.rollout_dit_same_class_pil_and_pooled_hidden(
            pipe,
            group_size,
            args.class_id,
            deno_steps,
            str(device),
            args.seed,
            hidden_layer_idx,
            guidance_scale=guidance_scale,
        )
        h = pooled_h.to(device=device, dtype=torch.float32)
        scores = ranker(h)

    scores_cpu = scores.cpu().tolist()
    print("ranker scores (logits, 未 softmax):")
    for i, (s, img) in enumerate(zip(scores_cpu, pil_list)):
        print(f"  [{i:3d}] {s:+.6f}")
    print(f"mean={sum(scores_cpu)/len(scores_cpu):.6f}  min={min(scores_cpu):.6f}  max={max(scores_cpu):.6f}")


if __name__ == "__main__":
    main()
