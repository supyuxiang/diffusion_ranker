#!/usr/bin/env python3
"""
Evaluate score correlation between a trained ranker and ImageReward, then plot a k-sweep curve.

Example:
  python3 eval_ranker_imagereward_correlation.py \
    --ranker_ckpt /data/home/yxfeng/diffusion_ranker/outputs/ranker_mse_kl/g32_alpha0.3_layer_ratio0.6_lr1e-4_cosine_0328/ranker_diffusion_epoch_0030.pt \
    --imagereward_root /data/home/yxfeng/ImageReward-main \
    --pool_size 1000 \
    --num_trials 200 \
    --ks 2,4,8,16,32,64,128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import train_ranker_imagenet as tr
from utils import load_dit

NUM_IMAGENET_CLASSES = getattr(tr, "NUM_IMAGENET_CLASSES", 1000)
EPS = 1e-12


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, 1-based ranks."""
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_vals = values[order]

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom < EPS:
        return 0.0
    return float(np.sum(x * y) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def resolve_ranker_paths(args: argparse.Namespace) -> tuple[Path, str]:
    if args.ranker_ckpt:
        ckpt_path = Path(args.ranker_ckpt)
        return ckpt_path.parent, ckpt_path.name
    if not args.ranker_dir:
        raise SystemExit("Please provide either --ranker_ckpt or --ranker_dir")
    return Path(args.ranker_dir), args.ckpt_name


def load_cfg_and_ranker(ranker_dir: Path, ckpt_name: str, device: torch.device):
    cfg_path = ranker_dir / "run_config.json"
    ckpt_path = ranker_dir / ckpt_name
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    fp16 = bool(cfg.get("fp16", True))
    pipe = load_dit(cfg["dit_path"], str(device), fp16=fp16)
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
    return cfg, pipe, ranker, hidden_layer_idx, n_blocks


def build_fixed_pool(
    pipe,
    *,
    pool_size: int,
    deno_steps: int,
    guidance_scale: float,
    hidden_layer_idx: int,
    device: str,
    pool_seed: int,
) -> tuple[torch.Tensor, list, list[int]]:
    rows = []
    pils: list = []
    cids: list[int] = []
    desc = "DiT pool" if pool_size > NUM_IMAGENET_CLASSES else "DiT pool (1 img / class)"
    for i in tqdm(range(pool_size), desc=desc):
        cid = i % NUM_IMAGENET_CLASSES
        seed = pool_seed + i * 10007
        pil_list, pooled_h = tr.rollout_dit_same_class_pil_and_pooled_hidden(
            pipe,
            1,
            cid,
            deno_steps,
            device,
            seed,
            hidden_layer_idx,
            guidance_scale=guidance_scale,
        )
        rows.append(pooled_h[0].detach().cpu().float())
        pils.append(pil_list[0])
        cids.append(cid)
    h_all = torch.stack(rows, dim=0)
    return h_all, pils, cids


def run_trials_for_k(
    *,
    k: int,
    pool_size: int,
    num_trials: int,
    pool_h: torch.Tensor,
    pool_pils: list,
    pool_cids: list[int],
    labels_1k: list[str],
    prompt_template: str,
    teacher,
    ranker: torch.nn.Module,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[list[float], list[float], list[dict]]:
    pearsons: list[float] = []
    spearmans: list[float] = []
    details: list[dict] = []

    for t in tqdm(range(num_trials), desc=f"trials k={k}", leave=False):
        idx = rng.choice(pool_size, size=k, replace=False)
        h_sub = pool_h[idx].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            ranker_scores = ranker(h_sub).float().cpu().numpy()

        ir_scores = np.empty(k, dtype=np.float64)
        for j, global_i in enumerate(idx):
            prompt = tr.prompt_for_class(labels_1k, pool_cids[int(global_i)], prompt_template)
            with torch.inference_mode():
                ir_scores[j] = float(teacher.score(prompt, pool_pils[int(global_i)]))

        p = _pearson(ranker_scores, ir_scores)
        s = _spearman(ranker_scores, ir_scores)
        pearsons.append(p)
        spearmans.append(s)
        details.append(
            {
                "trial": t,
                "k": k,
                "indices": idx.tolist(),
                "pearson": p,
                "spearman": s,
            }
        )
    return pearsons, spearmans, details


def _plot_curve(
    ks: list[int],
    pearson_mean: list[float],
    spearman_mean: list[float],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=180)

    ax.plot(ks, pearson_mean, marker="o", linewidth=2.2, markersize=5.5, label="Pearson")
    ax.plot(ks, spearman_mean, marker="s", linewidth=2.2, markersize=5.5, label="Spearman")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Number of Candidates (k)")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ranker_ckpt", type=str, default=None, help="Full path to ranker .pt file")
    p.add_argument("--ranker_dir", type=str, default=None, help="Directory containing run_config.json and ckpt")
    p.add_argument("--ckpt_name", type=str, default="ranker_diffusion.pt")
    p.add_argument("--imagereward_root", type=str, default="/data/home/yxfeng/ImageReward-main")
    p.add_argument("--pool_size", type=int, default=1000)
    p.add_argument("--num_trials", type=int, default=200, help="Trials per k")
    p.add_argument("--ks", type=str, default="2,4,8,16,32,64,128")
    p.add_argument("--pool_seed", type=int, default=42)
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--deno_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--out_plot", type=str, default=None)
    p.add_argument("--save_trial_details", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ranker_dir, ckpt_name = resolve_ranker_paths(args)
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    if not ks:
        raise SystemExit("--ks parsed empty")
    for k in ks:
        if k < 2:
            raise SystemExit(f"k must be >= 2, got {k}")
        if k > args.pool_size:
            raise SystemExit(f"k={k} cannot exceed pool_size={args.pool_size}")

    ir_root = Path(args.imagereward_root)
    if not ir_root.is_dir():
        raise SystemExit(f"ImageReward root not found: {ir_root}")
    if str(ir_root) not in sys.path:
        sys.path.insert(0, str(ir_root))

    import ImageReward as RM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, pipe, ranker, hidden_layer_idx, n_blocks = load_cfg_and_ranker(ranker_dir, ckpt_name, device)
    deno_steps = args.deno_steps if args.deno_steps is not None else int(cfg["deno_steps"])
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else float(cfg["guidance_scale"])
    )
    ir_device = cfg.get("imagereward_device") or str(device)
    labels_1k = tr.load_imagenet_simple_labels()
    prompt_template = str(cfg.get("prompt_template", "a high-quality picture of a {name}"))

    teacher = RM.load("ImageReward-v1.0", device=ir_device)
    teacher.eval()

    print(
        f"Build fixed pool: pool_size={args.pool_size}, deno_steps={deno_steps}, guidance={guidance_scale}, "
        f"capture_block={hidden_layer_idx}/{n_blocks}"
    )
    with torch.no_grad():
        pool_h, pool_pils, pool_cids = build_fixed_pool(
            pipe,
            pool_size=args.pool_size,
            deno_steps=int(deno_steps),
            guidance_scale=float(guidance_scale),
            hidden_layer_idx=hidden_layer_idx,
            device=str(device),
            pool_seed=args.pool_seed,
        )

    all_rows = []
    pearson_curve = []
    spearman_curve = []
    for k in tqdm(ks, desc="k sweep (correlation)"):
        rng = np.random.default_rng(int(args.eval_seed) + k * 1_000_003)
        pearsons, spearmans, details = run_trials_for_k(
            k=k,
            pool_size=args.pool_size,
            num_trials=args.num_trials,
            pool_h=pool_h,
            pool_pils=pool_pils,
            pool_cids=pool_cids,
            labels_1k=labels_1k,
            prompt_template=prompt_template,
            teacher=teacher,
            ranker=ranker,
            device=device,
            rng=rng,
        )
        p_mean = float(np.mean(pearsons))
        s_mean = float(np.mean(spearmans))
        pearson_curve.append(p_mean)
        spearman_curve.append(s_mean)

        row = {
            "k": k,
            "num_trials": args.num_trials,
            "pearson_mean": p_mean,
            "pearson_std": float(np.std(pearsons)),
            "spearman_mean": s_mean,
            "spearman_std": float(np.std(spearmans)),
        }
        if args.save_trial_details:
            row["trials"] = details
        all_rows.append(row)

        print(
            f"k={k:3d} | Pearson={row['pearson_mean']:.4f}±{row['pearson_std']:.4f} | "
            f"Spearman={row['spearman_mean']:.4f}±{row['spearman_std']:.4f}"
        )

    default_json = ranker_dir / f"eval_ranker_imagereward_correlation_k_sweep_{Path(ckpt_name).stem}.json"
    default_plot = ranker_dir / f"eval_ranker_imagereward_correlation_k_sweep_{Path(ckpt_name).stem}.png"
    out_json = Path(args.out_json) if args.out_json else default_json
    out_plot = Path(args.out_plot) if args.out_plot else default_plot

    payload = {
        "mode": "k_sweep_correlation",
        "ranker_dir": str(ranker_dir.resolve()),
        "ckpt_name": ckpt_name,
        "imagereward_root": str(ir_root.resolve()),
        "pool_size": args.pool_size,
        "num_trials_per_k": args.num_trials,
        "ks": ks,
        "pool_seed": args.pool_seed,
        "eval_seed": args.eval_seed,
        "deno_steps": int(deno_steps),
        "guidance_scale": float(guidance_scale),
        "hidden_layer_idx": int(hidden_layer_idx),
        "results": all_rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {out_json}")

    _plot_curve(
        ks,
        pearson_curve,
        spearman_curve,
        out_plot,
        title=f"Ranker vs ImageReward Correlation (ckpt={Path(ckpt_name).stem})",
    )
    print(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()
