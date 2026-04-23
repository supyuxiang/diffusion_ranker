#!/usr/bin/env python3
"""Plot loss_history.jsonl with a clean, publication-style figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> dict[str, np.ndarray]:
    steps, loss, mse, kl = [], [], [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            steps.append(row.get("step", row.get("optimizer_step", len(steps) + 1)))
            loss.append(row["loss"])
            mse.append(row["loss_mse"])
            kl.append(row["loss_kl"])
    return {
        "step": np.array(steps, dtype=np.float64),
        "loss": np.array(loss, dtype=np.float64),
        "loss_mse": np.array(mse, dtype=np.float64),
        "loss_kl": np.array(kl, dtype=np.float64),
    }


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def smart_ylim(
    y: np.ndarray,
    smooth: np.ndarray,
    burnin: int,
    head_steps: int,
    p_lo: float,
    p_hi: float,
) -> tuple[float, float]:
    """Balance early decay vs. later detail: head sets peak/trough of the initial phase;
    tail percentiles keep the axis from stretching to rare spikes after burn-in."""
    n = len(y)
    eh = max(1, min(head_steps, n))
    head_y = y[:eh][np.isfinite(y[:eh])]
    head_s = smooth[:eh][np.isfinite(smooth[:eh])]
    head_parts = [a for a in (head_y, head_s) if a.size]
    head_cat = np.concatenate(head_parts) if head_parts else np.array([], dtype=np.float64)

    b = max(0, min(burnin, n - 1))
    y_tail = y[b:]
    s_tail = smooth[b:]
    if y_tail.size == 0:
        y_tail, s_tail = y, smooth
    tail_parts = [a for a in (y_tail, s_tail) if a.size]
    tail_cat = np.concatenate(tail_parts) if tail_parts else np.array([], dtype=np.float64)
    tail_cat = tail_cat[np.isfinite(tail_cat)]

    allv = np.concatenate([y, smooth])
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return 0.0, 1.0

    hi_head = float(np.max(head_cat)) if head_cat.size else float(np.max(allv))
    hi_tail = float(np.percentile(tail_cat, p_hi)) if tail_cat.size else hi_head
    hi = max(hi_head, hi_tail)

    lo_head = float(np.min(head_cat)) if head_cat.size else float(np.min(allv))
    lo_tail = float(np.percentile(tail_cat, p_lo)) if tail_cat.size else lo_head
    lo = min(lo_head, lo_tail)

    if hi <= lo:
        lo, hi = float(np.min(allv)), float(np.max(allv))
    span = hi - lo if hi > lo else max(abs(hi), 1e-9) * 0.1
    pad = 0.07 * span
    nonneg = bool(np.min(allv) >= 0)
    ymin = max(0.0, lo - pad) if nonneg else lo - pad
    ymax = hi + pad
    return ymin, ymax


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path, help="Path to loss_history.jsonl")
    p.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    p.add_argument("--window", type=int, default=100, help="Rolling mean window (steps)")
    p.add_argument(
        "--burnin-fraction",
        type=float,
        default=0.05,
        help="Fraction of earliest steps skipped before 'tail' percentiles for y-limits (default 5%%)",
    )
    p.add_argument(
        "--burnin-steps",
        type=int,
        default=None,
        help="Fixed burn-in steps for tail percentiles (overrides fraction if set)",
    )
    p.add_argument(
        "--head-fraction",
        type=float,
        default=0.22,
        help="First fraction of steps used to include initial loss range (peak/decay) on y-axis (default 22%%)",
    )
    p.add_argument(
        "--head-steps",
        type=int,
        default=None,
        help="Fixed number of earliest steps for head y-range (overrides head-fraction if set)",
    )
    p.add_argument(
        "--y-percentile",
        type=float,
        nargs=2,
        default=(2.0, 99.0),
        metavar=("P_LO", "P_HI"),
        help="Tail percentiles for y-axis (after burn-in); upper bound is max(head, tail) (default 2 99)",
    )
    p.add_argument(
        "--full-y",
        action="store_true",
        help="Use matplotlib default y scaling (include early spikes in axis range)",
    )
    p.add_argument("--logy", action="store_true", help="Log scale on y (per subplot)")
    args = p.parse_args()

    data = load_jsonl(args.jsonl)
    step = data["step"]
    out = args.out or (args.jsonl.parent / "loss_curve.png")

    # Palette: deep teal / amber / rose on warm off-white
    c_total = "#0d5c63"
    c_mse = "#c45c26"
    c_kl = "#6b4c9a"
    c_grid = "#e8e4dc"
    c_bg = "#faf8f5"

    plt.rcParams.update(
        {
            "figure.facecolor": c_bg,
            "axes.facecolor": c_bg,
            "axes.edgecolor": "#2a2a2a",
            "axes.labelcolor": "#1a1a1a",
            "text.color": "#1a1a1a",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": c_grid,
            "grid.linewidth": 0.9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, dpi=150)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.07, hspace=0.22)

    title = args.jsonl.parent.name
    fig.suptitle(f"Training loss — {title}", fontsize=15, fontweight="600", y=0.98)

    series = [
        ("Total loss", "loss", c_total),
        ("MSE term (loss_mse)", "loss_mse", c_mse),
        ("KL term (loss_kl)", "loss_kl", c_kl),
    ]

    w = max(1, min(args.window, len(step) // 4 or 1))
    n_steps = len(step)
    if args.burnin_steps is not None:
        burnin = max(0, args.burnin_steps)
    else:
        burnin = int(n_steps * max(0.0, min(1.0, args.burnin_fraction)))
    if args.head_steps is not None:
        head_steps = max(1, args.head_steps)
    else:
        head_steps = max(1, int(n_steps * max(0.0, min(1.0, args.head_fraction))))
    p_lo, p_hi = float(args.y_percentile[0]), float(args.y_percentile[1])

    for ax, (label, key, color) in zip(axes, series, strict=True):
        y = data[key]
        ax.plot(step, y, color=color, alpha=0.18, linewidth=0.7, label="per step", rasterized=True)
        smooth = rolling_mean(y, w)
        ax.plot(step, smooth, color=color, linewidth=2.0, label=f"rolling mean (w={w})", zorder=3)
        if args.logy:
            ax.set_yscale("log")
        elif not args.full_y:
            ymin, ymax = smart_ylim(y, smooth, burnin, head_steps, p_lo, p_hi)
            ax.set_ylim(ymin, ymax)
        ax.set_ylabel(label, color="#222")
        ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
        ax.grid(True, linestyle="-", alpha=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Training step")
    fig.savefig(out, dpi=200, facecolor=c_bg, edgecolor="none", bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
