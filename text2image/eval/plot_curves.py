#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser("Plot score curve by method")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out_png", type=str, required=True)
    args = p.parse_args()

    rows = []
    with Path(args.csv).expanduser().resolve().open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((r["method"], float(r["score"])))
            except Exception:
                continue

    if not rows:
        raise RuntimeError("no rows to plot")

    xs = list(range(len(rows)))
    ys = [x[1] for x in rows]
    labels = [x[0] for x in rows]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, marker="o")
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylabel("score")
    plt.tight_layout()
    out = Path(args.out_png).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
