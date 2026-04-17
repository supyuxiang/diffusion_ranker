#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uncondition_mnist.midstep_layer9_class_probe import collect_hidden_and_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("t-SNE for MNIST middle-step features by class")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/1aurent_ddpm-mnist")
    p.add_argument(
        "--classifier_ckpt",
        type=str,
        default="/root/autodl-tmp/mnist_classifier/resnet18/classifier_best.pt",
    )
    p.add_argument("--layer_idx", type=int, default=9)
    p.add_argument("--step_idx", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_points_per_class", type=int, default=300)
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iters", type=int, default=1000)
    p.add_argument(
        "--output_png",
        type=str,
        default="/root/autodl-tmp/mnist_tsne_midstep_layer9_step50.png",
    )
    p.add_argument(
        "--output_npz",
        type=str,
        default="/root/autodl-tmp/mnist_tsne_midstep_layer9_step50.npz",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default="/root/autodl-tmp/mnist_tsne_midstep_layer9_step50_meta.json",
    )
    return p.parse_args()


def _balanced_subsample(X: np.ndarray, y: np.ndarray, max_points_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for c in range(10):
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        if idx.size > max_points_per_class:
            idx = rng.choice(idx, size=max_points_per_class, replace=False)
        xs.append(X[idx])
        ys.append(y[idx])
    if not xs:
        raise RuntimeError("no samples found for any class")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse the same hidden extraction and pseudo-label pipeline as the probe experiment.
    X_t, y_t = collect_hidden_and_labels(args, device)
    X = X_t.numpy().reshape(X_t.size(0), -1).astype(np.float32)
    y = y_t.numpy().astype(np.int64)

    X_sub, y_sub = _balanced_subsample(X, y, args.max_points_per_class, args.seed)

    # Normalize before dimensionality reduction.
    X_sub = (X_sub - X_sub.mean(axis=0, keepdims=True)) / (X_sub.std(axis=0, keepdims=True) + 1e-6)
    pca_dim = int(min(args.pca_dim, X_sub.shape[0] - 1, X_sub.shape[1]))
    if pca_dim < 2:
        raise RuntimeError(f"pca_dim too small after clipping: {pca_dim}")
    X_pca = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(X_sub)

    tsne = TSNE(
        n_components=2,
        perplexity=float(args.perplexity),
        init="pca",
        learning_rate="auto",
        max_iter=int(args.tsne_iters),
        random_state=args.seed,
    )
    Z = tsne.fit_transform(X_pca)

    out_png = Path(args.output_png).expanduser().resolve()
    out_npz = Path(args.output_npz).expanduser().resolve()
    out_json = Path(args.output_json).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cmap = plt.cm.get_cmap("tab10", 10)
    plt.figure(figsize=(9, 7), dpi=170)
    for c in range(10):
        idx = np.where(y_sub == c)[0]
        if idx.size == 0:
            continue
        plt.scatter(Z[idx, 0], Z[idx, 1], s=9, alpha=0.78, color=cmap(c), label=str(c))
    plt.title(f"MNIST mid-step features t-SNE (layer={args.layer_idx}, step={args.step_idx})")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(title="Pred class", loc="best", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    np.savez(out_npz, embedding=Z, labels=y_sub)
    class_counts = {str(c): int((y_sub == c).sum()) for c in range(10)}
    meta = {
        "layer_idx": args.layer_idx,
        "step_idx": args.step_idx,
        "num_samples_collected": int(X.shape[0]),
        "num_samples_tsne": int(Z.shape[0]),
        "class_counts_tsne": class_counts,
        "pca_dim": int(pca_dim),
        "perplexity": float(args.perplexity),
        "tsne_iters": int(args.tsne_iters),
        "output_png": str(out_png),
        "output_npz": str(out_npz),
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
