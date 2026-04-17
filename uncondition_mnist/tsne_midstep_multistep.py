#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMPipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uncondition_mnist.mnist_classifier_reward import load_mnist_classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("One-pass multi-step t-SNE for MNIST middle-step features")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/1aurent_ddpm-mnist")
    p.add_argument(
        "--classifier_ckpt",
        type=str,
        default="/root/autodl-tmp/mnist_classifier/resnet18/classifier_best.pt",
    )
    p.add_argument("--layer_idx", type=int, default=9)
    p.add_argument("--step_indices", type=str, default="10,20,30,40,50,60,70,80,90")
    p.add_argument("--num_samples", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_points_per_class", type=int, default=300)
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iters", type=int, default=1000)
    p.add_argument("--output_png", type=str, default="/root/mnist_tsne_midstep_layer9_steps10-90.png")
    p.add_argument("--output_npz", type=str, default="/root/mnist_tsne_midstep_layer9_steps10-90.npz")
    p.add_argument("--output_json", type=str, default="/root/mnist_tsne_midstep_layer9_steps10-90_meta.json")
    return p.parse_args()


@torch.inference_mode()
def _classify_final_images(classifier: nn.Module, x_final: torch.Tensor) -> torch.Tensor:
    x = (x_final / 2 + 0.5).clamp(0, 1)
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=x.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    logits = classifier(x).float()
    return logits.argmax(dim=-1)


def _parse_steps(step_indices: str, total_steps: int) -> list[int]:
    vals = sorted({int(x.strip()) for x in step_indices.split(",") if x.strip()})
    if not vals:
        raise ValueError("step_indices is empty")
    for s in vals:
        if s < 0 or s >= total_steps:
            raise ValueError(f"step {s} out of range [0, {total_steps-1}]")
    return vals


@torch.inference_mode()
def collect_hidden_and_labels_multistep(
    args: argparse.Namespace, device: torch.device
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    pipe = DDPMPipeline.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    step_indices = _parse_steps(args.step_indices, len(timesteps))

    classifier = load_mnist_classifier(args.classifier_ckpt, device)
    resnet_blocks = [(n, m) for n, m in pipe.unet.named_modules() if m.__class__.__name__ == "ResnetBlock2D"]
    if args.layer_idx < 0 or args.layer_idx >= len(resnet_blocks):
        raise IndexError(f"layer_idx={args.layer_idx} out of range, total={len(resnet_blocks)}")
    layer_name, layer_module = resnet_blocks[args.layer_idx]
    print(f"[info] target layer_idx={args.layer_idx}, name={layer_name}, steps={step_indices}")

    state: dict[str, object] = {"cur_step": -1, "captures": {}}

    def _hook(_m, _inp, _out):
        s = int(state["cur_step"])
        if s not in step_indices:
            return
        t = _out[0] if isinstance(_out, (tuple, list)) else _out
        captures = state["captures"]
        if isinstance(captures, dict) and s not in captures:
            captures[s] = t.detach().float().cpu()

    handle = layer_module.register_forward_hook(_hook)
    n_batches = math.ceil(args.num_samples / args.batch_size)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    all_hidden: dict[int, list[torch.Tensor]] = {s: [] for s in step_indices}
    all_labels: list[torch.Tensor] = []

    for bi in tqdm(range(n_batches), desc="collect one-pass hidden+labels"):
        bsz = min(args.batch_size, args.num_samples - bi * args.batch_size)
        if bsz <= 0:
            break
        x = torch.randn(
            (
                bsz,
                pipe.unet.config.in_channels,
                pipe.unet.config.sample_size,
                pipe.unet.config.sample_size,
            ),
            generator=gen,
            device=device,
            dtype=next(pipe.unet.parameters()).dtype,
        )
        state["captures"] = {}
        for si, t in enumerate(timesteps):
            state["cur_step"] = si
            eps = pipe.unet(x, t).sample
            x = pipe.scheduler.step(eps, t, x, generator=gen).prev_sample

        captures = state["captures"]
        assert isinstance(captures, dict)
        missing = [s for s in step_indices if s not in captures]
        if missing:
            raise RuntimeError(f"missing hidden captures at steps={missing}")

        labels = _classify_final_images(classifier, x).cpu()
        all_labels.append(labels)
        for s in step_indices:
            all_hidden[s].append(captures[s])

    handle.remove()
    hidden_out = {s: torch.cat(all_hidden[s], dim=0) for s in step_indices}
    labels_out = torch.cat(all_labels, dim=0)
    return hidden_out, labels_out


def _balanced_indices(y: np.ndarray, max_points_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx_out = []
    for c in range(10):
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        if idx.size > max_points_per_class:
            idx = rng.choice(idx, size=max_points_per_class, replace=False)
        idx_out.append(idx)
    if not idx_out:
        raise RuntimeError("no samples found for any class")
    return np.concatenate(idx_out, axis=0)


def _tsne_embed(X: np.ndarray, pca_dim: int, perplexity: float, tsne_iters: int, seed: int) -> np.ndarray:
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    pca_dim = int(min(pca_dim, X.shape[0] - 1, X.shape[1]))
    if pca_dim < 2:
        raise RuntimeError(f"pca_dim too small after clipping: {pca_dim}")
    X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=tsne_iters,
        random_state=seed,
    )
    return tsne.fit_transform(X_pca)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_by_step, y_t = collect_hidden_and_labels_multistep(args, device)
    y = y_t.numpy().astype(np.int64)
    keep_idx = _balanced_indices(y, args.max_points_per_class, args.seed)
    y_sub = y[keep_idx]

    step_indices = sorted(hidden_by_step.keys())
    emb_by_step: dict[int, np.ndarray] = {}
    for s in step_indices:
        X = hidden_by_step[s].numpy().reshape(hidden_by_step[s].size(0), -1).astype(np.float32)
        X_sub = X[keep_idx]
        emb_by_step[s] = _tsne_embed(
            X_sub,
            pca_dim=args.pca_dim,
            perplexity=float(args.perplexity),
            tsne_iters=int(args.tsne_iters),
            seed=int(args.seed),
        )
        print(f"[tsne] step={s} done, points={emb_by_step[s].shape[0]}")

    out_png = Path(args.output_png).expanduser().resolve()
    out_npz = Path(args.output_npz).expanduser().resolve()
    out_json = Path(args.output_json).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n = len(step_indices)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    cmap = plt.cm.get_cmap("tab10", 10)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.4 * nrows), dpi=170)
    axes = np.array(axes).reshape(-1)
    for i, s in enumerate(step_indices):
        ax = axes[i]
        Z = emb_by_step[s]
        for c in range(10):
            idx = np.where(y_sub == c)[0]
            if idx.size == 0:
                continue
            ax.scatter(Z[idx, 0], Z[idx, 1], s=7, alpha=0.76, color=cmap(c), label=str(c))
        ax.set_title(f"step={s}")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Pred class", loc="upper center", ncol=10, fontsize=8)
    fig.suptitle(f"MNIST middle-step feature t-SNE (layer_idx={args.layer_idx})", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png)
    plt.close(fig)

    npz_payload = {"labels": y_sub}
    for s in step_indices:
        npz_payload[f"step_{s}"] = emb_by_step[s]
    np.savez(out_npz, **npz_payload)

    meta = {
        "layer_idx": args.layer_idx,
        "step_indices": step_indices,
        "num_samples_collected": int(y.shape[0]),
        "num_samples_tsne": int(y_sub.shape[0]),
        "class_counts_tsne": {str(c): int((y_sub == c).sum()) for c in range(10)},
        "output_png": str(out_png),
        "output_npz": str(out_npz),
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
