#!/usr/bin/env python3
"""
DiT 生成过程中间层表示的 t-SNE 可视化

Representation 定义（与 paper_methodology.tex 一致）:
- 在最终去噪步 t=0，从 transformer 层 ℓ 提取 hidden states: H_i^(ℓ) ∈ R^{L×D}
  - L: patch token 数 (如 32×32 latent, patch=2 → 16×16=256)
  - D: hidden dimension
- 对 token 维做 global average pooling 得到 f_i^(ℓ) = (1/L) Σ_j H_i^(ℓ)[j] ∈ R^D
- 该 pooled representation 用于 ranker 等下游任务


CUDA_VISIBLE_DEVICES=7 python3 dit_tsne_visualization.py --/data/home/yxfeng/project1/idm/max_tsne_samples 20000 --pca_dims 0

"""
import os
# 避免 OpenBLAS 多线程导致的内存错误和段错误
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
import torch
import numpy as np
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

MODEL_DIR = Path("/data/yxfeng/models/dit_model")
OUTPUT_DIR = Path("/data/home/yxfeng/project1/idm/dit_tsne_output")


def load_dit(offline: bool = True):
    pipe = DiTPipeline.from_pretrained(
        str(MODEL_DIR), local_files_only=offline, torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def extract_multi_layer_repr(
    pipe,
    class_ids,
    samples_per_class,
    layer_indices,
    num_inference_steps,
    device,
    seed,
):
    """
    从多个 transformer block 提取最终去噪步 t=0 的 hidden states，
    按论文 Eq. 对 token 维做 global average pooling 得到 f_i^(ℓ)。

    返回: (reprs, labels_class, labels_layer)
    - reprs: [N, D] 即 f_i^(ℓ)，mean-pooled 表示
    - labels_class: [N] 类别 id
    - labels_layer: [N] 层索引 ℓ
    """
    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size
    num_blocks = len(pipe.transformer.transformer_blocks)

    # 存储每层的输出
    layer_outputs = {i: None for i in layer_indices}

    def make_hook(layer_idx):
        def hook(module, input, output):
            layer_outputs[layer_idx] = output.detach().float()
        return hook

    handles = []
    for idx in layer_indices:
        h = pipe.transformer.transformer_blocks[idx].register_forward_hook(make_hook(idx))
        handles.append(h)

    all_reprs = []
    all_class_labels = []
    all_layer_labels = []

    total = len(class_ids) * samples_per_class
    pbar = tqdm(total=total, desc="Extracting representations")
    for class_id in class_ids:
        for s in range(samples_per_class):
            pbar.update(1)
            gen_seed = seed + class_id * 1000 + s
            generator = torch.Generator(device=device).manual_seed(gen_seed)
            latents = randn_tensor(
                (1, latent_channels, latent_size, latent_size),
                generator=generator,
                device=device,
                dtype=pipe.transformer.dtype,
            )
            class_labels = torch.tensor([class_id], device=device)
            pipe.scheduler.set_timesteps(num_inference_steps)

            for t in pipe.scheduler.timesteps:
                latent_input = pipe.scheduler.scale_model_input(latents, t)
                timesteps = t if torch.is_tensor(t) else torch.tensor([t], device=device, dtype=torch.long)
                if len(timesteps.shape) == 0:
                    timesteps = timesteps.unsqueeze(0)
                timesteps = timesteps.expand(1).to(device)
                noise_pred = pipe.transformer(latent_input, timestep=timesteps, class_labels=class_labels).sample
                if pipe.transformer.config.out_channels // 2 == latent_channels:
                    model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
                else:
                    model_output = noise_pred
                latents = pipe.scheduler.step(model_output, t, latents).prev_sample

            # 提取各层表示 (最终去噪步 t=0): f_i^(ℓ) = (1/L) Σ_j H_i^(ℓ)[j]
            for layer_idx in layer_indices:
                hidden = layer_outputs[layer_idx]  # H_i^(ℓ): [1, L, D]
                if hidden is not None:
                    pooled = hidden.mean(dim=1).cpu().numpy()  # f_i^(ℓ): [1, D]
                    all_reprs.append(pooled[0])
                    all_class_labels.append(class_id)
                    all_layer_labels.append(layer_idx)
    pbar.close()

    for h in handles:
        h.remove()

    return (
        np.array(all_reprs, dtype=np.float32),
        np.array(all_class_labels),
        np.array(all_layer_labels),
    )


def run_tsne_safe(reprs, perplexity, n_iter, seed, max_samples, pca_dims):
    """
    对大规模数据安全运行 t-SNE：下采样 + 可选 PCA + t-SNE
    """
    rng = np.random.default_rng(seed)
    n = len(reprs)
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        reprs = reprs[idx]
        subsampled = True
    else:
        idx = np.arange(n)
        subsampled = False

    if pca_dims > 0 and reprs.shape[1] > pca_dims:
        scaler = StandardScaler()
        reprs_scaled = scaler.fit_transform(reprs)
        pca = PCA(n_components=min(pca_dims, reprs.shape[1], reprs.shape[0] - 1))
        reprs = pca.fit_transform(reprs_scaled)
        print(f"  PCA 降维至 {reprs.shape[1]} 维")

    if subsampled:
        print(f"  下采样至 {len(reprs)} 样本 (原 {n})")

    perplexity = min(perplexity, len(reprs) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=seed, verbose=1)
    emb = tsne.fit_transform(reprs)
    return emb, idx


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--class_ids", type=int, nargs="+", default=[50,100, 207, 281, 412, 963],
                        help="ImageNet 类别 ID 列表")
    parser.add_argument("--samples_per_class", type=int, default=1000)
    parser.add_argument("--layer_indices", type=int, nargs="+",
                        default=[0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
                        help="要提取的 transformer block 索引 (DiT-XL 共 28 层)")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (5-50)")
    parser.add_argument("--n_iter", type=int, default=1000, help="t-SNE 迭代次数")
    parser.add_argument("--max_tsne_samples", type=int, default=15000,
                        help="t-SNE 最大样本数，超出时随机下采样，避免 OpenBLAS 段错误")
    parser.add_argument("--pca_dims", type=int, default=50,
                        help="t-SNE 前先用 PCA 降维到此维度 (0 表示跳过)，可加速并减少内存")
    parser.add_argument("--mode", choices=["all", "per_class"], default="all",
                        help="all: 全量; per_class: 每个类别单独做 t-SNE，按层着色")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载 DiT: {args.model_dir}")
    pipe = load_dit(offline=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    print(f"提取中间层表示: classes={args.class_ids}, samples/class={args.samples_per_class}, layers={args.layer_indices}")
    reprs, labels_class, labels_layer = extract_multi_layer_repr(
        pipe,
        class_ids=args.class_ids,
        samples_per_class=args.samples_per_class,
        layer_indices=args.layer_indices,
        num_inference_steps=args.num_inference_steps,
        device=device,
        seed=args.seed,
    )

    print(f"收集到 {len(reprs)} 个表示向量, shape={reprs.shape}")

    if args.mode == "per_class":
        # 同一类不同层的 t-SNE：每个类别单独做 t-SNE，按层着色
        print("模式: per_class，每个类别单独 t-SNE，按层着色")
        n_classes = len(args.class_ids)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_classes == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        layer_colors = plt.cm.viridis(np.linspace(0, 1, len(args.layer_indices)))
        for idx, class_id in enumerate(args.class_ids):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            mask = labels_class == class_id
            reprs_c = reprs[mask]
            labels_layer_c = labels_layer[mask]

            print(f"  Class {class_id}: {len(reprs_c)} 样本")
            emb_c, sub_idx = run_tsne_safe(
                reprs_c, args.perplexity, args.n_iter, args.seed,
                args.max_tsne_samples, args.pca_dims
            )
            labels_layer_c = labels_layer_c[sub_idx]

            for i, lid in enumerate(args.layer_indices):
                m = labels_layer_c == lid
                ax.scatter(emb_c[m, 0], emb_c[m, 1], c=[layer_colors[i]], label=f"L{lid}", alpha=0.7, s=35)
            ax.set_title(f"Class {class_id}")
            ax.legend(loc="upper right", fontsize=8)
            ax.axis("off")

        for idx in range(n_classes, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()
        out_path = out_dir / "dit_tsne_per_class_layers.png"
    else:
        # 原逻辑：全量 t-SNE（带下采样和 PCA 以规避段错误）
        print("运行 t-SNE...")
        emb, sub_idx = run_tsne_safe(
            reprs, args.perplexity, args.n_iter, args.seed,
            args.max_tsne_samples, args.pca_dims
        )
        labels_class = labels_class[sub_idx]
        labels_layer = labels_layer[sub_idx]

        mid_layer = args.layer_indices[len(args.layer_indices) // 2]
        mask_mid = labels_layer == mid_layer
        emb_mid = emb[mask_mid]
        labels_mid = labels_class[mask_mid]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(args.class_ids)))
        for i, cid in enumerate(args.class_ids):
            m = labels_mid == cid
            ax1.scatter(emb_mid[m, 0], emb_mid[m, 1], c=[colors[i]], label=f"class{cid}", alpha=0.7, s=40)
        ax1.set_title(f"t-SNE by Class (layer {mid_layer})")
        ax1.legend()
        ax1.axis("off")

        ax2 = axes[1]
        layer_colors = plt.cm.viridis(np.linspace(0, 1, len(args.layer_indices)))
        for i, lid in enumerate(args.layer_indices):
            m = labels_layer == lid
            ax2.scatter(emb[m, 0], emb[m, 1], c=[layer_colors[i]], label=f"layer {lid}", alpha=0.6, s=30)
        ax2.set_title("t-SNE by Transformer Layer")
        ax2.legend()
        ax2.axis("off")

        plt.tight_layout()
        out_path = out_dir / "dit_tsne_visualization.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nt-SNE 可视化已保存至: {out_path}")


if __name__ == "__main__":
    main()
