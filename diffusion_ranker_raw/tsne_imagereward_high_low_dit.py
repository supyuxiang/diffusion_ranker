#!/usr/bin/env python3
"""
用 ImageReward 对一批 DiT 图打分，取分数最高的 K 张与最低的 K 张（默认各 500），
在指定 DiT Transformer block 的 GAP 表征上做 t-SNE，二维散点图区分两类。

默认：1000 张图（类 0..999 各 1 张）→ 排序后 top-500 / bottom-500，无重叠。

示例:
  python3 tsne_imagereward_high_low_dit.py \\
    --run_config /path/to/run_config.json \\
    --output_dir /path/to/tsne_ir_high_low

  # 每一层各一张 t-SNE 图（同一次 DiT 前向钩住所有 block，不重复采样）
  python3 tsne_imagereward_high_low_dit.py \
  --run_config outputs/ranker_mse_kl/run_config.json \
  --out_dir outputs/tsne_ir_high_low \
  --top_k 500 --bottom_k 500 \
  --tsne_max_iter 15000 --pca_dim 64

  # 默认会对 IR 高/低子集做逐层 linear probe（分层 LogReg K 折），结果见 linear_probe_by_layer.json
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


def build_pool_scored(
    pipe,
    *,
    pool_size: int,
    deno_steps: int,
    guidance_scale: float,
    hidden_layer_idx: int | None,
    device: str,
    pool_seed: int,
    labels_1k: list[str],
    prompt_template: str,
    teacher,
    ir_device: str,
    all_layers: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    生成 pool_size 张图（class i = i），ImageReward 打分。
    若 all_layers：feats [N, L, D]；否则 feats [N, D]。
    """
    feats_list = []
    scores_list = []
    cids: list[int] = []

    desc = "DiT + ImageReward (all blocks)" if all_layers else "DiT + ImageReward"
    for cid in tqdm(range(pool_size), desc=desc):
        seed = pool_seed + cid * 10007
        with torch.no_grad():
            if all_layers:
                pil_list, pooled_h = tr.rollout_dit_same_class_pil_and_all_layers_pooled_hidden(
                    pipe,
                    1,
                    cid,
                    deno_steps,
                    device,
                    seed,
                    guidance_scale=guidance_scale,
                )
                # [L, 1, D] -> [L, D]
                h = pooled_h[:, 0, :].detach().cpu().float().numpy()
            else:
                assert hidden_layer_idx is not None
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
                h = pooled_h[0].detach().cpu().float().numpy()
        prompt = tr.prompt_for_class(labels_1k, cid, prompt_template)
        with torch.inference_mode():
            sc = float(teacher.score(prompt, pil_list[0]))
        feats_list.append(h)
        scores_list.append(sc)
        cids.append(cid)

    feats = np.stack(feats_list, axis=0).astype(np.float32)
    scores = np.asarray(scores_list, dtype=np.float64)
    return feats, scores, cids


def _tsne_and_save_plot(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fig_path: Path,
    title: str,
    top_k: int,
    bottom_k: int,
    tsne_seed: int,
    perplexity: float,
    pca_dim: int,
    tsne_max_iter: int,
) -> None:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X_fit = X
    if pca_dim and pca_dim > 0:
        pca = PCA(n_components=min(pca_dim, X.shape[0], X.shape[1]), random_state=tsne_seed)
        X_fit = pca.fit_transform(X)
        print(f"  PCA -> {X_fit.shape[1]}D (var sum={pca.explained_variance_ratio_.sum():.4f})")

    perp = min(float(perplexity), float(X_fit.shape[0] - 1))
    perp = max(perp, 2.0)
    tsne = TSNE(
        n_components=2,
        random_state=tsne_seed,
        perplexity=perp,
        max_iter=int(tsne_max_iter),
    )
    Z = tsne.fit_transform(X_fit)
    if getattr(tsne, "n_iter_", None) is not None:
        print(f"  t-SNE 实际迭代: {tsne.n_iter_} (上限 max_iter={tsne_max_iter})")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 7))
    m_high = y == 0
    m_low = y == 1
    plt.scatter(Z[m_high, 0], Z[m_high, 1], s=10, alpha=0.6, c="C0", label=f"IR top-{top_k}")
    plt.scatter(Z[m_low, 0], Z[m_low, 1], s=10, alpha=0.6, c="C3", label=f"IR bottom-{bottom_k}")
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def _linear_probe_binary(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int,
    random_state: int,
) -> dict:
    """
    在冻结特征 X 上做二分类 linear probe：L2 LogisticRegression + 分层 K 折。
    y: 0 = IR 高分组，1 = IR 低分组。
    使用 torchmetrics 计算 accuracy 和 ROC-AUC。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from torchmetrics.functional.classification import binary_accuracy, binary_auroc

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n_samples, n_features = X.shape
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        return {"error": "每个类别至少需要 2 个样本"}

    n_folds_eff = int(min(n_folds, int(counts.min())))
    if n_folds_eff < 2:
        return {"error": f"分层折数不足：min_class_count={counts.min()}，无法做 n_folds={n_folds}"}

    skf = StratifiedKFold(n_splits=n_folds_eff, shuffle=True, random_state=random_state)
    accs: list[float] = []
    aucs: list[float] = []
    clf = LogisticRegression(max_iter=4000, random_state=random_state, solver="lbfgs")

    for train_i, test_i in skf.split(X, y):
        clf.fit(X[train_i], y[train_i])
        pred = clf.predict(X[test_i])
        proba = clf.predict_proba(X[test_i])[:, 1]
        pred_t = torch.from_numpy(pred.astype(np.int64))
        target_t = torch.from_numpy(y[test_i].copy())
        proba_t = torch.from_numpy(proba.astype(np.float64)).float()
        accs.append(float(binary_accuracy(pred_t, target_t).item()))
        aucs.append(float(binary_auroc(proba_t, target_t).item()))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "n_folds": n_folds_eff,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_config",
        type=str,
        default=None,
        help="run_config.json 路径（读取 dit_path、deno_steps、guidance_scale、layer_ratio、fp16、prompt 等）",
    )
    p.add_argument("--dit_path", type=str, default=None)
    p.add_argument(
        "--output_dir",
        "--out_dir",
        type=str,
        default=str(_ROOT / "outputs" / "tsne_ir_high_low"),
        dest="output_dir",
        help="输出目录（与 --out_dir 等价）",
    )
    p.add_argument("--pool_size", type=int, default=1000, help="生成并打分的总张数（默认 1000，类 0..N-1）")
    p.add_argument("--top_k", type=int, default=500, help="ImageReward 最高的 K 张为一类")
    p.add_argument("--bottom_k", type=int, default=500, help="ImageReward 最低的 K 张为一类")
    p.add_argument("--pool_seed", type=int, default=42)
    p.add_argument("--deno_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--layer_ratio", type=float, default=None)
    p.add_argument("--hidden_layer_idx", type=int, default=None)
    p.add_argument(
        "--all_layers",
        action="store_true",
        help="对每一个 DiT block 的 GAP 各画一张 t-SNE（每张图仍用 IR 高/低各 K 张；DiT 只跑一轮多 hook）",
    )
    p.add_argument("--fp16", action="store_true", default=None, help="默认沿用 run_config；仅显式传入时覆盖")
    p.add_argument("--no_fp16", action="store_true", help="强制 float32 DiT")
    p.add_argument("--tsne_seed", type=int, default=42)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument(
        "--tsne_max_iter",
        type=int,
        default=11451,
        help="t-SNE 优化最大迭代（原 1000 易未收敛；可分块调大，如 8000~10000）",
    )
    p.add_argument("--pca_dim", type=int, default=0, help=">0 时先 PCA 再 t-SNE（高维推荐 32~64）")
    p.add_argument("--imagereward_device", type=str, default=None)
    p.add_argument(
        "--no_linear_probe",
        action="store_true",
        help="跳过 IR 高/低二分类 linear probe（默认开启）",
    )
    p.add_argument("--probe_folds", type=int, default=5, help="linear probe 分层 K 折的 K")
    p.add_argument("--probe_seed", type=int, default=None, help="probe 随机种子，默认与 --tsne_seed 相同")
    args = p.parse_args()

    if args.pool_size > 1000:
        raise SystemExit("当前实现为每类 1 张，pool_size 不可超过 1000")
    if args.top_k + args.bottom_k > args.pool_size:
        raise SystemExit("top_k + bottom_k 不能大于 pool_size（否则无法保证两类不重叠）")
    if args.top_k < 1 or args.bottom_k < 1:
        raise SystemExit("top_k / bottom_k 须 >= 1")

    cfg: dict = {}
    if args.run_config:
        rc = Path(args.run_config)
        if not rc.is_file():
            raise SystemExit(f"找不到 run_config: {rc}")
        with open(rc, encoding="utf-8") as f:
            cfg = json.load(f)

    dit_path = args.dit_path or cfg.get("dit_path")
    if not dit_path:
        raise SystemExit("需要 --dit_path 或 --run_config 中含 dit_path")

    deno_steps = int(args.deno_steps if args.deno_steps is not None else cfg.get("deno_steps", 25))
    guidance_scale = float(
        args.guidance_scale if args.guidance_scale is not None else cfg.get("guidance_scale", 1.0)
    )
    lratio = args.layer_ratio if args.layer_ratio is not None else cfg.get("layer_ratio", 0.6)
    hid_idx_arg = args.hidden_layer_idx if args.hidden_layer_idx is not None else cfg.get("hidden_layer_idx")

    if args.no_fp16:
        fp16 = False
    elif args.fp16 is True:
        fp16 = True
    else:
        fp16 = bool(cfg.get("fp16", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ir_device = args.imagereward_device or cfg.get("imagereward_device") or str(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_dit(dit_path, str(device), fp16=fp16)
    n_blocks = len(pipe.transformer.transformer_blocks)
    if args.all_layers:
        hidden_layer_idx = None
    else:
        hidden_layer_idx = tr.resolve_hidden_layer_idx(n_blocks, lratio, hid_idx_arg)

    labels_1k = tr.load_imagenet_simple_labels()
    template = str(cfg.get("prompt_template", "a high-quality picture of a {name}"))

    tr._ensure_imagereward_path()
    import ImageReward as RM

    teacher = RM.load("ImageReward-v1.0", device=ir_device)
    teacher.eval()

    if args.all_layers:
        print(
            f"DiT all {n_blocks} blocks (GAP), deno_steps={deno_steps}, guidance={guidance_scale}, "
            f"pool={args.pool_size}, top_k={args.top_k}, bottom_k={args.bottom_k}"
        )
    else:
        print(
            f"DiT block={hidden_layer_idx}/{n_blocks}, deno_steps={deno_steps}, guidance={guidance_scale}, "
            f"pool={args.pool_size}, top_k={args.top_k}, bottom_k={args.bottom_k}"
        )

    feats, scores, cids = build_pool_scored(
        pipe,
        pool_size=args.pool_size,
        deno_steps=deno_steps,
        guidance_scale=guidance_scale,
        hidden_layer_idx=hidden_layer_idx,
        device=str(device),
        pool_seed=args.pool_seed,
        labels_1k=labels_1k,
        prompt_template=template,
        teacher=teacher,
        ir_device=ir_device,
        all_layers=args.all_layers,
    )

    order = np.argsort(-scores)
    high_idx = order[: args.top_k]
    low_idx = order[-args.bottom_k :]

    y = np.array([0] * args.top_k + [1] * args.bottom_k, dtype=np.int64)

    meta: dict = {
        "pool_size": args.pool_size,
        "top_k": args.top_k,
        "bottom_k": args.bottom_k,
        "all_layers": bool(args.all_layers),
        "hidden_layer_idx": hidden_layer_idx,
        "n_blocks": n_blocks,
        "deno_steps": deno_steps,
        "guidance_scale": guidance_scale,
        "pool_seed": args.pool_seed,
        "high_global_indices": high_idx.tolist(),
        "low_global_indices": low_idx.tolist(),
        "high_class_ids": [cids[i] for i in high_idx],
        "low_class_ids": [cids[i] for i in low_idx],
        "scores": scores.tolist(),
        "perplexity": args.perplexity,
        "tsne_max_iter": args.tsne_max_iter,
        "tsne_seed": args.tsne_seed,
        "pca_dim": args.pca_dim,
    }
    if args.all_layers:
        meta["feat_shape"] = list(feats.shape)
        np.savez_compressed(out_dir / "dit_gap_features_all_layers.npz", feats=feats, scores=scores)
    else:
        np.savez_compressed(out_dir / "dit_gap_features.npz", feats=feats, scores=scores)

    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise SystemExit("请安装 scikit-learn: pip install scikit-learn") from e

    probe_seed = args.probe_seed if args.probe_seed is not None else args.tsne_seed
    if not args.no_linear_probe:
        probe_layers: list[dict] = []
        if args.all_layers:
            for li in range(n_blocks):
                X_h = feats[high_idx, li]
                X_l = feats[low_idx, li]
                Xp = np.concatenate([X_h, X_l], axis=0)
                row = {"layer_idx": li, **_linear_probe_binary(Xp, y, n_folds=args.probe_folds, random_state=probe_seed)}
                probe_layers.append(row)
                if "error" not in row:
                    print(
                        f"linear probe block {li:02d}/{n_blocks}: "
                        f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}, "
                        f"roc_auc={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}"
                    )
                else:
                    print(f"linear probe block {li:02d}: {row['error']}")
        else:
            assert hidden_layer_idx is not None
            Xp = np.concatenate([feats[high_idx], feats[low_idx]], axis=0)
            row = {
                "layer_idx": hidden_layer_idx,
                **_linear_probe_binary(Xp, y, n_folds=args.probe_folds, random_state=probe_seed),
            }
            probe_layers.append(row)
            if "error" not in row:
                print(
                    f"linear probe block {hidden_layer_idx}: "
                    f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}, "
                    f"roc_auc={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}"
                )
            else:
                print(f"linear probe: {row['error']}")

        probe_doc = {
            "task": "binary: ImageReward top-K (label=0) vs bottom-K (label=1) on frozen DiT GAP",
            "classifier": "sklearn.linear_model.LogisticRegression(L2, lbfgs, max_iter=4000)",
            "requested_folds": args.probe_folds,
            "random_state": probe_seed,
            "layers": probe_layers,
        }
        lp_path = out_dir / "linear_probe_by_layer.json"
        with open(lp_path, "w", encoding="utf-8") as f:
            json.dump(probe_doc, f, indent=2, ensure_ascii=False)
        meta["linear_probe_json"] = lp_path.name
        print(f"linear probe 结果: {lp_path}")

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        plt = None

    if args.all_layers:
        if feats.ndim != 3 or feats.shape[1] != n_blocks:
            raise RuntimeError(f"预期 feats [N,L,D]，得到 {feats.shape}，n_blocks={n_blocks}")
        png_names: list[str] = []
        for li in range(n_blocks):
            X_high = feats[high_idx, li]
            X_low = feats[low_idx, li]
            X = np.concatenate([X_high, X_low], axis=0)
            fig_path = out_dir / f"tsne_imagereward_high_low_layer{li:02d}.png"
            title = (
                f"t-SNE of DiT GAP (block {li} / {n_blocks})\n"
                f"high vs low ImageReward ({args.top_k}+{args.bottom_k} / {args.pool_size} pool)"
            )
            if plt is None:
                raise SystemExit("逐层 t-SNE 需要 matplotlib: pip install matplotlib")
            print(f"t-SNE block {li}/{n_blocks} -> {fig_path.name}")
            _tsne_and_save_plot(
                X,
                y,
                fig_path=fig_path,
                title=title,
                top_k=args.top_k,
                bottom_k=args.bottom_k,
                tsne_seed=args.tsne_seed,
                perplexity=args.perplexity,
                pca_dim=args.pca_dim,
                tsne_max_iter=args.tsne_max_iter,
            )
            png_names.append(fig_path.name)
        meta["tsne_png_basenames"] = png_names
        print(f"已写入 {len(png_names)} 张图到 {out_dir}")
    else:
        X_high = feats[high_idx]
        X_low = feats[low_idx]
        X = np.concatenate([X_high, X_low], axis=0)
        fig_path = out_dir / "tsne_imagereward_high_low.png"
        title = (
            f"t-SNE of DiT GAP (block {hidden_layer_idx})\n"
            f"high vs low ImageReward ({args.top_k}+{args.bottom_k} / {args.pool_size} pool)"
        )
        if plt is None:
            from sklearn.manifold import TSNE

            X_fit = X
            if args.pca_dim and args.pca_dim > 0:
                from sklearn.decomposition import PCA

                pca = PCA(
                    n_components=min(args.pca_dim, X.shape[0], X.shape[1]),
                    random_state=args.tsne_seed,
                )
                X_fit = pca.fit_transform(X)
            perp = min(float(args.perplexity), float(X_fit.shape[0] - 1))
            perp = max(perp, 2.0)
            Z = TSNE(
                n_components=2,
                random_state=args.tsne_seed,
                perplexity=perp,
                max_iter=int(args.tsne_max_iter),
            ).fit_transform(X_fit)
            np.savez_compressed(out_dir / "tsne_coords.npz", Z=Z, y=y)
            print("未安装 matplotlib，已写入 tsne_coords.npz")
        else:
            _tsne_and_save_plot(
                X,
                y,
                fig_path=fig_path,
                title=title,
                top_k=args.top_k,
                bottom_k=args.bottom_k,
                tsne_seed=args.tsne_seed,
                perplexity=args.perplexity,
                pca_dim=args.pca_dim,
                tsne_max_iter=args.tsne_max_iter,
            )
            print(f"图像已保存: {fig_path}")
            meta["tsne_png_basenames"] = [fig_path.name]
        if plt is None:
            meta["tsne_png_basenames"] = []

    with open(out_dir / "tsne_source_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"元数据: {out_dir / 'tsne_source_meta.json'}")


if __name__ == "__main__":
    main()
