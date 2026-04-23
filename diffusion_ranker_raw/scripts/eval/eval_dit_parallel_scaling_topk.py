#!/usr/bin/env python3
"""
DiT parallel scaling 实验：每个 rollout-level denoising step 独立重复 num_runs 次，
每次生成 num_samples 张图并算 FID / IS；对 5 次结果做 top-k 聚合（k=1..5）。

FID 越低越好：top-k 取 k 个最小 FID 的均值。
IS / CLIPScore 越高越好：top-k 取 k 个最大值的均值。
CLIPScore 为本地 Hugging Face 格式 openai/clip-vit-large-patch14 的图文嵌入余弦相似度均值（默认从 /data/yxfeng/models/clip/ 下缓存读取；DiT 为类条件生成，用通用英文 prompt）。

示例（完整 50k × 5 × 10 steps，耗时很长）：

cd /data/home/yxfeng/project1/idm && CUDA_VISIBLE_DEVICES=5 python3 scripts/eval/eval_dit_parallel_scaling_topk.py \
  --dit_path /data/yxfeng/models/dit_model \
  --imagenet_val /data/yxfeng/datasets/imagenet/val \
  --output_dir /data/yxfeng/eval_dit_parallel_scaling_topk \
  --num_samples 50000 --num_runs 5 \
  --deno_steps 10 20 30 40 50 60 70 80 90 100 \
  --batch_size 24 --compile --fp16 --kid --prc \
  --fid_real_max_samples 0

  # --fid_real_max_samples 50000 --fid_real_subset_seed 0
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List, Tuple, Union, Iterable
import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from PIL import Image
from tqdm import tqdm

from utils import compute_metrics, setup_seed

_rollout_path = Path(__file__).resolve().parent / "eval_dit_scaling_laws_rollout_level.py"
_spec = importlib.util.spec_from_file_location("dit_rollout_eval", _rollout_path)
_rollout = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_rollout)
load_dit = _rollout.load_dit
rollout_dit_with_steps = _rollout.rollout_dit_with_steps


def topk_fid_means(fids: list[float], k_max: int = 5) -> dict[str, float]:
    """k 个最低 FID 的均值；若有效运行数 < k 则为 nan。"""
    s = sorted(float(x) for x in fids if x == x)
    out: dict[str, float] = {}
    for k in range(1, k_max + 1):
        if not s or k > len(s):
            out[str(k)] = float("nan")
        else:
            out[str(k)] = sum(s[:k]) / k
    return out


def resolve_clip_model_dir(clip_model_path: str | None) -> Path:
    """
    解析可传给 transformers CLIPModel.from_pretrained 的目录（须含 config.json）。
    若传入仓库根目录（含 snapshots/），则取 snapshots 下第一个 revision。
    未传入时默认：/data/yxfeng/models/clip/models--openai--clip-vit-large-patch14
    """
    if clip_model_path:
        p = Path(clip_model_path).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"CLIP 路径不存在或非目录: {p}")
        if (p / "config.json").is_file():
            return p
        snaps = sorted((p / "snapshots").glob("*"))
        if snaps and (snaps[0] / "config.json").is_file():
            return snaps[0]
        raise FileNotFoundError(
            f"无法在 {p} 找到含 config.json 的快照（可传 .../snapshots/<revision_hash>）"
        )

    default_repo = Path("/data/yxfeng/models/clip/models--openai--clip-vit-large-patch14")
    snaps = sorted((default_repo / "snapshots").glob("*"))
    if not snaps or not (snaps[0] / "config.json").is_file():
        raise FileNotFoundError(
            f"默认 CLIP 未找到，请用 --clip_model_path 指定本地目录。已查: {default_repo}/snapshots"
        )
    return snaps[0]


def topk_is_means(iss: list[float], k_max: int = 5) -> dict[str, float]:
    """k 个最高 IS 的均值；若有效运行数 < k 则为 nan。"""
    s = sorted((float(x) for x in iss if x == x), reverse=True)
    out: dict[str, float] = {}
    for k in range(1, k_max + 1):
        if not s or k > len(s):
            out[str(k)] = float("nan")
        else:
            out[str(k)] = sum(s[:k]) / k
    return out


def mean_clip_score_from_dir(
    gen_dir: Path,
    clip_metric,
    device: str,
    prompt: str,
    batch_size: int,
) -> float:
    """对目录内全部 PNG 用 torchmetrics CLIPScore 计算均值（0-100 scale）。"""
    from torchvision.transforms.functional import pil_to_tensor

    clip_metric.reset()
    paths = sorted(gen_dir.glob("*.png"))
    if not paths:
        return float("nan")

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        tensors = []
        for p in batch_paths:
            with Image.open(p) as im:
                tensors.append(pil_to_tensor(im.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        texts = [prompt] * len(tensors)
        clip_metric.update(batch, texts)

    return float(clip_metric.compute().item())


def save_plots(payload: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    per_step = payload["per_step"]
    steps = [row["steps"] for row in per_step]
    k_plot_max = max(
        int(k)
        for row in per_step
        for k in row["aggregate"]["topk_fid_mean"].keys()
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    cmap = plt.cm.viridis

    for ki in range(1, k_plot_max + 1):
        color = cmap((ki - 1) / max(k_plot_max - 1, 1))
        fid_series = [row["aggregate"]["topk_fid_mean"][str(ki)] for row in per_step]
        is_series = [row["aggregate"]["topk_is_mean"][str(ki)] for row in per_step]
        clip_series = [row["aggregate"]["topk_clip_mean"][str(ki)] for row in per_step]
        axes[0].plot(steps, fid_series, "o-", color=color, label=f"top-{ki} mean (FID)", linewidth=1.8, markersize=3)
        axes[1].plot(steps, is_series, "s-", color=color, label=f"top-{ki} mean (IS)", linewidth=1.8, markersize=3)
        axes[2].plot(steps, clip_series, "^-", color=color, label=f"top-{ki} mean (CLIP)", linewidth=1.8, markersize=3)

    axes[0].set_ylabel("FID (lower better)")
    axes[0].set_title("Parallel scaling: mean of k best FIDs across runs")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8, ncol=2)

    axes[1].set_ylabel("IS (higher better)")
    axes[1].set_title("Parallel scaling: mean of k best IS across runs")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8, ncol=2)

    axes[2].set_ylabel("CLIPScore (higher better)")
    axes[2].set_xlabel("Denoising steps (rollout-level)")
    axes[2].set_title("Parallel scaling: mean of k best CLIPScore across runs")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8, ncol=2)

    plt.tight_layout()
    p = output_dir / "parallel_scaling_topk_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"曲线已保存: {p}")

    # 另存：5 次运行的 mean ± std（非 top-k）
    fig2, ax = plt.subplots(1, 3, figsize=(14, 4))
    fid_m = [row["aggregate"]["fid_mean"] for row in per_step]
    fid_s = [row["aggregate"]["fid_std"] for row in per_step]
    is_m = [row["aggregate"]["is_mean"] for row in per_step]
    is_s = [row["aggregate"]["is_std"] for row in per_step]
    clip_m = [row["aggregate"]["clip_mean"] for row in per_step]
    clip_s = [row["aggregate"]["clip_std"] for row in per_step]
    ax[0].errorbar(steps, fid_m, yerr=fid_s, fmt="o-", capsize=3, color="C0")
    ax[0].set_xlabel("Denoising steps")
    ax[0].set_ylabel("FID")
    ax[0].set_title("Per-step: mean ± std over runs")
    ax[0].grid(True, alpha=0.3)
    ax[1].errorbar(steps, is_m, yerr=is_s, fmt="s-", capsize=3, color="C2")
    ax[1].set_xlabel("Denoising steps")
    ax[1].set_ylabel("IS")
    ax[1].set_title("Per-step: mean ± std over runs")
    ax[1].grid(True, alpha=0.3)
    ax[2].errorbar(steps, clip_m, yerr=clip_s, fmt="^-", capsize=3, color="C1")
    ax[2].set_xlabel("Denoising steps")
    ax[2].set_ylabel("CLIPScore")
    ax[2].set_title("Per-step: mean ± std over runs")
    ax[2].grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = output_dir / "parallel_scaling_mean_std.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"曲线已保存: {p2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_path", default="/data/yxfeng/models/dit_model")
    parser.add_argument("--output_dir", default="/data/yxfeng/eval_dit_parallel_scaling_topk")
    parser.add_argument("--imagenet_val", default="/data/yxfeng/datasets/imagenet/val")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--seed_stride",
        type=int,
        default=1_000_003,
        help="第 i 次运行的 generator 种子偏移 base_seed + i * seed_stride（避免与 batch 内 seed+batch_idx 撞车）",
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--kid", action="store_true")
    parser.add_argument("--prc", action="store_true")
    parser.add_argument(
        "--deno_steps",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )
    parser.add_argument(
        "--fid_real_max_samples",
        type=int,
        default=50000,
        help="FID（及 KID/PRC）参考集从 ImageNet val 中随机子采样的张数；与生成 50k 对齐时常用 50000。设为 0 表示使用 val 目录下全部图片。",
    )
    parser.add_argument(
        "--fid_real_subset_seed",
        type=int,
        default=0,
        help="参考集子采样的随机种子（可复现）。",
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        default=None,
        help=(
            "本地 Hugging Face 格式 CLIP 目录（含 config.json），或 models--... 仓库根目录（自动选 snapshots/*）。"
            "默认: /data/yxfeng/models/clip/models--openai--clip-vit-large-patch14"
        ),
    )
    parser.add_argument(
        "--clip_prompt",
        type=str,
        default="a high-quality photo",
        help="CLIPScore 使用的英文 prompt（DiT 为类条件生成，与 step-level 评测一致用通用句）。",
    )
    parser.add_argument(
        "--clip_batch_size",
        type=int,
        default=64,
        help="计算 CLIPScore 时的图像 batch 大小。",
    )
    args = parser.parse_args()

    deno_steps = sorted({s for s in args.deno_steps if 1 <= s <= 1000})
    assert args.num_runs >= 1
    fid_real_max = args.fid_real_max_samples if args.fid_real_max_samples > 0 else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    imagenet_val = Path(args.imagenet_val)
    assert imagenet_val.exists(), f"ImageNet val 未找到: {imagenet_val}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_dir = resolve_clip_model_dir(args.clip_model_path)
    from torchmetrics.multimodal.clip_score import CLIPScore

    print(f"加载 CLIP (torchmetrics): {clip_dir}")
    clip_metric = CLIPScore(model_name_or_path=str(clip_dir)).to(device)

    print(f"加载 DiT: {args.dit_path}")
    pipe = load_dit(args.dit_path, device, fp16=args.fp16)
    if args.compile and device == "cuda":
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        print("已启用 torch.compile (transformer)")

    run_seeds = [args.base_seed + r * args.seed_stride for r in range(args.num_runs)]

    payload = {
        "meta": {
            "deno_steps": deno_steps,
            "num_runs": args.num_runs,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "base_seed": args.base_seed,
            "seed_stride": args.seed_stride,
            "run_seeds": run_seeds,
            "dit_path": str(args.dit_path),
            "imagenet_val": str(imagenet_val),
            "kid": args.kid,
            "prc": args.prc,
            "fid_real_max_samples": fid_real_max,
            "fid_real_subset_seed": args.fid_real_subset_seed,
            "clip_model_path": str(clip_dir),
            "clip_prompt": args.clip_prompt,
            "clip_batch_size": args.clip_batch_size,
        },
        "per_step": [],
    }

    for steps in tqdm(deno_steps, desc="deno_steps"):
        runs_out = []
        fids: list[float] = []
        iss: list[float] = []
        clips: list[float] = []

        for run_idx in range(args.num_runs):
            seed = run_seeds[run_idx]
            setup_seed(seed)
            gen_dir = output_dir / f"gen_steps_{steps}_run{run_idx}"
            n = rollout_dit_with_steps(
                pipe,
                gen_dir,
                steps,
                args.num_samples,
                args.batch_size,
                device,
                seed,
                guidance_scale=args.guidance_scale,
            )
            metrics = compute_metrics(
                gen_dir,
                imagenet_val,
                device,
                real_cache_name="imagenet_val",
                fid=True,
                isc=True,
                kid=args.kid,
                prc=args.prc,
                real_max_samples=fid_real_max,
                real_subset_seed=args.fid_real_subset_seed,
            )
            clip_val = mean_clip_score_from_dir(
                gen_dir,
                clip_metric,
                device,
                prompt=args.clip_prompt,
                batch_size=args.clip_batch_size,
            )
            metrics["CLIPScore"] = clip_val
            row = {"run_id": run_idx, "seed": seed, "num_generated": n, **metrics}
            runs_out.append(row)
            fids.append(float(metrics.get("FID", float("nan"))))
            iss.append(float(metrics.get("IS", float("nan"))))
            clips.append(float(clip_val))
            shutil.rmtree(gen_dir, ignore_errors=True)

            fid_s = metrics.get("FID", float("nan"))
            is_s = metrics.get("IS", float("nan"))
            print(
                f"  steps={steps} run={run_idx} FID={fid_s:.2f} IS={is_s:.4f} CLIPScore={clip_val:.4f}"
            )

        import statistics

        valid_f = [x for x in fids if x == x]
        valid_i = [x for x in iss if x == x]
        valid_c = [x for x in clips if x == x]
        fid_mean = statistics.mean(valid_f) if valid_f else float("nan")
        fid_std = statistics.stdev(valid_f) if len(valid_f) > 1 else 0.0
        is_mean = statistics.mean(valid_i) if valid_i else float("nan")
        is_std = statistics.stdev(valid_i) if len(valid_i) > 1 else 0.0
        clip_mean = statistics.mean(valid_c) if valid_c else float("nan")
        clip_std = statistics.stdev(valid_c) if len(valid_c) > 1 else 0.0

        topk_fid = topk_fid_means(fids, k_max=5)
        topk_is = topk_is_means(iss, k_max=5)
        topk_clip = topk_is_means(clips, k_max=5)

        step_row = {
            "steps": steps,
            "runs": runs_out,
            "aggregate": {
                "fid_mean": fid_mean,
                "fid_std": fid_std,
                "is_mean": is_mean,
                "is_std": is_std,
                "clip_mean": clip_mean,
                "clip_std": clip_std,
                "topk_fid_mean": topk_fid,
                "topk_is_mean": topk_is,
                "topk_clip_mean": topk_clip,
            },
        }
        payload["per_step"].append(step_row)

        with open(output_dir / "metrics_results_partial.json", "w") as f:
            json.dump(payload, f, indent=2)

    out_path = output_dir / "metrics_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"完整结果: {out_path}")

    save_plots(payload, output_dir)


if __name__ == "__main__":
    main()
