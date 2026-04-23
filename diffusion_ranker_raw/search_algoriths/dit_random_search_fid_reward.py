#!/usr/bin/env python3
"""
DiT Random Search with FID as Verifier (Inference-Time Scaling)

论文方法：增量构造 + FID 作为 verifier 信号（完全对齐 Appendix A.4）
- 预计算真实集 (ImageNet) 的 μ_ref, Σ_ref
- 固定 deno_steps=20，Random Search 对随机种子（不同噪声/类别）
- 初始化：前 init_random_samples 张图随机生成，用于初始化 μ̂, Σ̂
- 迭代：每轮生成 K 批候选（不同 seed），选使 FID 降低最多的一批加入已选集

用法:
  CUDA_VISIBLE_DEVICES=6 python3 dit_random_search_fid_reward.py \
    --dit_path /data/yxfeng/models/dit_model \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --output_dir /data/yxfeng/eval_dit_random_search_fid \
    --num_samples 50000 \
    --batch_size 24 \
    --num_candidates 8 \
    --init_random_samples 1024 \
    --deno_steps 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DiTPipeline
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dit_random_search_common import (
    CANDIDATE_SEED_HIGH,
    INIT_BATCH_SEED_STRIDE,
    load_dit,
    rollout_dit_batch_fid_images,
)
from utils import compute_fid_and_is, setup_seed


def fid_features_to_statistics(features: np.ndarray) -> dict:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {"mu": mu, "sigma": sigma}


def merge_fid_statistics(stats_a: dict, n_a: int, stats_b: dict, n_b: int) -> dict:
    if n_a == 0:
        return stats_b
    if n_b == 0:
        return stats_a

    mu_a, sigma_a = stats_a["mu"], stats_a["sigma"]
    mu_b, sigma_b = stats_b["mu"], stats_b["sigma"]
    n = n_a + n_b

    mu_merged = (n_a * mu_a + n_b * mu_b) / n
    sum_xx_a = ((n_a - 1) * sigma_a if n_a > 1 else 0) + n_a * np.outer(mu_a, mu_a)
    sum_xx_b = ((n_b - 1) * sigma_b if n_b > 1 else 0) + n_b * np.outer(mu_b, mu_b)
    sum_xx = sum_xx_a + sum_xx_b
    sigma_merged = (sum_xx - n * np.outer(mu_merged, mu_merged)) / (n - 1) if n > 1 else np.zeros_like(sigma_a)

    return {"mu": mu_merged, "sigma": sigma_merged}


def fid_statistics_to_distance(stat_gen: dict, stat_ref: dict) -> float:
    mu1, sigma1 = stat_gen["mu"], stat_gen["sigma"]
    mu2, sigma2 = stat_ref["mu"], stat_ref["sigma"]
    diff = mu1 - mu2
    evals = np.linalg.eigvals(sigma1 @ sigma2)
    evals = np.maximum(evals.real, 0.0)
    tr_covmean = np.sum(np.sqrt(evals.astype("float64") + 1e-14))
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fid)


def get_inception_feature_extractor(device: str):
    from torch_fidelity.utils import create_feature_extractor

    return create_feature_extractor(
        "inception-v3-compat",
        ["2048"],
        cuda=(device == "cuda"),
        verbose=False,
    )


def extract_features_from_images(
    images: list[Image.Image],
    feat_extractor,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    from torch_fidelity.datasets import TransformPILtoRGBTensor

    transform = Compose(
        [
            Resize(256),
            CenterCrop(256),
            TransformPILtoRGBTensor(),
        ]
    )

    feats_list = []
    feat_extractor.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            tensors = torch.stack([transform(img) for img in batch_imgs])
            if device == "cuda":
                tensors = tensors.cuda(non_blocking=True)
            out = feat_extractor(tensors)
            if isinstance(out, tuple):
                out = out[0]
            feats_list.append(out.cpu().numpy())
    return np.concatenate(feats_list, axis=0)


def compute_ref_statistics(real_dir: Path, device: str, cache_name: str = "imagenet_val") -> dict:
    from torch_fidelity.metric_fid import fid_input_id_to_statistics_cached
    from torch_fidelity.utils import create_feature_extractor

    kwargs = dict(
        input2=str(real_dir),
        input2_cache_name=cache_name,
        samples_find_deep=True,
        samples_find_ext="png,jpg,jpeg",
        samples_resize_and_crop=256,
        cuda=(device == "cuda"),
        batch_size=64,
        verbose=True,
        cache=True,
    )
    feat_extractor = create_feature_extractor("inception-v3-compat", ["2048"], **kwargs)
    feat_layer = "2048"
    return fid_input_id_to_statistics_cached(2, feat_extractor, feat_layer, **kwargs)


def init_random_samples_fid(
    pipe: DiTPipeline,
    feat_extractor,
    output_dir: Path,
    init_samples: int,
    batch_size: int,
    deno_steps: int,
    device: str,
    seed: int,
    guidance_scale: float = 1.0,
) -> tuple[dict, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images: list[Image.Image] = []
    img_counter = 0

    num_batches = (init_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="随机初始化"):
        n_this = min(batch_size, init_samples - len(all_images))
        if n_this <= 0:
            break
        batch_images = rollout_dit_batch_fid_images(
            pipe,
            deno_steps,
            n_this,
            device,
            seed=seed + batch_idx * INIT_BATCH_SEED_STRIDE,
            guidance_scale=guidance_scale,
        )
        for img in batch_images:
            if img_counter >= init_samples:
                break
            img.save(output_dir / f"{img_counter:06d}.png")
            img_counter += 1
            all_images.append(img)

    init_feats = extract_features_from_images(all_images, feat_extractor, device, batch_size=64)
    init_stats = fid_features_to_statistics(init_feats)
    return init_stats, img_counter


def run_random_search_fid(
    pipe: DiTPipeline,
    feat_extractor,
    ref_stats: dict,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    deno_steps: int,
    device: str,
    seed: int,
    guidance_scale: float = 1.0,
    init_stats: dict | None = None,
    n_init: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_stats = init_stats
    n_selected = n_init
    rng = np.random.default_rng(seed)
    img_counter = n_init

    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size
    for _ in tqdm(range(num_rounds), desc="Random Search FID"):
        n_this_batch = min(batch_size, num_samples - img_counter)
        if n_this_batch <= 0:
            break

        candidate_seeds = rng.integers(0, CANDIDATE_SEED_HIGH, size=num_candidates).tolist()

        best_fid = float("inf")
        best_batch = None
        best_batch_stats = None

        for cand_seed in candidate_seeds:
            batch_images = rollout_dit_batch_fid_images(
                pipe,
                deno_steps,
                n_this_batch,
                device,
                seed=cand_seed,
                guidance_scale=guidance_scale,
            )
            batch_feats = extract_features_from_images(batch_images, feat_extractor, device, batch_size=64)
            batch_stats = fid_features_to_statistics(batch_feats)

            if selected_stats is None:
                merged_stats = batch_stats
                n_merged = n_this_batch
            else:
                merged_stats = merge_fid_statistics(selected_stats, n_selected, batch_stats, n_this_batch)
                n_merged = n_selected + n_this_batch

            fid = fid_statistics_to_distance(merged_stats, ref_stats)
            if fid < best_fid:
                best_fid = fid
                best_batch = batch_images
                best_batch_stats = batch_stats

        for img in best_batch:
            if img_counter >= num_samples:
                break
            img.save(output_dir / f"{img_counter:06d}.png")
            img_counter += 1

        if selected_stats is None:
            selected_stats = best_batch_stats
        else:
            selected_stats = merge_fid_statistics(
                selected_stats, n_selected, best_batch_stats, len(best_batch)
            )
        n_selected += len(best_batch)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dit_path", default="/data/home/yxfeng/data/models/dit_model")
    p.add_argument("--imagenet_val", default="/data/yxfeng/datasets/imagenet/val")
    p.add_argument("--output_dir", default="/data/yxfeng/eval_dit_random_search_fid")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_candidates", type=int, default=8, help="每轮随机采样的候选 batch 数（不同 seed）")
    p.add_argument("--deno_steps", type=int, default=20, help="固定 denoising step 数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DiT 半精度（默认开）；可用 --no-fp16",
    )
    p.add_argument("--ref_cache_name", default="imagenet_val")
    p.add_argument(
        "--init_random_samples",
        type=int,
        default=1024,
        help="论文：前 N 张随机选择以初始化 μ̂, Σ̂，默认 1024",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert args.deno_steps >= 1, "deno_steps 需 >= 1"
    assert 0 <= args.init_random_samples <= args.num_samples, "init_random_samples 需在 [0, num_samples] 内"

    output_dir = Path(args.output_dir)
    imagenet_val = Path(args.imagenet_val)
    assert imagenet_val.exists(), f"ImageNet val 未找到: {imagenet_val}"

    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 预计算真实集 FID 统计量...")
    ref_stats = compute_ref_statistics(imagenet_val, device, args.ref_cache_name)
    print(f"   μ_ref shape: {ref_stats['mu'].shape}, Σ_ref shape: {ref_stats['sigma'].shape}")

    print("2. 加载 DiT...")
    pipe = load_dit(Path(args.dit_path), device, fp16=args.fp16)

    print("3. 创建 Inception 特征提取器...")
    feat_extractor = get_inception_feature_extractor(device)

    init_stats = None
    n_init = 0
    if args.init_random_samples > 0:
        print("4a. 随机初始化（论文：前 1024 张）...")
        init_stats, n_init = init_random_samples_fid(
            pipe,
            feat_extractor,
            output_dir,
            args.init_random_samples,
            args.batch_size,
            args.deno_steps,
            device,
            args.seed,
            guidance_scale=args.guidance_scale,
        )
        print(f"   已初始化 {n_init} 张，μ̂ shape={init_stats['mu'].shape}")

    print("4b. 运行 Random Search (FID verifier)...")
    print(f"   deno_steps={args.deno_steps}, num_candidates={args.num_candidates}")
    run_random_search_fid(
        pipe,
        feat_extractor,
        ref_stats,
        output_dir,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        args.deno_steps,
        device,
        args.seed,
        guidance_scale=args.guidance_scale,
        init_stats=init_stats,
        n_init=n_init,
    )

    log_path = output_dir / "config.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "init_random_samples": args.init_random_samples,
                "deno_steps": args.deno_steps,
                "num_candidates": args.num_candidates,
                "fp16": args.fp16,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n完成。图像保存至 {output_dir}，配置: {log_path}")

    print("\n5. 计算最终 FID / IS...")
    fid_val, is_val = compute_fid_and_is(
        output_dir, imagenet_val, device, real_cache_name=args.ref_cache_name
    )
    print(f"   最终 FID: {fid_val:.2f}, IS: {is_val:.4f}")
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"FID": fid_val, "IS": is_val}, f, indent=2)


if __name__ == "__main__":
    main()
