#!/usr/bin/env python3
"""
DiT Random Search with ImageReward as Verifier (Inference-Time Scaling)

与 dit_random_search_fid_reward.py 结构相同，但每轮从 K 个候选 batch 中选
「batch 平均 ImageReward 最高」的一批加入已选集（每张图使用对应 ImageNet 类名的文本条件）。

结束后仍对 output_dir 与 ImageNet val 计算 FID / IS（与 FID 版一致）。

用法:
  CUDA_VISIBLE_DEVICES=6 python3 dit_random_search_imagereward.py \
    --dit_path /data/yxfeng/models/dit_model \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --output_dir /data/yxfeng/eval_dit_random_search_ir \
    --num_samples 50000 \
    --batch_size 32 \
    --num_candidates 8 \
    --init_random_samples 32 \
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
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import ensure_image_reward_on_path

ensure_image_reward_on_path()
import ImageReward as RM

from dit_random_search_common import (
    CANDIDATE_SEED_HIGH,
    init_random_samples,
    load_imagenet_simple_labels,
    model_loader,
    prompt_for_class,
    rollout_dit_batch,
)
from utils import compute_fid_and_is, setup_seed


def mean_batch_imagereward(
    reward_model,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
) -> float:
    scores = []
    with torch.no_grad():
        for img, cid in zip(images, class_ids):
            text = prompt_for_class(labels_1k, cid, prompt_template)
            scores.append(float(reward_model.score(text, img)))
    return float(np.mean(scores)) if scores else float("-inf")


def run_random_search_imagereward(
    pipe: DiTPipeline,
    reward_model,
    labels_1k: list[str],
    prompt_template: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    deno_steps: int,
    device: str,
    seed: int,
    guidance_scale: float = 1.0,
    n_init: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for _ in tqdm(range(num_rounds), desc="Random Search ImageReward"):
        n_this_batch = min(batch_size, num_samples - img_counter)
        if n_this_batch <= 0:
            break

        candidate_seeds = rng.integers(0, CANDIDATE_SEED_HIGH, size=num_candidates).tolist()
        best_score = float("-inf")
        best_batch: list[Image.Image] | None = None

        for cand_seed in candidate_seeds:
            batch_images, batch_classes = rollout_dit_batch(
                pipe,
                deno_steps,
                n_this_batch,
                device,
                seed=int(cand_seed),
                guidance_scale=guidance_scale,
            )
            score = mean_batch_imagereward(
                reward_model, batch_images, batch_classes, labels_1k, prompt_template
            )
            if score > best_score:
                best_score = score
                best_batch = batch_images

        assert best_batch is not None
        for img in best_batch:
            if img_counter >= num_samples:
                break
            img.save(output_dir / f"{img_counter:06d}.png")
            img_counter += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dit_path", default="/data/home/yxfeng/data/models/dit_model")
    p.add_argument("--imagenet_val", default="/data/yxfeng/datasets/imagenet/val")
    p.add_argument("--output_dir", default="/data/yxfeng/eval_dit_random_search_imagereward")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--num_candidates",
        type=int,
        default=8,
        help="每轮随机采样的候选 batch 数（不同 seed）",
    )
    p.add_argument("--deno_steps", type=int, default=10)
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
        default=32,
        help="前 N 张随机生成再进入 ImageReward 搜索",
    )
    p.add_argument(
        "--prompt_template",
        type=str,
        default="a high-quality picture of a {name}",
        help="ImageReward 文本；{name}、{id} 可用",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert args.deno_steps >= 1
    assert 0 <= args.init_random_samples <= args.num_samples

    output_dir = Path(args.output_dir)
    imagenet_val = Path(args.imagenet_val)
    assert imagenet_val.exists(), f"ImageNet val 未找到: {imagenet_val}"

    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_1k = load_imagenet_simple_labels()

    print("1. 加载 ImageReward...")
    reward_model = RM.load("ImageReward-v1.0")

    print("2. 加载 DiT...")
    pipe = model_loader(
        Path(args.dit_path),
        device,
        dtype="fp16" if args.fp16 else "fp32",
    )

    n_init = 0
    if args.init_random_samples > 0:
        print("3a. 随机初始化...")
        n_init = init_random_samples(
            pipe,
            output_dir,
            args.init_random_samples,
            args.batch_size,
            args.deno_steps,
            device,
            args.seed,
            guidance_scale=args.guidance_scale,
        )
        print(f"   已写入 {n_init} 张")

    print("3b. Random Search (ImageReward verifier)...")
    print(f"   deno_steps={args.deno_steps}, num_candidates={args.num_candidates}")
    run_random_search_imagereward(
        pipe,
        reward_model,
        labels_1k,
        args.prompt_template,
        output_dir,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        args.deno_steps,
        device,
        args.seed,
        guidance_scale=args.guidance_scale,
        n_init=n_init,
    )

    log_path = output_dir / "config.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "verifier": "imagereward",
                "init_random_samples": args.init_random_samples,
                "deno_steps": args.deno_steps,
                "num_candidates": args.num_candidates,
                "prompt_template": args.prompt_template,
                "fp16": args.fp16,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n完成。图像保存至 {output_dir}，配置: {log_path}")

    print("\n4. 计算最终 FID / IS...")
    fid_val, is_val = compute_fid_and_is(
        output_dir, imagenet_val, device, real_cache_name=args.ref_cache_name
    )
    print(f"   最终 FID: {fid_val:.2f}, IS: {is_val:.4f}")
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"FID": fid_val, "IS": is_val}, f, indent=2)


if __name__ == "__main__":
    main()
