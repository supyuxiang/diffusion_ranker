#!/usr/bin/env python3
"""
用 DiT 按指定 ImageNet 类生成多张图，用 ImageReward 打分：
  - 全体平均 reward；
  - 按 reward 取 top-k，再算这 k 张的平均 reward。

依赖与 test_imagereward.py 一致：ImageReward-main 在默认路径或 IMAGEREWARD_ROOT。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_DEFAULT_IR = _ROOT.parent / "ImageReward-main"
if (_ir := Path(os.environ.get("IMAGEREWARD_ROOT", str(_DEFAULT_IR)))).is_dir():
    sys.path.insert(0, str(_ir))

import numpy as np
import torch
from tqdm import tqdm

import ImageReward as RM
from utils import load_dit


def load_imagenet_simple_labels(cache_dir: Path | None = None) -> list[str]:
    cache_dir = cache_dir or (Path.home() / ".cache" / "diffusion_ranker")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "imagenet-simple-labels.json"
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    if not path.exists():
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"警告: 无法下载 ImageNet 标签 ({e})，将使用占位 prompt。")
            return [f"imagenet class {i}" for i in range(1000)]
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if len(labels) != 1000:
        return [f"imagenet class {i}" for i in range(1000)]
    return labels


def prompt_for_class(labels: list[str], class_id: int, template: str) -> str:
    name = labels[class_id] if 0 <= class_id < len(labels) else f"class {class_id}"
    return template.format(name=name, id=class_id)


def generate_images_batched(
    pipe,
    class_id: int,
    total: int,
    gen_batch_size: int,
    num_inference_steps: int,
    device: torch.device,
    base_seed: int,
) -> list:
    out = []
    for start in range(0, total, gen_batch_size):
        k = min(gen_batch_size, total - start)
        labels = [class_id] * k
        gen = torch.Generator(device=device).manual_seed(base_seed + start)
        batch = pipe(
            labels,
            num_inference_steps=num_inference_steps,
            generator=gen,
        ).images
        out.extend(batch)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class_id", type=int, default=218, help="ImageNet 类 id，默认 218")
    p.add_argument("--num_images", type=int, default=1000)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--dit_path", type=str, default="/data/yxfeng/models/dit_model")
    p.add_argument("--gen_batch_size", type=int, default=16, help="单次 DiT 前向 batch，按显存调整")
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompt_template",
        type=str,
        default="a high-quality picture of a {name}",
        help="与 train_ranker_imagenet 默认一致；{name} {id} 可用",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="若非空则保存全部生成图（png）",
    )
    p.add_argument(
        "--results_json",
        type=str,
        default="",
        help="若非空则写入每张图的 reward 与汇总统计",
    )
    args = p.parse_args()

    if not (0 <= args.class_id < 1000):
        raise SystemExit("class_id 须在 [0, 999]")
    if args.top_k > args.num_images:
        raise SystemExit("top_k 不能大于 num_images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_1k = load_imagenet_simple_labels()
    text_prompt = prompt_for_class(labels_1k, args.class_id, args.prompt_template)
    print(f"ImageReward 文本条件: {text_prompt!r}")

    print("加载 DiT ...")
    pipe = load_dit(args.dit_path, device=str(device), fp16=True)

    print(f"生成 {args.num_images} 张图 (class_id={args.class_id}) ...")
    images = generate_images_batched(
        pipe,
        args.class_id,
        args.num_images,
        args.gen_batch_size,
        args.num_inference_steps,
        device,
        args.seed,
    )

    save_root = Path(args.save_dir) if args.save_dir else None
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(save_root / f"class{args.class_id}_{i:04d}.png")

    print("加载 ImageReward ...")
    reward_model = RM.load("ImageReward-v1.0")

    rewards = []
    with torch.no_grad():
        for img in tqdm(images, desc="ImageReward"):
            r = float(reward_model.score(text_prompt, img))
            rewards.append(r)

    rewards_np = np.asarray(rewards, dtype=np.float64)
    mean_all = float(rewards_np.mean())
    top_idx = np.argsort(rewards_np)[-args.top_k :]
    mean_top = float(rewards_np[top_idx].mean())

    print("-" * 60)
    print(f"类 {args.class_id} | 张数 {args.num_images} | top_k {args.top_k}")
    print(f"全体平均 ImageReward: {mean_all:.6f}")
    print(f"top-{args.top_k} 平均 ImageReward: {mean_top:.6f}")
    print("-" * 60)

    if args.results_json:
        out_path = Path(args.results_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class_id": args.class_id,
            "text_prompt": text_prompt,
            "num_images": args.num_images,
            "top_k": args.top_k,
            "mean_reward_all": mean_all,
            "mean_reward_top_k": mean_top,
            "top_k_indices": top_idx.tolist(),
            "per_image_reward": rewards,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
