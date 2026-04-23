#!/usr/bin/env python3
"""
DiT Random Search，用训练好的 CrossSampleDiffusionRanker 做 verifier（推理时 scaling）。

与 dit_random_search_imagereward.py 结构一致：每轮从 K 个候选 batch 中选
「batch 内 ranker 打分（logits）平均值最高」的一批写入输出目录；类条件为随机 ImageNet 类（与 IR 版相同）。

结束后对 output_dir 与 ImageNet val 计算 FID / IS（torch_fidelity，与 IR 版一致）。

依赖：同仓库 train_ranker_imagenet（rollout 与 ranker 结构）、dit_random_search_common、utils.compute_fid_and_is。

用法:
  CUDA_VISIBLE_DEVICES=6 python3 dit_random_search_ranker.py \
    --ranker_dir /data/home/yxfeng/diffusion_ranker/outputs/ranker_mse_kl/g32_alpha0.3_layer_ratio0.6_lr1e-4_cosine/ranker_diffusion_epoch_0003.pt \
    --dit_path /data/yxfeng/models/dit_model \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --output_dir /data/yxfeng/eval_dit_random_search_ranker \
    --num_samples 50000 \
    --init_random_samples 32 \
    --deno_steps 10 \
    --guidance_scale 1.0 \
    --batch_size 32 \
    --num_candidates 8
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

import train_ranker_imagenet as tr
from dit_random_search_common import (
    CANDIDATE_SEED_HIGH,
    init_random_samples,
    load_dit,
    rollout_dit_batch_with_pooled_hidden,
)
from utils import compute_fid_and_is, setup_seed


def load_ranker_and_meta(
    ranker_dir: Path,
    ckpt_name: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:

    cfg_path = ranker_dir / "run_config.json"
    ckpt_path = ranker_dir / ckpt_name
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    dit_hidden_dim = int(cfg["dit_hidden_dim"])
    ranker = tr.CrossSampleDiffusionRanker(
        dit_hidden_dim=dit_hidden_dim,
        proj_dim=int(cfg["ranker_proj_dim"]),
        num_tf_layers=int(cfg["ranker_tf_layers"]),
        n_heads=int(cfg.get("ranker_heads", 4)),
        ff_mult=int(cfg.get("ranker_ff_mult", 4)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ranker.load_state_dict(ckpt["ranker_state_dict"])
    ranker.eval()

    return ranker, cfg


def resolve_pipe_and_layer(pipe: DiTPipeline, cfg: dict) -> int:
    n_blocks = len(pipe.transformer.transformer_blocks)
    return tr.resolve_hidden_layer_idx(
        n_blocks,
        cfg.get("layer_ratio", 0.6),
        cfg.get("hidden_layer_idx"),
    )


def mean_batch_ranker_score(
    ranker: torch.nn.Module,
    pooled_h: torch.Tensor,
    device: torch.device,
) -> float:
    """pooled_h: [B, D] CPU；返回 batch 内 ranker logits 的均值（越高越好）。"""
    with torch.no_grad():
        h = pooled_h.to(device=device, dtype=torch.float32)
        logits = ranker(h)
        return float(logits.float().mean().item())


def run_random_search_ranker(
    pipe: DiTPipeline,
    ranker: torch.nn.Module,
    hidden_layer_idx: int,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    deno_steps: int,
    device: str,
    device_torch: torch.device,
    seed: int,
    guidance_scale: float = 1.0,
    n_init: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for _ in tqdm(range(num_rounds), desc="Random Search Ranker"):
        n_this_batch = min(batch_size, num_samples - img_counter)
        if n_this_batch <= 0:
            break

        candidate_seeds = rng.integers(0, CANDIDATE_SEED_HIGH, size=num_candidates).tolist()
        best_score = float("-inf")
        best_batch: list[Image.Image] | None = None

        for cand_seed in candidate_seeds:
            batch_images, _class_ids, pooled = rollout_dit_batch_with_pooled_hidden(
                pipe,
                deno_steps,
                n_this_batch,
                device,
                seed=int(cand_seed),
                hidden_layer_idx=hidden_layer_idx,
                guidance_scale=guidance_scale,
            )
            score = mean_batch_ranker_score(ranker, pooled, device_torch)
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
    p.add_argument(
        "--ranker_dir",
        type=str,
        default=str(_ROOT / "outputs" / "ranker_mse_kl"),
        help="含 run_config.json 与 ranker_diffusion.pt",
    )
    p.add_argument("--ckpt_name", type=str, default="ranker_diffusion.pt")
    p.add_argument("--dit_path", type=str, default=None, help="覆盖 run_config 中的 dit_path（默认用配置）")
    p.add_argument("--imagenet_val", default="/data/yxfeng/datasets/imagenet/val")
    p.add_argument("--output_dir", default="/data/yxfeng/eval_dit_random_search_ranker")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument(
        "--num_candidates",
        type=int,
        default=8,
        help="每轮随机采样的候选 batch 数（不同 seed）",
    )
    p.add_argument("--deno_steps", type=int, default=None, help="默认使用 run_config.deno_steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=None, help="默认使用 run_config.guidance_scale")
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DiT 使用 float16（默认开启）；可用 --no-fp16 关闭",
    )
    p.add_argument("--ref_cache_name", default="imagenet_val")
    p.add_argument("--init_random_samples", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ranker_dir = Path(args.ranker_dir)
    ckpt_name = args.ckpt_name
    if ranker_dir.is_file() or ranker_dir.suffix in (".pt", ".pth", ".bin"):
        ckpt_name = ranker_dir.name
        ranker_dir = ranker_dir.parent
    deno_steps_cfg = args.deno_steps
    guidance_cfg = args.guidance_scale

    ranker, cfg = load_ranker_and_meta(ranker_dir, ckpt_name, torch.device("cpu"))
    if deno_steps_cfg is None:
        deno_steps_cfg = int(cfg["deno_steps"])
    if guidance_cfg is None:
        guidance_cfg = float(cfg["guidance_scale"])

    assert deno_steps_cfg >= 1
    assert 0 <= args.init_random_samples <= args.num_samples

    dit_path = args.dit_path or cfg.get("dit_path")
    if not dit_path:
        raise SystemExit("run_config 中无 dit_path，请传 --dit_path")

    output_dir = Path(args.output_dir)
    imagenet_val = Path(args.imagenet_val)
    if not imagenet_val.exists():
        raise SystemExit(f"ImageNet val 未找到: {imagenet_val}")

    setup_seed(args.seed)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_torch = torch.device(device_str)
    ranker = ranker.to(device_torch)

    print("1. 加载 DiT...")
    pipe = load_dit(Path(dit_path), device_str, fp16=args.fp16)
    hidden_layer_idx = resolve_pipe_and_layer(pipe, cfg)
    n_blocks = len(pipe.transformer.transformer_blocks)
    print(f"   capture_block={hidden_layer_idx}/{n_blocks}, deno_steps={deno_steps_cfg}, guidance={guidance_cfg}")
    
    print("2a. 随机初始化...")
    n_init = init_random_samples(
        pipe,
        output_dir,
        args.init_random_samples,
        args.batch_size,
        deno_steps_cfg,
        device_str,
        args.seed,
        guidance_scale=guidance_cfg,
    ) if args.init_random_samples > 0 else 0
    print(f"   已写入 {n_init} 张")

    print("2b. Random Search (Ranker verifier, mean logits per batch)...")
    print(f"   num_candidates={args.num_candidates}")
    run_random_search_ranker(
        pipe,
        ranker,
        hidden_layer_idx,
        output_dir,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        deno_steps_cfg,
        device_str,
        device_torch,
        args.seed,
        guidance_scale=guidance_cfg,
        n_init=n_init,
    )

    log_path = output_dir / "config.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "verifier": "ranker",
                "ranker_dir": str(ranker_dir.resolve()),
                "ckpt_name": args.ckpt_name,
                "dit_path": str(dit_path),
                "hidden_layer_idx": hidden_layer_idx,
                "init_random_samples": args.init_random_samples,
                "deno_steps": deno_steps_cfg,
                "guidance_scale": guidance_cfg,
                "num_candidates": args.num_candidates,
                "batch_score": "mean(ranker_logits_over_batch)",
                "fp16": args.fp16,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n完成。图像保存至 {output_dir}，配置: {log_path}")

    print("\n3. 计算最终 FID / IS...")
    fid_val, is_val = compute_fid_and_is(
        output_dir, imagenet_val, device_str, real_cache_name=args.ref_cache_name
    )
    print(f"   最终 FID: {fid_val:.2f}, IS: {is_val:.4f}")
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"FID": fid_val, "IS": is_val}, f, indent=2)


if __name__ == "__main__":
    main()
