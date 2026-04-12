#!/usr/bin/env python3
"""
DiT Random Search with FID as Verifier (Inference-Time Scaling)

论文方法：增量构造 + FID 作为 verifier 信号（对齐 Appendix A.4）
- 预计算真实集 (ImageNet) 的 μ_ref, Σ_ref（torch_fidelity 缓存）
- 固定 ``deno_steps``，Random Search 对随机种子（不同噪声/类别）
- 初始化：前 ``init_random_samples`` 张图随机生成，用于初始化 μ̂, Σ̂
- 迭代：每轮生成 K 批候选（不同 seed），选使 FID（解析式）相对 μ_ref, Σ_ref 下降最多的一批加入已选集

**最终** FID / IS（及可选 KID）由 ``utils.compute_metrics``（torchmetrics）在生成目录与真实目录上计算，与
``eval_dit_scaling_laws_rollout_level.py`` 一致。

**权重精度**：``--dtype bf16|fp16|fp32``（见 ``utils.load_dit``），可选 ``--compile`` 加速 transformer。

``--imagenet_val`` 必须指向**已解压**的 ILSVRC val 图片根目录（递归搜索常见图片后缀）。
autodl 镜像里 ``/root/autodl-pub/ImageNet/ILSVRC2012`` 只有 ``ILSVRC2012_img_val.tar`` 等压缩包，**不能**直接当路径用；请先解压（同目录 ``extract_imagenet.py`` 默认解压到 ``/root/autodl-tmp/imagenet/val``）。

**无外网**：torch-fidelity / FID 用的 Inception 权重默认会从 GitHub 拉取。请在本机下载
``weights-inception-2015-12-05-6726825d.pth``（torch-fidelity v0.2.0 release）拷到服务器
``/root/autodl-pub/torch_fidelity_weights/`` 或 ``~/.cache/torch/hub/checkpoints/``，或设置环境变量
``TORCH_FIDELITY_INCEPTION_WEIGHTS`` 指向该文件。详见 ``utils.resolve_fidelity_inception_weights_path``。

用法:
  CUDA_VISIBLE_DEVICES=0 python3 /root/diffusion_ranker_main/search_algoriths/dit_random_search_fid_reward.py \
    --model_name dit-xl-2-256 \
    --imagenet_val /root/autodl-tmp/imagenet/val \
    --output_dir /root/autodl-tmp/eval_dit_random_search_fid \
    --num_samples 50000 \
    --batch_size 24 \
    --num_candidates 8 \
    --init_random_samples 1024 \
    --deno_steps 10 \
    --dtype bf16 \
    --compile
"""

from __future__ import annotations

import argparse
import gc
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
    rollout_dit_batch_fid_images,
)

from utils import compute_metrics, fid_features_to_statistics, fid_statistics_to_distance, load_dit, merge_fid_statistics, resolve_fidelity_inception_weights_path, set_seed


def get_inception_feature_extractor(device: str):
    from torch_fidelity.utils import create_feature_extractor

    w = resolve_fidelity_inception_weights_path()
    kw = dict(feature_extractor_weights_path=w) if w else {}
    return create_feature_extractor(
        "inception-v3-compat",
        ["2048"],
        cuda=(device == "cuda"),
        verbose=False,
        **kw,
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

    w = resolve_fidelity_inception_weights_path()
    inc_kw = dict(feature_extractor_weights_path=w) if w else {}
    kwargs = dict(
        input2=str(real_dir),
        input2_cache_name=cache_name,
        samples_find_deep=True,
        samples_find_ext="png,jpg,jpeg,JPEG,JPG",
        samples_resize_and_crop=256,
        cuda=(device == "cuda"),
        batch_size=64,
        verbose=True,
        cache=True,
        **inc_kw,
    )
    feat_extractor = create_feature_extractor("inception-v3-compat", ["2048"], **kwargs)
    feat_layer = "2048"
    return fid_input_id_to_statistics_cached(2, feat_extractor, feat_layer, **kwargs)


def _parse_fid_real_max_samples(s: str) -> int | None:
    t = str(s).strip().lower()
    if t in ("none", "null", ""):
        return None
    return int(t)


@torch.inference_mode()
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
    for batch_idx in tqdm(range(num_batches), desc="random init"):
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


@torch.inference_mode()
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
    for _ in tqdm(range(num_rounds), desc="random search fid"):
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
            else:
                merged_stats = merge_fid_statistics(selected_stats, n_selected, batch_stats, n_this_batch)

            fid = fid_statistics_to_distance(merged_stats, ref_stats)
            if fid < best_fid:
                best_fid = fid
                best_batch = batch_images
                best_batch_stats = batch_stats

        assert best_batch is not None and best_batch_stats is not None

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
    p.add_argument("--model_name", default="dit-xl-2-256")
    p.add_argument(
        "--imagenet_val",
        default="/root/autodl-tmp/imagenet/val",
        help=(
            "解压后的 ILSVRC val 图片根目录（非 tar）。autodl extract_imagenet.py 默认输出到此路径。"
        ),
    )
    p.add_argument("--output_dir", default="/hy-tmp/diffusion_ranker/eval_dit_random_search_fid")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_candidates", type=int, default=8, help="每轮随机采样的候选 batch 数（不同 seed）")
    p.add_argument("--deno_steps", type=int, default=20, help="固定 denoising step 数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=("fp32", "fp16", "bf16"),
        help="Pipeline weight dtype (see utils.load_dit). Default bf16.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(transformer) on CUDA (mode=reduce-overhead).",
    )
    p.add_argument("--ref_cache_name", default="imagenet_val")
    p.add_argument(
        "--init_random_samples",
        type=int,
        default=1024,
        help="论文：前 N 张随机选择以初始化 μ̂, Σ̂，默认 1024",
    )
    p.add_argument(
        "--kid",
        action="store_true",
        help="最终 metrics 额外计算 KID（torchmetrics，更慢）。",
    )
    p.add_argument(
        "--fid_real_max_samples",
        type=_parse_fid_real_max_samples,
        default=None,
        help="最终 FID/KID 参考真实图上限；none 表示用目录内全部。默认 none。",
    )
    p.add_argument("--fid_real_subset_seed", type=int, default=0, help="参考子集抽样种子（仅当限制张数时）。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert args.deno_steps >= 1, "deno_steps 需 >= 1"
    assert 0 <= args.init_random_samples <= args.num_samples, "init_random_samples 需在 [0, num_samples] 内"

    output_dir = Path(args.output_dir)
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()
    if not imagenet_val.is_dir():
        raise FileNotFoundError(f"--imagenet_val is not a directory: {imagenet_val}")
    print(f"Using ImageNet real dir: {imagenet_val}")

    if resolve_fidelity_inception_weights_path() is None:
        print(
            "警告: 未找到本地 Inception 权重，将尝试经 torch.hub 从外网下载（国内无代理常失败）。"
            " 请将 weights-inception-2015-12-05-6726825d.pth 放到 "
            "/root/autodl-pub/torch_fidelity_weights/ 或设置 TORCH_FIDELITY_INCEPTION_WEIGHTS。"
        )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 预计算真实集 FID 统计量 (torch_fidelity 缓存，供搜索阶段解析式 FID)...")
    ref_stats = compute_ref_statistics(imagenet_val, device, args.ref_cache_name)
    print(f"   μ_ref shape: {ref_stats['mu'].shape}, Σ_ref shape: {ref_stats['sigma'].shape}")

    print("2. 加载 DiT...")
    pipe = load_dit(args.model_name, device, dtype=args.dtype)
    if args.compile and device == "cuda":
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        print("   torch.compile enabled (transformer)")

    print("3. 创建 Inception 特征提取器 (搜索内在线特征)...")
    feat_extractor = get_inception_feature_extractor(device)

    init_stats = None
    n_init = 0
    if args.init_random_samples > 0:
        print("4a. 随机初始化（论文：前 N 张）...")
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

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    log_path = output_dir / "config.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "imagenet_val": str(imagenet_val),
                "init_random_samples": args.init_random_samples,
                "deno_steps": args.deno_steps,
                "num_candidates": args.num_candidates,
                "dtype": args.dtype,
                "compile": bool(args.compile),
                "kid_final": args.kid,
                "fid_real_max_samples": args.fid_real_max_samples,
                "fid_real_subset_seed": args.fid_real_subset_seed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n完成。图像保存至 {output_dir}，配置: {log_path}")

    print("\n5. 最终 FID / IS（torchmetrics，与 rollout eval 一致）...")
    metrics = compute_metrics(
        output_dir,
        imagenet_val,
        device,
        real_cache_name=args.ref_cache_name,
        fid=True,
        isc=True,
        kid=args.kid,
        real_max_samples=args.fid_real_max_samples,
        real_subset_seed=args.fid_real_subset_seed,
    )
    fid_val = metrics.get("FID", float("nan"))
    is_val = metrics.get("IS", float("nan"))
    print(f"   FID: {fid_val:.2f}, IS: {is_val:.4f}" + (f", KID: {metrics['KID']:.6f}" if args.kid else ""))

    out_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) or v is None}
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
