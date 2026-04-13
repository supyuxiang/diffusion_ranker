#!/usr/bin/env python3
"""
DiT Random Search with FID as Verifier（仅预生成图）

对应 *Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps*
(arXiv:2501.09732)。须先用 ``search/rollout.py`` 写好 ``candidate_0000`` … 与可选 ``init_1024``；
本脚本只读盘 + Inception + FID 搜索，**不加载 DiT**。

目录（与 ``rollout.py`` 一致）::

    /root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}/
        init_1024/              # 可选；否则用 candidate_0000 前 N 张
        candidate_0000/ …

用法::

  python3 search/dit_random_search_fid_reward.py \
      --imagenet_val /root/autodl-tmp/imagenet/val \
      --output_dir /root/autodl-tmp/eval_dit_rs_fid \
      --deno_steps 10 --guidance_scale 1.0 \
      --num_candidates 16 --batch_size 24 --init_random_samples 1024 \
      --model_name dit-xl-2-256 \
      --dtype bf16
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
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import (
    INCEPTION_WEIGHTS_PATH,
    compute_fid_is_from_dirs,
    fid_features_to_statistics,
    fid_statistics_to_distance,
    get_inception_feature_extractor,
    merge_fid_statistics,
    set_device,
    set_seed,
)


def default_candidates_root(guidance_scale: float, deno_steps: int) -> Path:
    return Path(
        f"/root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}"
    )


def sorted_rollout_image_paths(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def load_images_from_paths(paths: list[Path]) -> list[Image.Image]:
    out: list[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())
    return out


def build_candidate_path_lists(root: Path, num_candidates: int) -> list[list[Path]]:
    path_lists: list[list[Path]] = []
    for i in range(num_candidates):
        d = root / f"candidate_{i:04d}"
        path_ls = sorted_rollout_image_paths(d)
        path_lists.append(path_ls)
    return path_lists


def resolve_init_paths(
    candidates_root: Path,
    init_dir: Path | None,
    n_init: int,
    candidate_lists: list[list[Path]],
) -> list[Path]:
    if n_init <= 0:
        return []
    if init_dir is not None and init_dir.is_dir():
        pl = sorted_rollout_image_paths(init_dir)
        if len(pl) < n_init:
            raise RuntimeError(f"{init_dir}: need >= {n_init} images, got {len(pl)}")
        return pl[:n_init]
    pl0 = candidate_lists[0]
    if len(pl0) < n_init:
        raise RuntimeError(f"candidate_0000: need >= {n_init} images for init, got {len(pl0)}")
    return pl0[:n_init]


def _device_cuda_str(device: torch.device | str) -> str:
    if isinstance(device, str):
        return "cuda" if device == "cuda" else "cpu"
    return "cuda" if device.type == "cuda" else "cpu"


def extract_features_from_images(
    images: list[Image.Image],
    feat_extractor,
    device: torch.device | str,
    batch_size: int = 512,
) -> np.ndarray:
    from torch_fidelity.datasets import TransformPILtoRGBTensor

    transform = Compose(
        [
            Resize(256),
            CenterCrop(256),
            TransformPILtoRGBTensor(),
        ]
    )
    dstr = _device_cuda_str(device)

    feats_list = []
    feat_extractor.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            tensors = torch.stack([transform(img) for img in batch_imgs])
            if dstr == "cuda":
                tensors = tensors.cuda(non_blocking=True)
            out = feat_extractor(tensors)
            feats_list.append(out.cpu().numpy())
    return np.concatenate(feats_list, axis=0)


def compute_ref_statistics(real_dir: Path, device: torch.device | str, cache_name: str = "imagenet_val") -> dict:
    from torch_fidelity.metric_fid import fid_input_id_to_statistics_cached
    from torch_fidelity.utils import create_feature_extractor

    cuda = _device_cuda_str(device) == "cuda"
    kwargs = dict(
        input2=str(real_dir),
        input2_cache_name=cache_name,
        samples_find_deep=True,
        samples_find_ext="png,jpg,jpeg,JPEG,JPG",
        samples_resize_and_crop=256,
        cuda=cuda,
        batch_size=512,
        verbose=True,
        cache=True,
        feature_extractor_weights_path=INCEPTION_WEIGHTS_PATH,
    )
    feat_extractor = create_feature_extractor("inception-v3-compat", ["2048"], **kwargs)
    feat_layer = "2048"
    return fid_input_id_to_statistics_cached(2, feat_extractor, feat_layer, **kwargs)


def init_from_precomputed(
    feat_extractor,
    output_dir: Path,
    init_paths: list[Path],
    device: torch.device,
) -> tuple[dict, int]:
    """从磁盘读 init，写入 ``output_dir`` 为 ``%06d.png``。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = load_images_from_paths(init_paths)
    for i, img in enumerate(images):
        img.save(output_dir / f"{i:06d}.png")
    feats = extract_features_from_images(images, feat_extractor, device, batch_size=512)
    return fid_features_to_statistics(feats), len(images)


@torch.inference_mode()
def run_random_search_fid(
    feat_extractor,
    ref_stats: dict,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    device: torch.device,
    candidate_lists: list[list[Path]],
    init_stats: dict | None = None,
    n_init: int = 0,
) -> None:
    """每轮从各 candidate 池同一全局下标切片，选相对 ref FID 最低的一批写入 ``output_dir``。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for c in range(num_candidates):
        if len(candidate_lists[c]) < num_samples:
            raise RuntimeError(
                f"candidate_{c:04d} has {len(candidate_lists[c])} images, need >= {num_samples}"
            )

    selected_stats = init_stats
    n_selected = n_init
    cursor = n_init
    img_counter = n_init

    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size
    for _ in tqdm(range(num_rounds), desc="random search fid"):
        n_this_batch = min(batch_size, num_samples - cursor)
        if n_this_batch <= 0:
            break

        best_fid = float("inf")
        best_batch: list[Image.Image] | None = None
        best_batch_stats: dict | None = None

        for c in range(num_candidates):
            sub = candidate_lists[c][cursor : cursor + n_this_batch]
            batch_images = load_images_from_paths(sub)
            batch_feats = extract_features_from_images(
                batch_images, feat_extractor, device, batch_size=min(64, n_this_batch)
            )
            batch_stats = fid_features_to_statistics(batch_feats)

            if selected_stats is None:
                merged_stats = batch_stats
            else:
                merged_stats = merge_fid_statistics(
                    selected_stats, n_selected, batch_stats, len(batch_images)
                )

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
        cursor += n_this_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--imagenet_val",
        default="/root/autodl-tmp/imagenet/val",
        help="解压后的 ImageNet val 根目录",
    )
    p.add_argument("--output_dir", default="/root/autodl-tmp/eval_dit_random_search_fid")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--init_random_samples", type=int, default=1024, help="0 表示无 init")
    p.add_argument("--deno_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--ref_cache_name", default="imagenet_val")
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument(
        "--candidates_root",
        default="",
        help="预生成根目录；空则用 dit_candidates_50k_cfg{gs}_deno{steps}",
    )
    p.add_argument(
        "--init_dir",
        default="",
        help="init 目录；空则尝试 candidates_root/init_1024，否则 candidate_0000 前 N 张",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()

    if not imagenet_val.is_dir():
        raise FileNotFoundError(f"--imagenet_val is not a directory: {imagenet_val}")

    set_seed(args.seed)
    device = set_device()

    print("precompute real set FID statistics (torch_fidelity)")
    ref_stats = compute_ref_statistics(imagenet_val, device, args.ref_cache_name)
    print(f"   μ_ref shape: {ref_stats['mu'].shape}, Σ_ref shape: {ref_stats['sigma'].shape}")

    print("create inception feature extractor")
    feat_extractor = get_inception_feature_extractor(_device_cuda_str(device))

    cr = (
        Path(args.candidates_root).expanduser().resolve()
        if args.candidates_root
        else default_candidates_root(args.guidance_scale, args.deno_steps)
    )
    if not cr.is_dir():
        raise FileNotFoundError(
            f"candidates root missing: {cr}\nRun search/rollout.py per candidate_idx, or set --candidates_root."
        )

    init_dir_opt = Path(args.init_dir).expanduser().resolve() if args.init_dir else None
    auto_init = cr / "init_1024"
    if init_dir_opt is None and auto_init.is_dir() and sorted_rollout_image_paths(auto_init):
        init_dir_opt = auto_init

    lists = build_candidate_path_lists(cr, args.num_candidates)

    init_stats: dict | None = None
    n_init = 0
    if args.init_random_samples > 0:
        init_paths = resolve_init_paths(cr, init_dir_opt, args.init_random_samples, lists)
        print(f"init from disk ({len(init_paths)} images) -> {output_dir}")
        init_stats, n_init = init_from_precomputed(feat_extractor, output_dir, init_paths, device)
        print(f"   μ̂ shape={init_stats['mu'].shape}")

    print(
        f"random search: root={cr}, num_candidates={args.num_candidates}, batch_size={args.batch_size}"
    )
    run_random_search_fid(
        feat_extractor,
        ref_stats,
        output_dir,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        device,
        lists,
        init_stats=init_stats,
        n_init=n_init,
    )

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    resolved_root = (
        str(Path(args.candidates_root).expanduser().resolve())
        if args.candidates_root
        else str(default_candidates_root(args.guidance_scale, args.deno_steps))
    )
    cfg_path = output_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "imagenet_val": str(imagenet_val),
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "num_candidates": args.num_candidates,
                "init_random_samples": args.init_random_samples,
                "deno_steps": args.deno_steps,
                "seed": args.seed,
                "guidance_scale": args.guidance_scale,
                "ref_cache_name": args.ref_cache_name,
                "candidates_root": resolved_root,
                "init_dir": str(Path(args.init_dir).expanduser().resolve()) if args.init_dir else "",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"config saved to {cfg_path}")

    print("\nfinal FID / IS (torch_fidelity, compute_fid_is_from_dirs)")
    metrics = compute_fid_is_from_dirs(
        output_dir,
        imagenet_val,
        device,
        batch_size=args.metrics_batch_size,
    )
    print(f"   FID: {metrics['FID']:.2f}, IS: {metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}")

    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2, ensure_ascii=False)
    print(f"final metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
