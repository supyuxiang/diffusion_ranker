#!/usr/bin/env python3
"""
DiT Random Search + FID verifier（仅读预生成图）。init 固定 N_INIT=1024（论文）。

须先用 ``search/rollout.py`` 生成 ``candidate_0000`` …；可选 ``init_1024/``。

    /root/autodl-tmp/dit_candidates_50k_cfg{gs}_deno{steps}/candidate_XXXX/image_*_*.png

用法:

  python3 search/dit_random_search_fid_reward.py \
      --imagenet_val /root/autodl-tmp/imagenet/val \
      --output_dir /root/autodl-tmp/eval_dit_rs_fid \
      --deno_steps 10 --guidance_scale 1.0 \
      --num_candidates 16 --batch_size 24
"""

from __future__ import annotations

N_INIT = 1024

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


def extract_features_from_images(
    images: list[Image.Image],
    feat_extractor,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    from torch_fidelity.datasets import TransformPILtoRGBTensor

    transform = Compose([Resize(256), CenterCrop(256), TransformPILtoRGBTensor()])
    feat_extractor.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            tensors = torch.stack(
                [transform(img) for img in images[i : i + batch_size]],dim=0
            )
            tensors = tensors.cuda(non_blocking=True)
            chunks.append(feat_extractor(tensors).cpu().numpy())
    return np.concatenate(chunks, axis=0)  # (N, 2048)


def compute_ref_statistics() -> dict:

    real_dir = Path("/root/autodl-tmp/imagenet/val")
    cache_name = "imagenet_val"

    from torch_fidelity.metric_fid import fid_input_id_to_statistics_cached
    from torch_fidelity.utils import create_feature_extractor

    kwargs = dict(
        input2=str(real_dir),
        input2_cache_name=cache_name,
        samples_find_deep=True,
        samples_find_ext="png,jpg,jpeg,JPEG,JPG",
        samples_resize_and_crop=256,
        cuda=True,
        batch_size=512,
        verbose=True,
        cache=True,
        feature_extractor_weights_path=INCEPTION_WEIGHTS_PATH,
    )
    feat_extractor = create_feature_extractor("inception-v3-compat", ["2048"], **kwargs)
    # 已是 {"mu","sigma"}；勿对返回值再调 fid_features_to_statistics（后者输入须为 (N,2048) 特征）
    return fid_input_id_to_statistics_cached(2, feat_extractor, "2048", **kwargs)


def init_stats_loader(
    feat_extractor,
    init_dir:Path,
    device: torch.device,
) -> tuple[dict, int]:
    init_path_ls = sorted(init_dir.glob("image_*_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    init_image_ls = [Image.open(path).convert("RGB").copy() for path in init_path_ls]
    init_stats = fid_features_to_statistics(
        extract_features_from_images(init_image_ls, feat_extractor, device)
    )
    return init_stats, len(init_image_ls)



@torch.inference_mode()
def run_random_search_fid(
    feat_extractor,
    ref_stats: dict,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    device: torch.device,
    candidate_path_lists: list[list[Path]],
    init_stats: dict,
    n_init: int,
) -> None:
    from reward import fid_reward, merge_fid_statistics, fid_statistics_to_distance, fid_features_to_statistics
    
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_stats, n_selected = init_stats, n_init
    cursor = img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for _ in tqdm(range(num_rounds), desc="random search fid"):
        n_this_batch = min(batch_size, num_samples - cursor)
        if n_this_batch <= 0:
            break
        best_fid = float("inf")
        best_batch_images: list[Image.Image] | None = None
        best_batch_stats: dict | None = None

        for candidate_idx in range(num_candidates):
            batch_images = [
                Image.open(p).convert("RGB").copy()
                for p in candidate_path_lists[candidate_idx][
                    cursor : cursor + n_this_batch
                ]
            ]
            batch_stats = fid_features_to_statistics(
                extract_features_from_images(
                    batch_images,
                    feat_extractor,
                    device,
                    min(64, n_this_batch),
                )
            )
            merged_stats = merge_fid_statistics(
                selected_stats, n_selected, batch_stats, len(batch_images)
            )
            fid = fid_statistics_to_distance(merged_stats, ref_stats)
            if fid < best_fid:
                best_fid = fid
                best_batch_images = batch_images
                best_batch_stats = batch_stats

        assert best_batch_images is not None and best_batch_stats is not None
        for image in best_batch_images:
            if img_counter >= num_samples:
                break
            image.save(output_dir / f"{img_counter:06d}.png")
            img_counter += 1
        selected_stats = merge_fid_statistics(
            selected_stats, n_selected, best_batch_stats, len(best_batch_images)
        )
        n_selected += len(best_batch_images)
        cursor += n_this_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/root/autodl-tmp/eval_dit_random_search_fid")
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--deno_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics_batch_size", type=int, default=64)
    return parser.parse_args()




def main() -> None:

    imagenet_val = Path("/root/autodl-tmp/imagenet/val")
    cache_name = "imagenet_val"

    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    rollout_root = Path(
        f'/root/autodl-tmp/dit_candidates_50k_cfg{args.guidance_scale}_deno{args.deno_steps}'
    )

    # candidate_dir = rollout_root / f"candidate_{i:04d}"
    # init_dir = rollout_root / "init1024"
    # load candidate dir and init dir
    candidate_dir_ls = [
        rollout_root / f'candidate_{i:04d}' for i in range(args.num_candidates)
    ]
    init_dir = rollout_root / "init1024"
    
    # check 
    assert all([
        [len([path for path in candidate_dir.glob("image_*_*.png")])==int(50000-1024)]
        for candidate_dir in candidate_dir_ls
    ]
    ) and len([path for path in init_dir.glob("image_*_*.png")])==1024
    , f"init_dir or candidate_dir_ls is not correct"


    set_seed(args.seed)
    device = set_device()

    print("ref statistics …")
    ref_stats = compute_ref_statistics() # dict: {"mu": (2048,), "sigma": (2048, 2048)}
    print(f"   μ {ref_stats['mu'].shape}, Σ {ref_stats['sigma'].shape}")
    feat_extractor = get_inception_feature_extractor(
        "cuda" if device.type == "cuda" else "cpu"
    ) # built inception_feature_extractor

    # prepare candidate pathes list : list[list[Path]] : [[path1, path2, ...], [path1, path2, ...], ...]
    candidate_pathes_ls: list[list[Path]] = []
    for i in range(args.num_candidates):
        candidate_dir = candidates_root / f"candidate_{i:04d}"
        path_list = sorted(
            candidate_dir.glob("image_*_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        candidate_pathes_ls.append(path_list)

    # prepare init pathes list : list[Path] : [path1, path2, ...]
    init_source_dir = Path(args.init_dir).expanduser().resolve() if args.init_dir else None
    init_path_ls = (
        sorted(
            init_source_dir.glob("image_*_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
    )
    assert len(init_path_ls) ==  1024

    init_stats, n_init = init_stats_loader(
        feat_extractor, output_dir, init_path_ls, device
    )

    run_random_search_fid(
        feat_extractor,
        ref_stats,
        output_dir,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        device,
        candidate_pathes_ls,
        init_stats,
        n_init,
    )

    gc.collect()
    torch.cuda.empty_cache()
    

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "imagenet_val": str(imagenet_val),
                "candidates_root": str(candidates_root),
                "init_dir": str(init_source_dir) if init_source_dir else "",
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "num_candidates": args.num_candidates,
                "n_init": N_INIT,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "ref_cache_name": args.ref_cache_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metrics = compute_fid_is_from_dirs(
        output_dir, imagenet_val, device, batch_size=args.metrics_batch_size
    )
    print(f"FID {metrics['FID']:.2f}, IS {metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}")
    (output_dir / "final_metrics.json").write_text(
        json.dumps(
            {k: float(v) for k, v in metrics.items()}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
