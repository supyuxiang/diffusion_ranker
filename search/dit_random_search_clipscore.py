#!/usr/bin/env python3
"""
init 固定 N_INIT=1024（与 FID 脚本一致）。

example:

python3 /root/diffusion_ranker/search/dit_random_search_clipscore.py \
    --output_dir /root/autodl-tmp/dit_random_search_clipscore_reward \
    --num_samples 50000 \
    --deno_steps 10 \
    --guidance_scale 1.0 \
    --num_candidates 16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import compute_fid_is_clipscore, set_device, set_seed
from reward import load_clip, clipscore_reward, labels_1k

N_INIT = 1024

_DEFAULT_CLIP_DIR = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")



def load_images_and_class_ids(paths: list[Path]) -> tuple[list[Image.Image], list[int]]:
    images: list[Image.Image] = []
    cids: list[int] = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB").copy())
        cids.append(class_id_from_rollout_path(p))
    return images, cids


def save_best_images(best_path_ls: list[Path], output_dir: Path, *, start_idx: int = 0) -> None:
    for j, path in enumerate(best_path_ls):
        i = start_idx + j
        cid = class_id_from_rollout_path(path)
        dest = output_dir / f"image_{i:05d}_{cid:05d}.png"
        shutil.copy2(path, dest)


@torch.inference_mode()
def run_random_search_clipscore(
    clip_backend: str,
    clip_model,
    clip_aux,
    labels_1k: list[str],
    prompt_template: str,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    device_str: str,
    candidate_pathes_dict: dict[int, list[Path]],
    n_init: int,
) -> list[Path]:
    best_path_ls: list[Path] = []
    current_idx = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for c in range(num_candidates):
        if len(candidate_pathes_dict[c]) < num_samples:
            raise RuntimeError(
                f"candidate_{c:04d}: need >= {num_samples} images, got {len(candidate_pathes_dict[c])}"
            )

    for _ in tqdm(range(num_rounds), desc="random search clipscore"):
        n_this_batch = min(batch_size, num_samples - current_idx)
        if n_this_batch <= 0:
            break

        best_score = float("-inf")
        best_candidate_idx = -1

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            sub = candidate_path_ls[current_idx : current_idx + n_this_batch]
            batch_images, batch_classes = load_images_and_class_ids(sub)
            if clip_backend == "hf":
                score = mean_batch_clipscore_hf(
                    clip_model,
                    clip_aux,
                    device_str,
                    batch_images,
                    batch_classes,
                    labels_1k,
                    prompt_template,
                )
            else:
                tokenizer, image_transform = clip_aux
                score = mean_batch_clipscore_open_clip(
                    clip_model,
                    tokenizer,
                    image_transform,
                    device_str,
                    batch_images,
                    batch_classes,
                    labels_1k,
                    prompt_template,
                )
            if score > best_score:
                best_score = score
                best_candidate_idx = candidate_idx

        assert best_candidate_idx != -1
        win_paths = candidate_pathes_dict[best_candidate_idx][
            current_idx : current_idx + n_this_batch
        ]
        best_path_ls.extend(win_paths)
        current_idx += n_this_batch
        if current_idx >= num_samples:
            break

    return best_path_ls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/root/autodl-tmp/random_search_clipscore")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument(
        "--search_batch_size",
        type=int,
        default=32,
        help="每轮每个 candidate 池切片长度（search batch，默认 32）",
    )
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10, help="与 rollout 目录名一致")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)

    return p.parse_args()


def main() -> None:
    imagenet_val = Path("/root/autodl-tmp/imagenet/val")
    cache_name = "imagenet_val"

    args = parse_args()
    
    output_dir = Path(args.output_dir).expanduser().resolve()

    rollout_root = Path(
        f"/root/autodl-tmp/dit_candidates_50k_cfg{args.guidance_scale}_deno{args.deno_steps}"
    )
    candidate_dir_ls = [
        rollout_root / f"candidate_{i:04d}" for i in range(args.num_candidates)
    ]
    init_dir = rollout_root / "init1024"

    set_seed(args.seed)
    device = set_device()
    dstr = "cuda" if device.type == "cuda" else "cpu"

    clip, processor = load_clip(device)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        candidate_pathes_dict[i] = sorted(
            candidate_dir_ls[i].glob("image_*_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

    init_path_ls = sorted(
        init_dir.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    best_dir = output_dir / "best_images"
    print(f"init {N_INIT} -> {best_dir}")
    save_init_images(init_path_ls, best_dir)

    print(
        f"random search (clipscore): root={rollout_root}, "
        f"num_candidates={args.num_candidates}, batch_size={args.batch_size}"
    )
    best_path_ls = run_random_search_clipscore(
        args.clip_backend,
        clip_model,
        clip_aux,
        labels_1k,
        args.prompt_template,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        dstr,
        candidate_pathes_dict,
        N_INIT,
    )
    save_best_images(best_path_ls, best_dir, start_idx=N_INIT)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "verifier": "clipscore",
                "imagenet_val": str(imagenet_val),
                "candidates_root": str(rollout_root),
                "init_dir": str(init_dir),
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "num_candidates": args.num_candidates,
                "n_init": N_INIT,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "ref_cache_name": cache_name,
                "prompt_template": args.prompt_template,
                "clip_backend": args.clip_backend,
                **(
                    {
                        "clip_model_path": args.clip_model_path,
                        "clip_model_path_resolved": str(clip_dir_resolved),
                    }
                    if args.clip_backend == "hf" and clip_dir_resolved is not None
                    else {
                        "clip_model": args.clip_model,
                        "clip_pretrained": args.clip_pretrained,
                    }
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nfinal FID / IS / CLIPScore")
    metrics = compute_fid_is_clipscore(best_dir, imagenet_val, device=device)
    print(
        f"   FID: {metrics['fid']:.2f}, IS: {metrics['is']:.4f} ± {metrics['is_std']:.4f}, "
        f"CLIPScore: {metrics['clipscore']:.4f}"
    )
    (best_dir / "final_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
