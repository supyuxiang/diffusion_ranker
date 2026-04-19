#!/usr/bin/env python3
"""
example:

python3 /root/diffusion_ranker/search/dit_random_search_dino.py \
    --output_dir /root/autodl-tmp/dit_random_search_dino_ps8_cfg1.0_deno10 \
    --num_samples 50000 \
    --search_batch_size 32 \
    --num_candidates 8 \
    --deno_steps 10 \
    --guidance_scale 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import compute_fid_is_clipscore, set_device, set_seed
from class_condition.utils import _sorted_candidate_images, _reset_best_dir, save_best_images


def _iter_image_paths(img_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _load_dino(device: torch.device, dino_model_dir: Path) -> tuple[AutoModel, AutoImageProcessor]:
    model = AutoModel.from_pretrained(str(dino_model_dir), local_files_only=True).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(str(dino_model_dir), local_files_only=True)
    return model, processor


@torch.inference_mode()
def _dino_features(
    images: list[Image.Image],
    dino: AutoModel,
    processor: AutoImageProcessor,
) -> torch.Tensor:
    inputs = processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
    inputs = {k: v.to(dino.device) for k, v in inputs.items()}
    out = dino(**inputs)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output.float()
    else:
        feat = out.last_hidden_state[:, 0, :].float()
    return F.normalize(feat, dim=-1)


@torch.inference_mode()
def _build_dino_prototype(
    real_dir: Path,
    dino: AutoModel,
    processor: AutoImageProcessor,
    *,
    dino_batch_size: int,
    max_ref_images: int,
) -> torch.Tensor:
    paths = _iter_image_paths(real_dir)
    if max_ref_images > 0:
        paths = paths[:max_ref_images]
    if not paths:
        raise FileNotFoundError(f"no images found under {real_dir}")

    feats_sum = None
    n = 0
    for i in tqdm(range(0, len(paths), dino_batch_size), desc="build dino prototype"):
        chunk = paths[i : i + dino_batch_size]
        pil_ls: list[Image.Image] = []
        for p in chunk:
            try:
                pil_ls.append(Image.open(p).convert("RGB").copy())
            except Exception:
                continue
        if not pil_ls:
            continue
        feat = _dino_features(pil_ls, dino, processor)
        cur = feat.sum(dim=0)
        feats_sum = cur if feats_sum is None else feats_sum + cur
        n += feat.shape[0]
    if feats_sum is None or n == 0:
        raise RuntimeError(f"failed to build dino prototype from {real_dir}")
    return F.normalize(feats_sum / n, dim=-1)


@torch.inference_mode()
def _batch_dino_reward(
    paths: list[Path],
    dino: AutoModel,
    processor: AutoImageProcessor,
    prototype: torch.Tensor,
) -> float:
    images = [Image.open(p).convert("RGB").copy() for p in paths]
    feat = _dino_features(images, dino, processor)
    sim = (feat * prototype.unsqueeze(0)).sum(dim=-1)
    return float(sim.mean().item())


@torch.inference_mode()
def run_random_search_dino(
    dino: AutoModel,
    processor: AutoImageProcessor,
    prototype: torch.Tensor,
    num_samples: int,
    search_batch_size: int,
    candidate_pathes_dict: dict[int, list[Path]],
) -> tuple[list[Path], list[dict[str, float]]]:
    best_path_ls: list[Path] = []
    round_logs: list[dict[str, float]] = []
    start_idx = 0
    num_loops = (num_samples + search_batch_size - 1) // search_batch_size

    for loop_idx in tqdm(range(num_loops), desc="random search dino"):
        n_this_batch = min(search_batch_size, num_samples - start_idx)
        if n_this_batch <= 0:
            break

        best_reward = float("-inf")
        best_candidate_idx = -1

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_candidate_paths = candidate_path_ls[start_idx : start_idx + n_this_batch]
            reward = _batch_dino_reward(batch_candidate_paths, dino, processor, prototype)
            if reward > best_reward:
                best_reward = reward
                best_candidate_idx = candidate_idx

        assert best_candidate_idx != -1
        best_paths_this_loop = candidate_pathes_dict[best_candidate_idx][
            start_idx : start_idx + n_this_batch
        ]
        best_path_ls.extend(best_paths_this_loop)
        round_logs.append(
            {
                "loop_idx": float(loop_idx),
                "start_idx": float(start_idx),
                "batch_size": float(n_this_batch),
                "winner_candidate_idx": float(best_candidate_idx),
                "winner_reward": float(best_reward),
            }
        )
        start_idx += n_this_batch

    return best_path_ls, round_logs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/root/autodl-tmp/dit_random_search_dino")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--search_batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--real_dir", type=Path, default=Path("/root/autodl-tmp/imagenet/val"))
    p.add_argument("--dino_model_dir", type=Path, default=Path("/root/autodl-tmp/facebook_dinov2-base"))
    p.add_argument("--dino_batch_size", type=int, default=64)
    p.add_argument("--max_ref_images", type=int, default=4096)
    p.add_argument("--clip_model_dir", type=Path, default=Path("/root/autodl-tmp/openai_clip-vit-large-patch14"))
    p.add_argument("--clip_batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    best_dir = output_dir / "best_images"
    rollout_root = Path(
        f"/root/autodl-tmp/dit_candidates_50k_cfg{args.guidance_scale}_deno{args.deno_steps}"
    )
    candidate_dir_ls = [rollout_root / f"candidate_{i:04d}" for i in range(args.num_candidates)]

    set_seed(args.seed)
    device = set_device()
    dino, processor = _load_dino(device, args.dino_model_dir.expanduser().resolve())
    prototype = _build_dino_prototype(
        args.real_dir.expanduser().resolve(),
        dino,
        processor,
        dino_batch_size=args.dino_batch_size,
        max_ref_images=args.max_ref_images,
    )

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i, candidate_dir in enumerate(candidate_dir_ls):
        if not candidate_dir.is_dir():
            raise FileNotFoundError(f"missing candidate dir: {candidate_dir}")
        paths = _sorted_candidate_images(candidate_dir)
        if len(paths) < args.num_samples:
            raise ValueError(f"{candidate_dir} has only {len(paths)} images, expected >= {args.num_samples}")
        candidate_pathes_dict[i] = paths

    best_path_ls, round_logs = run_random_search_dino(
        dino,
        processor,
        prototype,
        args.num_samples,
        args.search_batch_size,
        candidate_pathes_dict,
    )
    _copy_selected(best_path_ls, best_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "verifier": "dino_prototype_cosine",
        "candidates_root": str(rollout_root),
        "real_dir": str(args.real_dir.expanduser().resolve()),
        "num_samples": args.num_samples,
        "search_batch_size": args.search_batch_size,
        "num_candidates": args.num_candidates,
        "deno_steps": args.deno_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "dino_model_dir": str(args.dino_model_dir.expanduser().resolve()),
        "dino_batch_size": args.dino_batch_size,
        "max_ref_images": args.max_ref_images,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "round_log.json").write_text(
        json.dumps(round_logs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics = compute_fid_is_clipscore(
        best_dir,
        args.real_dir.expanduser().resolve(),
        device=device,
        clip_model_dir=args.clip_model_dir.expanduser().resolve(),
        clip_batch_size=args.clip_batch_size,
    )
    metrics["selection_reward"] = "dino_prototype_cosine"
    metrics["num_selected"] = float(len(best_path_ls))
    metrics["num_candidates"] = float(args.num_candidates)
    (best_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
