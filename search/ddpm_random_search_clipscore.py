#!/usr/bin/env python3
r"""
DDPM（无条件人脸，默认与 ``google/ddpm-celebahq-256`` rollout 布局一致）版 **CLIP reward random search**，
流程对齐 ``dit_random_search_clipscore.py``：多 candidate 池按块比 CLIP 图文相似度（均值，越大越好）。

DiT 版用 ``image_*_{imagenet_label}.png`` + ImageNet 类名；本脚本用 **固定 prompt**（默认
``a photo of a human face``），候选文件名为 ``rollout_ddpm_with_steps`` 的 ``{idx:05d}.png``。

目录约定（与 ``ddpm_random_search_fid_reward.py`` 一致）::

  /root/autodl-tmp/ddpm_candidates_50k_deno{steps}/
    candidate_0000/00000.png …
    candidate_0001/…

示例::

  python3 /root/diffusion_ranker/search/ddpm_random_search_clipscore.py \
    --output_dir /root/autodl-tmp/ddpm_random_search_clipscore_deno10 \
    --num_samples 50000 \
    --deno_steps 10 \
    --num_candidates 16 \
    --search_batch_size 32 \
    --real_dir /root/autodl-tmp/img_align_celeba_fid_ref_50k
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
from transformers import CLIPModel, CLIPProcessor

_ROOT = Path(__file__).resolve().parent.parent
_SEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SEARCH))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import (
    compute_clipscore,
    compute_fid_is_from_dirs,
    set_device,
    set_seed,
    _load_images_from_dir,
)
from reward import load_clip


def _sorted_rollout_pngs(d: Path) -> list[Path]:
    paths = [p for p in d.glob("*.png") if p.is_file()]

    def sort_key(p: Path) -> int:
        stem = p.stem
        if stem.isdigit():
            return int(stem)
        parts = stem.split("_")
        for tok in parts:
            if tok.isdigit():
                return int(tok)
        raise ValueError(f"无法从文件名解析序号: {p.name}")

    return sorted(paths, key=sort_key)


def save_best_images(best_path_ls: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in best_path_ls:
        shutil.copy2(path, output_dir / path.name)
    print(f"saved {len(best_path_ls)} best images to {output_dir}")


@torch.inference_mode()
def clipscore_reward_fixed_prompt(
    batch_paths: list[Path],
    clip: CLIPModel,
    processor: CLIPProcessor,
    prompt: str,
) -> float:
    """与 ``reward.clipscore_reward`` 相同计算，但全文用同一 ``prompt``。"""
    batch_images = [Image.open(path).convert("RGB").copy() for path in batch_paths]
    batch_prompts = [prompt] * len(batch_images)
    inputs = processor(
        text=batch_prompts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(clip.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = clip(**inputs)
        image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_features = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (image_features * text_features).sum(dim=-1)
    return float(sim.mean().item())


@torch.inference_mode()
def run_random_search_clipscore_ddpm(
    clip: CLIPModel,
    processor: CLIPProcessor,
    prompt: str,
    num_samples: int,
    search_batch_size: int,
    candidate_pathes_dict: dict[int, list[Path]],
) -> list[Path]:
    n_expected = num_samples
    start_idx = 0
    num_loops = (n_expected + search_batch_size - 1) // search_batch_size
    best_path_ls: list[Path] = []

    for _ in tqdm(range(num_loops), desc="random search clipscore (ddpm)"):
        n_this_batch = min(search_batch_size, n_expected - start_idx)
        if n_this_batch <= 0:
            break

        best_reward = float("-inf")
        best_candidate_idx = -1

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_candidate_paths = candidate_path_ls[start_idx : start_idx + n_this_batch]
            reward = clipscore_reward_fixed_prompt(
                batch_candidate_paths, clip, processor, prompt
            )
            if reward > best_reward:
                best_reward = reward
                best_candidate_idx = candidate_idx

        assert best_candidate_idx != -1
        win_paths_this_loop = candidate_pathes_dict[best_candidate_idx][
            start_idx : start_idx + n_this_batch
        ]
        best_path_ls.extend(win_paths_this_loop)
        start_idx += n_this_batch

    return best_path_ls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDPM CLIP-score random search (fixed text prompt)")
    p.add_argument("--output_dir", default="/root/autodl-tmp/ddpm_random_search_clipscore")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument(
        "--search_batch_size",
        type=int,
        default=32,
        help="每轮每个 candidate 池切片长度",
    )
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10, help="与 rollout 根目录名一致")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--candidates_root",
        type=Path,
        default=None,
        help="覆盖默认 ddpm_candidates_50k_deno{T}",
    )
    p.add_argument(
        "--clip_prompt",
        type=str,
        default="a photo of a human face",
        help="无条件生成：search 与最终 CLIPScore 均用该句",
    )
    p.add_argument(
        "--real_dir",
        type=Path,
        default=Path("/root/autodl-tmp/img_align_celeba_fid_ref_50k"),
        help="FID/IS 参考图目录",
    )
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument("--fid_real_max_samples", type=int, default=50_000)
    p.add_argument("--fid_real_subset_seed", type=int, default=0)
    p.add_argument("--clip_model_dir", type=Path, default=None)
    p.add_argument("--clip_batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    real_dir = Path(args.real_dir).expanduser().resolve()
    if not real_dir.is_dir():
        raise SystemExit(f"real_dir 不存在: {real_dir}")

    rollout_root = (
        args.candidates_root.expanduser().resolve()
        if args.candidates_root is not None
        else Path(f"/root/autodl-tmp/ddpm_candidates_50k_deno{args.deno_steps}")
    )
    candidate_dir_ls = [rollout_root / f"candidate_{i:04d}" for i in range(args.num_candidates)]

    set_seed(args.seed)
    device = set_device()
    clip, processor = load_clip(device)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        d = candidate_dir_ls[i]
        if not d.is_dir():
            raise SystemExit(f"缺少 candidate 目录: {d}")
        candidate_pathes_dict[i] = _sorted_rollout_pngs(d)

    best_path_ls = run_random_search_clipscore_ddpm(
        clip,
        processor,
        args.clip_prompt,
        args.num_samples,
        args.search_batch_size,
        candidate_pathes_dict,
    )

    best_dir = output_dir / "best_images"
    best_dir.mkdir(parents=True, exist_ok=True)
    save_best_images(best_path_ls, best_dir)

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "verifier": "clipscore_fixed_prompt",
                "clip_prompt": args.clip_prompt,
                "num_samples": args.num_samples,
                "search_batch_size": args.search_batch_size,
                "num_candidates": args.num_candidates,
                "deno_steps": args.deno_steps,
                "candidates_root": str(rollout_root),
                "real_dir": str(real_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dev_str = "cuda" if device.type == "cuda" else "cpu"
    metrics: dict[str, float] = {
        k: float(v)
        for k, v in compute_fid_is_from_dirs(
            best_dir,
            real_dir,
            dev_str,
            batch_size=args.metrics_batch_size,
            real_max_samples=args.fid_real_max_samples,
            real_subset_seed=args.fid_real_subset_seed,
        ).items()
    }

    gen_images = _load_images_from_dir(best_dir)
    n = gen_images.shape[0]
    prompts = [args.clip_prompt] * n
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics["CLIPScore"] = float(
        compute_clipscore(
            gen_images,
            prompts,
            device=device,
            clip_model_dir=args.clip_model_dir,
            clip_batch_size=args.clip_batch_size,
        )
    )
    del gen_images, prompts
    gc.collect()

    (best_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(metrics)


if __name__ == "__main__":
    main()
