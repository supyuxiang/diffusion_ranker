#!/usr/bin/env python3
r"""
DDPM（无条件人脸）版 **ImageReward random search**，流程对齐 ``dit_random_search_imagereward.py``。

DiT 版用 ``image_*_{imagenet_label}.png``，ImageReward 的文本来自 ImageNet 类名；本脚本对每张图使用
**固定 ``--imagereward_prompt``**（默认 ``a photo of a human face``）。

候选目录与 ``ddpm_random_search_clipscore.py`` / ``rollout_ddpm_with_steps`` 一致::

  /root/autodl-tmp/ddpm_candidates_50k_deno{steps}/candidate_XXXX/{idx:05d}.png

示例::

  python3 /root/diffusion_ranker/search/ddpm_random_search_imagereward.py \
      --num_samples 50000 \
      --num_candidates 16 \
      --deno_steps 10 \
      --search_batch_size 32 \
      --real_dir /root/autodl-tmp/img_align_celeba_fid_ref_50k \
      --output_dir /root/autodl-tmp/ddpm_random_search_imagereward_deno10
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
from tqdm import tqdm

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
from reward import imagereward_reward_fixed_prompt, load_imagereward


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


@torch.inference_mode()
def run_random_search_imagereward_ddpm(
    reward_model,
    prompt: str,
    num_samples: int,
    search_batch_size: int,
    candidate_pathes_dict: dict[int, list[Path]],
) -> list[Path]:
    best_path_ls: list[Path] = []
    start_idx = 0
    num_loops = (num_samples + search_batch_size - 1) // search_batch_size

    for _ in tqdm(range(num_loops), desc="random search imagereward (ddpm)"):
        n_this_batch = min(search_batch_size, num_samples - start_idx)
        if n_this_batch <= 0:
            break

        best_reward = float("-inf")
        best_candidate_idx = -1

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_candidate_paths = candidate_path_ls[start_idx : start_idx + n_this_batch]
            reward = imagereward_reward_fixed_prompt(
                batch_candidate_paths, reward_model, prompt
            )
            if reward > best_reward:
                best_reward = reward
                best_candidate_idx = candidate_idx

        assert best_candidate_idx != -1
        best_paths_this_loop = candidate_pathes_dict[best_candidate_idx][
            start_idx : start_idx + n_this_batch
        ]
        best_path_ls.extend(best_paths_this_loop)
        start_idx += n_this_batch

    return best_path_ls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDPM ImageReward random search (fixed text prompt)")
    p.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/ddpm_random_search_imagereward",
    )
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--search_batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--candidates_root",
        type=Path,
        default=None,
        help="覆盖默认 ddpm_candidates_50k_deno{T}",
    )
    p.add_argument(
        "--imagereward_prompt",
        type=str,
        default="a photo of a human face",
        help="传给 ImageReward.score(text, image) 的固定句",
    )
    p.add_argument(
        "--real_dir",
        type=Path,
        default=Path("/root/autodl-tmp/img_align_celeba_fid_ref_50k"),
        help="FID/IS 参考图目录",
    )
    p.add_argument(
        "--metrics_clip_prompt",
        type=str,
        default="a photo of a human face",
        help="写 final_metrics 时 CLIPScore 用的句（可与 imagereward_prompt 不同）",
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
    reward_model = load_imagereward(device)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        d = candidate_dir_ls[i]
        if not d.is_dir():
            raise SystemExit(f"缺少 candidate 目录: {d}")
        candidate_pathes_dict[i] = _sorted_rollout_pngs(d)

    best_path_ls = run_random_search_imagereward_ddpm(
        reward_model,
        args.imagereward_prompt,
        args.num_samples,
        args.search_batch_size,
        candidate_pathes_dict,
    )

    best_dir = output_dir / "best_images"
    save_best_images(best_path_ls, best_dir)

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "verifier": "imagereward_fixed_prompt",
                "imagereward_prompt": args.imagereward_prompt,
                "metrics_clip_prompt": args.metrics_clip_prompt,
                "candidates_root": str(rollout_root),
                "num_samples": args.num_samples,
                "search_batch_size": args.search_batch_size,
                "num_candidates": args.num_candidates,
                "deno_steps": args.deno_steps,
                "seed": args.seed,
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
    prompts = [args.metrics_clip_prompt] * n
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
