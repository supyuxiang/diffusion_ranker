#!/usr/bin/env python3
r"""
DDPM（无条件，默认 ``google/ddpm-celebahq-256``）版 **FID reward random search**，
流程对齐 ``dit_random_search_fid_reward.py``：init1024 统计 + 多 candidate 池切片比 FID reward。

**目录布局**（与 DiT 版 ``rollout.py`` 的 candidate 结构一致，仅根目录名与 PNG 命名不同）::

  先为每个 candidate（及 init）用 ``utils.rollout_ddpm_with_steps`` 各写满 ``num_samples`` 张图；
  PNG 文件名为 ``{global_idx:05d}.png``（与 ``rollout_ddpm_with_steps`` 一致）。

  /root/autodl-tmp/ddpm_candidates_50k_deno{steps}/
    init1024/00000.png …
    candidate_0000/00000.png …
    candidate_0001/…
    …

**参考集**：默认 ``--real_dir`` 指向 CelebA-HQ/align 子集（与 ``unet_scaling_with_different_steps.py`` 一致），
**不是** ImageNet。

单跑示例::

  python3 /root/diffusion_ranker/search/ddpm_random_search_fid_reward.py \
      --num_samples 50000 \
      --num_candidates 16 \
      --deno_steps 10 \
      --real_dir /root/autodl-tmp/img_align_celeba_fid_ref_50k \
      --output_dir /root/autodl-tmp/ddpm_random_search_fid_deno10

可选 CLIP（单句 prompt，与无条件人脸一致）::

  ... --compute_clipscore --clip_prompt "a photo of a human face"
"""

from __future__ import annotations

N_INIT = 1024

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_SEARCH = Path(__file__).resolve().parent
# 先 search（reward），再项目根（utils）
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SEARCH))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import (
    INCEPTION_WEIGHTS_PATH,
    compute_clipscore,
    compute_fid_is_from_dirs,
    get_inception_feature_extractor,
    set_device,
    set_seed,
    _load_images_from_dir,
)

from reward import (
    extract_features_from_images,
    fid_features_to_statistics,
    fid_reward,
    merge_fid_statistics,
)


def _sorted_rollout_pngs(d: Path) -> list[Path]:
    """``rollout_ddpm_with_steps`` 写出 ``00000.png``；也兼容 ``image_*_*.png`` 取第二段序号。"""
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


def compute_ref_statistics(real_dir: Path, cache_name: str) -> dict:
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
    return fid_input_id_to_statistics_cached(2, feat_extractor, "2048", **kwargs)


def init_stats_loader(
    feat_extractor,
    init_dir: Path,
    device: torch.device,
) -> tuple[dict, int]:
    init_path_ls = _sorted_rollout_pngs(init_dir)
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
    num_candidates: int,
    device: torch.device,
    candidate_pathes_dict: dict[int, list[Path]],
    init_stats: dict,
    *,
    n_init: int,
    search_batch_size: int,
) -> tuple[dict, list[Path]]:
    n_expected = num_samples - n_init

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_stats, n_selected = init_stats, n_init
    best_path_ls: list[Path] = []

    start_idx = 0
    num_loops = (n_expected + search_batch_size - 1) // search_batch_size

    for _ in tqdm(range(num_loops), desc="random search fid (ddpm)"):
        n_this_batch = min(search_batch_size, n_expected - start_idx)
        if n_this_batch <= 0:
            break

        best_reward = -float("inf")
        best_candidate_idx = -1
        best_images: list[Image.Image] | None = None
        best_stats: dict | None = None

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_candidate_images = [
                Image.open(path).convert("RGB").copy()
                for path in candidate_path_ls[start_idx : start_idx + n_this_batch]
            ]

            batch_stats = fid_features_to_statistics(
                extract_features_from_images(
                    batch_candidate_images, feat_extractor, device, min(512, n_this_batch)
                )
            )

            reward_this_candidate_batch = fid_reward(
                batch_stats,
                len(batch_candidate_images),
                selected_stats,
                n_selected,
                ref_stats,
            )

            if reward_this_candidate_batch > best_reward:
                best_reward = reward_this_candidate_batch
                best_candidate_idx = candidate_idx
                best_images = batch_candidate_images
                best_stats = batch_stats

            del batch_candidate_images
            gc.collect()

        assert best_images is not None and best_stats is not None
        assert best_candidate_idx >= 0

        win_paths_this_loop = candidate_pathes_dict[best_candidate_idx][
            start_idx : start_idx + n_this_batch
        ]
        best_path_ls.extend(win_paths_this_loop)

        selected_stats = merge_fid_statistics(
            selected_stats, n_selected, best_stats, len(best_images)
        )
        n_selected += len(best_images)
        start_idx += n_this_batch
    return selected_stats, best_path_ls


def save_best_images(best_path_ls: list[Path], output_dir: Path) -> None:
    """统一写成 ``image_{i:05d}_00000.png``，便于目录排序与可选 CLIP 单 prompt。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(best_path_ls):
        dest = output_dir / f"image_{i:05d}_00000.png"
        shutil.copy2(path, dest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDPM FID-reward random search (CelebA-HQ style)")
    p.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/ddpm_random_search_fid_reward",
    )
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument(
        "--real_dir",
        type=Path,
        default=Path("/root/autodl-tmp/img_align_celeba_fid_ref_50k"),
        help="FID 参考图像目录（默认 CelebA align 5 万子集）",
    )
    p.add_argument(
        "--fid_ref_cache_name",
        type=str,
        default="celeba_align_fid_ref",
        help="torch_fidelity 缓存名，勿与 ImageNet val 混用",
    )
    p.add_argument(
        "--candidates_root",
        type=Path,
        default=None,
        help="覆盖默认 rollout 根路径；默认按 deno_steps 拼 ddpm_candidates_50k_deno{T}",
    )
    p.add_argument("--n_init", type=int, default=N_INIT, help="init 目录期望张数，与 selected_stats 对齐")
    p.add_argument(
        "--search_batch_size",
        type=int,
        default=1024,
        help="每轮 FID 切片长度（与 DiT 版 paper 设定一致）",
    )
    p.add_argument("--fid_real_max_samples", type=int, default=50_000)
    p.add_argument("--fid_real_subset_seed", type=int, default=0)
    p.add_argument(
        "--compute_clipscore",
        action="store_true",
        help="对 best 目录算 CLIPScore（无条件模型用固定 prompt）",
    )
    p.add_argument(
        "--clip_prompt",
        type=str,
        default="a photo of a human face",
    )
    p.add_argument("--clip_model_dir", type=Path, default=None)
    p.add_argument("--clip_batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    real_dir = Path(args.real_dir).expanduser().resolve()
    if not real_dir.is_dir():
        raise SystemExit(f"real_dir 不存在或不是目录: {real_dir}")

    rollout_root = (
        Path(args.candidates_root).expanduser().resolve()
        if args.candidates_root is not None
        else Path(f"/root/autodl-tmp/ddpm_candidates_50k_deno{args.deno_steps}")
    )

    candidate_dir_ls = [rollout_root / f"candidate_{i:04d}" for i in range(args.num_candidates)]
    init_dir = rollout_root / "init1024"

    set_seed(args.seed)
    device = set_device()

    print(f"ref statistics ({real_dir}) …")
    ref_stats = compute_ref_statistics(real_dir, args.fid_ref_cache_name)
    print(f"   μ {ref_stats['mu'].shape}, Σ {ref_stats['sigma'].shape}")

    feat_extractor = get_inception_feature_extractor(
        "cuda" if device.type == "cuda" else "cpu"
    )

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        candidate_dir = candidate_dir_ls[i]
        if not candidate_dir.is_dir():
            raise SystemExit(f"缺少 candidate 目录: {candidate_dir}")
        candidate_pathes_dict[i] = _sorted_rollout_pngs(candidate_dir)

    if not init_dir.is_dir():
        raise SystemExit(f"缺少 init 目录: {init_dir}")

    init_stats, n_init_loaded = init_stats_loader(feat_extractor, init_dir, device)
    if n_init_loaded != args.n_init:
        print(
            f"warning: init 目录实际 {n_init_loaded} 张，期望 n_init={args.n_init}",
            file=sys.stderr,
        )

    selected_stats, best_path_ls = run_random_search_fid(
        feat_extractor,
        ref_stats,
        output_dir,
        args.num_samples,
        args.num_candidates,
        device,
        candidate_pathes_dict,
        init_stats,
        n_init=n_init_loaded,
        search_batch_size=args.search_batch_size,
    )
    gc.collect()
    torch.cuda.empty_cache()

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "real_dir": str(real_dir),
                "candidates_root": str(rollout_root),
                "init_dir": str(init_dir),
                "num_samples": args.num_samples,
                "num_candidates": args.num_candidates,
                "n_init": args.n_init,
                "n_init_loaded": n_init_loaded,
                "deno_steps": args.deno_steps,
                "search_batch_size": args.search_batch_size,
                "seed": args.seed,
                "fid_ref_cache_name": args.fid_ref_cache_name,
                "compute_clipscore": args.compute_clipscore,
                "clip_prompt": args.clip_prompt,
                "clip_model_dir": str(args.clip_model_dir) if args.clip_model_dir else None,
                "clip_batch_size": args.clip_batch_size,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    best_dir = output_dir / "best_images"
    save_best_images(best_path_ls, best_dir)

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

    if args.compute_clipscore:
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
        if device.type == "cuda":
            torch.cuda.empty_cache()

    clip_msg = (
        f", CLIPScore {metrics['CLIPScore']:.4f}" if args.compute_clipscore else ""
    )
    print(
        f"FID {metrics['FID']:.2f}, IS {metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}{clip_msg}"
    )
    (best_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
