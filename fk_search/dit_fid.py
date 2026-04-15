#!/usr/bin/env python3
r"""
与 ``search/dit_random_search_fid_reward.py`` 相同 **FID reward + init1024 + 大 batch** 设定，
将每步 **argmax** 换为 **FK steering**（累积 log-belief + 可选重采样）。

``selected_stats`` 仍按**每步实际选中的 candidate** 的 batch 做 ``merge_fid_statistics``，
因此与 random search 一样，FK 会改变 FID 轨迹。

example::

    python3 fk_search/dit_fid.py \\
        --output_dir /root/autodl-tmp/fk_search_fid \\
        --num_samples 50000 --num_candidates 16 --deno_steps 10 \\
        --guidance_scale 1.0 --tau 1.0 --fk_resample
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_FK = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "search"))
sys.path.insert(0, str(_FK))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fk_utils import fk_update_state, pick_candidate, save_best_images_fid_layout
from reward import (
    extract_features_from_images,
    fid_features_to_statistics,
    fid_reward,
    merge_fid_statistics,
)
from utils import (
    INCEPTION_WEIGHTS_PATH,
    compute_clipscore,
    compute_fid_is_from_dirs,
    get_inception_feature_extractor,
    imagenet_label2text_for_clipscore,
    set_device,
    set_seed,
    _load_images_from_dir,
)

N_INIT = 1024
_SEARCH_BATCH = 1024


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
    return fid_input_id_to_statistics_cached(2, feat_extractor, "2048", **kwargs)


def init_stats_loader(
    feat_extractor,
    init_dir: Path,
    device: torch.device,
) -> tuple[dict, int]:
    init_path_ls = sorted(
        init_dir.glob("image_*_*.png"), key=lambda p: int(p.stem.split("_")[1])
    )
    init_image_ls = [Image.open(path).convert("RGB").copy() for path in init_path_ls]
    init_stats = fid_features_to_statistics(
        extract_features_from_images(init_image_ls, feat_extractor, device)
    )
    return init_stats, len(init_image_ls)


@torch.inference_mode()
def run_fk_steering_fid(
    feat_extractor,
    ref_stats: dict,
    num_samples: int,
    num_candidates: int,
    device: torch.device,
    candidate_pathes_dict: dict[int, list[Path]],
    init_stats: dict,
    *,
    tau: float,
    fk_resample: bool,
    pick: str,
) -> tuple[list[Path], list[dict]]:
    n_init = N_INIT
    n_expected = num_samples - n_init
    selected_stats, n_selected = init_stats, n_init
    start_idx = 0
    num_loops = (n_expected + _SEARCH_BATCH - 1) // _SEARCH_BATCH
    k_cand = len(candidate_pathes_dict)
    log_belief = torch.zeros(k_cand, dtype=torch.float64)

    best_path_ls: list[Path] = []
    trace: list[dict] = []

    for loop_i in tqdm(range(num_loops), desc="fk steer fid"):
        n_this_batch = min(_SEARCH_BATCH, n_expected - start_idx)
        if n_this_batch <= 0:
            break

        scores = torch.empty(k_cand, dtype=torch.float64)
        batch_stats_per_k: list[dict | None] = [None] * k_cand
        batch_images_per_k: list[list[Image.Image] | None] = [None] * k_cand

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_candidate_images = [
                Image.open(path).convert("RGB").copy()
                for path in candidate_path_ls[start_idx : start_idx + n_this_batch]
            ]
            batch_stats = fid_features_to_statistics(
                extract_features_from_images(
                    batch_candidate_images,
                    feat_extractor,
                    device,
                    min(512, n_this_batch),
                )
            )
            r = float(
                fid_reward(
                    batch_stats,
                    len(batch_candidate_images),
                    selected_stats,
                    n_selected,
                    ref_stats,
                )
            )
            scores[candidate_idx] = r
            batch_stats_per_k[candidate_idx] = batch_stats
            batch_images_per_k[candidate_idx] = batch_candidate_images

        log_belief, w = fk_update_state(
            log_belief, scores, tau=tau, do_resample=fk_resample
        )
        j = pick_candidate(w, pick=pick)

        best_stats = batch_stats_per_k[j]  # type: ignore[index]
        best_images = batch_images_per_k[j]  # type: ignore[index]
        win_paths = candidate_pathes_dict[j][start_idx : start_idx + n_this_batch]
        best_path_ls.extend(win_paths)

        selected_stats = merge_fid_statistics(
            selected_stats, n_selected, best_stats, len(best_images)
        )
        n_selected += len(best_images)

        trace.append(
            {
                "loop": loop_i,
                "start_idx": start_idx,
                "n_batch": n_this_batch,
                "scores": {int(i): float(scores[i]) for i in range(k_cand)},
                "weights": [float(x) for x in w.tolist()],
                "picked_candidate": j,
            }
        )
        start_idx += n_this_batch

        del batch_images_per_k
        gc.collect()

    return best_path_ls, trace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FK steering + FID reward (precomputed suite)")
    p.add_argument("--output_dir", default="/root/autodl-tmp/fk_search_fid_reward")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument("--clip_model_dir", type=Path, default=None)
    p.add_argument("--clip_batch_size", type=int, default=64)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--fk_resample", action="store_true")
    p.add_argument("--pick", type=str, default="sample", choices=("sample", "argmax"))
    p.add_argument("--rollout_root", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path("/root/autodl-tmp/imagenet/val")
    cache_name = "imagenet_val"

    if args.rollout_root.strip():
        rollout_root = Path(args.rollout_root).expanduser().resolve()
    else:
        rollout_root = Path(
            f"/root/autodl-tmp/dit_candidates_50k_cfg{args.guidance_scale}_deno{args.deno_steps}"
        )

    candidate_dir_ls = [
        rollout_root / f"candidate_{i:04d}" for i in range(args.num_candidates)
    ]
    init_dir = rollout_root / "init1024"

    set_seed(args.seed)
    device = set_device()

    print("ref statistics …")
    ref_stats = compute_ref_statistics()
    feat_extractor = get_inception_feature_extractor(
        "cuda" if device.type == "cuda" else "cpu"
    )

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        candidate_pathes_dict[i] = sorted(
            candidate_dir_ls[i].glob("image_*_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

    init_stats, _ = init_stats_loader(feat_extractor, init_dir, device)

    best_path_ls, trace = run_fk_steering_fid(
        feat_extractor,
        ref_stats,
        args.num_samples,
        args.num_candidates,
        device,
        candidate_pathes_dict,
        init_stats,
        tau=args.tau,
        fk_resample=args.fk_resample,
        pick=args.pick,
    )
    gc.collect()
    torch.cuda.empty_cache()

    best_dir = output_dir / "best_images"
    save_best_images_fid_layout(best_path_ls, best_dir)

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "verifier": "fid_fk_steering",
                "imagenet_val": str(imagenet_val),
                "candidates_root": str(rollout_root),
                "init_dir": str(init_dir),
                "num_samples": args.num_samples,
                "num_candidates": args.num_candidates,
                "n_init": N_INIT,
                "search_batch_size": _SEARCH_BATCH,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
                "tau": args.tau,
                "fk_resample": args.fk_resample,
                "pick": args.pick,
                "ref_cache_name": cache_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "fk_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dev_str = "cuda" if device.type == "cuda" else "cpu"
    metrics: dict[str, float] = {
        k: float(v)
        for k, v in compute_fid_is_from_dirs(
            best_dir,
            imagenet_val,
            dev_str,
            batch_size=args.metrics_batch_size,
        ).items()
    }
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(
        p for p in best_dir.iterdir() if p.suffix.lower() in exts and p.is_file()
    )
    labels_ls = [int(p.stem.split("_")[-1]) for p in paths]
    gen_images = _load_images_from_dir(best_dir)
    prompts = imagenet_label2text_for_clipscore(labels_ls)
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

    print(
        f"FID {metrics['FID']:.2f}, IS {metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}, "
        f"CLIPScore {metrics['CLIPScore']:.4f}"
    )
    (best_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
