#!/usr/bin/env python3
"""

after rollout: /root/diffusion_ranker/search/rollout.py
 -> /root/autodl-tmp/dit_candidates_50k_cfg{gs}_deno{steps}/candidate_XXXX/image_*_*.png
    and /root/autodl-tmp/dit_candidates_50k_cfg{gs}_deno{steps}/init1024/image_*_*.png

run single: 
  python3 /root/diffusion_ranker/search/dit_random_search_fid_reward.py \
      --num_samples 50000 \
      --num_candidates 16 \
      --deno_steps 10 \
      --guidance_scale 1.0

# run all:
bash /root/diffusion_ranker/scripts/run_fid_search.sh

default: init_num = 1024, num_gen = 50000, search_batch_size=1024
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
    compute_clipscore,
    compute_fid_is_from_dirs,
    fid_features_to_statistics,
    fid_statistics_to_distance,
    get_inception_feature_extractor,
    imagenet_label2text_for_clipscore,
    merge_fid_statistics,
    set_device,
    set_seed,
    _load_images_from_dir,
)

from reward import fid_reward, merge_fid_statistics, fid_statistics_to_distance, fid_features_to_statistics, extract_features_from_images




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
    return fid_input_id_to_statistics_cached(2, feat_extractor, "2048", **kwargs) # dict: {"mu": (2048,), "sigma": (2048, 2048)}
    # 已是 {"mu","sigma"}；勿对返回值再调 fid_features_to_statistics（后者输入须为 (N,2048) 特征）


def init_stats_loader(
    feat_extractor,
    init_dir:Path,
    device: torch.device,
) -> tuple[dict, int]:
    init_path_ls = sorted(init_dir.glob("image_*_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    #item example: image_01023_00467.png, image_01024_00468.png, ... list[Path], 00467 is class_id which need to be extracted
    init_image_ls = [Image.open(path).convert("RGB").copy() for path in init_path_ls]
    init_stats = fid_features_to_statistics(
        extract_features_from_images(init_image_ls, feat_extractor, device)
    ) # dict: {"mu": (2048,), "sigma": (2048, 2048)}
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
) -> tuple[dict, list[Path]]:
    n_init = 1024
    n_expected = num_samples - n_init # 50_000 - 1024 = 49876

    _search_batch_size = 1024 # paper中的设定

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_stats, n_selected = init_stats, n_init # 初始化 selected_stats 和 n_selected， 后续每个loop进行update
    best_path_ls: list[Path] = [] # 以便后续save和计算metrics
    
    current_num_samples = n_init
    num_loops = (n_expected + _search_batch_size - 1) // _search_batch_size # 总的search loop数

    for _ in tqdm(range(num_loops), desc="random search fid"):
        n_this_batch = min(_search_batch_size, num_samples - current_num_samples) # 当前loop需要生成的样本数
        if n_this_batch <= 0:
            break
        # predefine best_fid, best_batch_images, best_batch_stats
        best_reward = -float("inf")
        best_candidate_idx = -1
        best_batch_images: list[Image.Image] | None = None
        best_batch_stats: dict | None = None

        # iterate over all candidates
        for candidate_idx,candidate_path_ls in candidate_pathes_dict.items():
            batch_images = [
                Image.open(path).convert('RGB').copy()
                for path in candidate_path_ls[current_num_samples: current_num_samples + n_this_batch]
            ] # list[Image.Image] this candidate's batch images

            batch_stats = fid_features_to_statistics(
                extract_features_from_images(batch_images, feat_extractor, device, min(512, n_this_batch))
            ) # dict: {"mu": (2048,), "sigma": (2048, 2048)}

            # compute reward and select best candidate this round
            reward_this = fid_reward(
                batch_stats, len(batch_images),selected_stats, n_selected, ref_stats
            )

            if reward_this > best_reward:
                best_reward = reward_this
                best_candidate_idx = candidate_idx
                best_batch_images = batch_images
                best_batch_stats = batch_stats
            
            del batch_images
            gc.collect()

        assert best_batch_images is not None and best_batch_stats is not None and best_candidate_idx != -1

        # 必须用胜出 candidate 的路径列表；循环结束后 candidate_path_ls 只是「最后一个」candidate
        win_paths = candidate_pathes_dict[best_candidate_idx][
            current_num_samples : current_num_samples + n_this_batch
        ]
        best_path_ls.extend(win_paths)
        
        selected_stats = merge_fid_statistics(
            selected_stats, n_selected, best_batch_stats, len(best_batch_images)
        )
        n_selected += len(best_batch_images)
        current_num_samples += n_this_batch
        if current_num_samples >= num_samples:
            break
    return selected_stats, best_path_ls


def save_best_images(best_path_ls: list[Path], output_dir: Path) -> None:
    """写出与 ``rollout_dit_with_steps`` 一致：``image_{序}_{class_id}.png``，供 ``compute_fid_is_clipscore`` 用 ``stem`` 最后一段解析类别。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    for i, path in enumerate(best_path_ls):
        parts = path.stem.split("_")
        class_id = int(parts[-1])
        # 与 utils.rollout 中 ``image_{counter:05d}_{label:05d}.png`` 对齐；CLIP 读 ``parts[-1]``
        dest = output_dir / f"image_{i:05d}_{class_id:05d}.png"
        shutil.copy2(path, dest)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/root/autodl-tmp/random_search_fid_reward")
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--deno_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics_batch_size", type=int, default=64)
    parser.add_argument(
        "--clip_model_dir",
        type=Path,
        default=None,
        help="覆盖默认 utils.CLIP_HF_LOCAL_DIR",
    )
    parser.add_argument("--clip_batch_size", type=int, default=64)
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
        rollout_root / f'candidate_{i:04d}'
        for i in range(args.num_candidates) # list[Path]
    ]
    init_dir = rollout_root / "init1024" # Path
    
    # check 
    n_expect = args.num_samples - N_INIT # 50000-1024 = 49876
    # assert all(
    #     len(list(candidate_dir.glob("image_*_*.png"))) == n_expect
    #     for candidate_dir in candidate_dir_ls
    # ) and len(list(init_dir.glob("image_*_*.png"))) == N_INIT, (
    #     f"expected each candidate to have {n_expect} images and init {N_INIT}; "
    #     f"check rollout_root={rollout_root}"
    # )


    set_seed(args.seed)
    device = set_device()

    print("ref statistics …")
    ref_stats = compute_ref_statistics() # dict: {"mu": (2048,), "sigma": (2048, 2048)}
    print(f"   μ {ref_stats['mu'].shape}, Σ {ref_stats['sigma'].shape}")
    feat_extractor = get_inception_feature_extractor(
        "cuda" if device.type == "cuda" else "cpu"
    ) # built inception_feature_extractor

    # prepare candidate pathes list : list[list[Path]] : [[path1, path2, ...], [path1, path2, ...], ...]
    candidate_pathes_dict = {}
    for i in range(args.num_candidates):
        candidate_dir = candidate_dir_ls[i]
        candidate_pathes_dict[i] = sorted(
            candidate_dir.glob("image_*_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
    # prepare init pathes list : list[Path] : [path1, path2, ...]
    init_path_ls = sorted(init_dir.glob("image_*_*.png"), key=lambda p: int(p.stem.split("_")[1]))

    init_stats, n_init = init_stats_loader(
        feat_extractor, init_dir, device # dict: {"mu": (2048,), "sigma": (2048, 2048)}, int: 1024
    )
    

    #  run random search  , return selected_stats: dict: {"mu": (2048,), "sigma": (2048, 2048)}, best_path_ls: list[Path]
    selected_stats, best_path_ls = run_random_search_fid(
        feat_extractor,
        ref_stats,
        output_dir,
        args.num_samples,
        args.num_candidates,
        device,
        candidate_pathes_dict,
        init_stats,
    )
    gc.collect()
    torch.cuda.empty_cache()
    
    # save cinfig json
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "imagenet_val": str(imagenet_val),
                "candidates_root": str(rollout_root),
                "init_dir": str(init_dir),
                "num_samples": args.num_samples,
                "num_candidates": args.num_candidates,
                "n_init": N_INIT,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "ref_cache_name": cache_name,
                "clip_model_dir": str(args.clip_model_dir)
                if args.clip_model_dir is not None
                else None,
                "clip_batch_size": args.clip_batch_size,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    best_dir = output_dir / 'best_images'
    save_best_images(best_path_ls, best_dir)
    # compute metrics
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
    if gen_images.shape[0] != len(labels_ls):
        raise ValueError(
            f"best_dir image count {gen_images.shape[0]} vs labels {len(labels_ls)}"
        )
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
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
