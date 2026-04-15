#!/usr/bin/env python3
r"""
**预计算 rollout suite**（``idx{i}.pt`` + ``candidate_{i:04d}/``）上，用 **Diffusion ranker 均分**
作奖励，**FK steering** 选 candidate（对应 ``search/dit_random_search_ranker.py`` 里
``verifier=ranker`` + suite 模式，但每轮对所有 K 个 candidate 打分后走 FK，而非 argmax）。

不加载 DiT；输出 ``best_images`` 下 ``image_{counter:05d}_{label:05d}.png``，并写 FID/IS/CLIP 指标。

example::

    python3 fk_search/dit_ranker.py \\
        --output_dir /root/autodl-tmp/fk_search_ranker \\
        --suite_image_root /root/autodl-tmp/my_suite \\
        --suite_repr_dir /root/autodl-tmp/repr_candidate_deno10_cfg1.0 \\
        --ranker_ckpt /root/autodl-tmp/ranker_train/ranker_final.pt \\
        --num_candidates 16 \\
        --batch_size 32 \\
        --num_samples 50000 \\
        --n_init 1024 \\
        --tau 0.5 --fk_resample
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_FK = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "search"))
sys.path.insert(0, str(_FK))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import dit_random_search_ranker as dsr

from fk_utils import fk_update_state, pick_candidate
from utils import (
    compute_clipscore,
    compute_fid_is_from_dirs,
    imagenet_label2text_for_clipscore,
    set_seed,
    _load_images_from_dir,
)


@torch.inference_mode()
def run_fk_steering_ranker_suite(
    ranker: torch.nn.Module,
    selected_keys: list[str],
    listwise: bool,
    device: torch.device,
    pool: list[tuple[list, list[int], dict]],
    *,
    num_samples: int,
    batch_size: int,
    n_init: int,
    tau: float,
    fk_resample: bool,
    pick: str,
) -> tuple[list[tuple], list[dict]]:
    """
    返回 ``[(Image, label), ...]`` 按保存顺序（先 init 条由调用方处理时可跳过），
    这里只返回 **FK 阶段** 产生的 (img, lab) 列表；为简单起见返回可迭代保存对。
    实际上我们边循环边写盘在 main 更清晰。
    """
    k_cand = len(pool)
    b0 = len(pool[0][1])
    if any(len(pool[i][1]) != b0 for i in range(k_cand)):
        raise ValueError("预计算 pool 各 candidate batch 大小须一致")
    if b0 < batch_size:
        raise ValueError(f"pool batch {b0} < --batch_size={batch_size}")

    log_belief = torch.zeros(k_cand, dtype=torch.float64)
    trace: list[dict] = []
    saved_pairs: list[tuple] = []

    img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for loop_i in tqdm(range(num_rounds), desc="fk steer ranker"):
        n_this = min(batch_size, num_samples - img_counter)
        if n_this <= 0:
            break

        scores = torch.empty(k_cand, dtype=torch.float64)
        round_entries: list[tuple[list, list[int], dict]] = []

        for j in range(k_cand):
            imgs_f, labels_f, repr_f = pool[j]
            if n_this < len(imgs_f):
                imgs_s = imgs_f[:n_this]
                labels_s = labels_f[:n_this]
                repr_s = dsr.slice_repr_batch(repr_f, n_this)
            else:
                imgs_s, labels_s, repr_s = imgs_f, labels_f, repr_f

            dsr.verify_keys_subset(repr_s, selected_keys)
            tok = dsr.repr_dict_to_tokens(repr_s, selected_keys, len(imgs_s))
            s = float(
                dsr.mean_batch_ranker_score(
                    ranker, tok, device, listwise=listwise
                )
            )
            scores[j] = s
            round_entries.append((imgs_s, labels_s, repr_s))

        log_belief, w = fk_update_state(
            log_belief, scores, tau=tau, do_resample=fk_resample
        )
        j = pick_candidate(w, pick=pick)
        best_imgs, best_labels, _ = round_entries[j]

        for img, lab in zip(best_imgs, best_labels):
            if img_counter >= num_samples:
                break
            saved_pairs.append((img, int(lab)))
            img_counter += 1

        trace.append(
            {
                "loop": loop_i,
                "n_this": n_this,
                "scores": {int(i): float(scores[i]) for i in range(k_cand)},
                "weights": [float(x) for x in w.tolist()],
                "picked_candidate": j,
            }
        )

    return saved_pairs, trace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FK steering + ranker score (precomputed suite)")
    p.add_argument("--output_dir", default="/root/autodl-tmp/fk_search_ranker")
    p.add_argument("--suite_image_root", type=str, required=True)
    p.add_argument("--suite_repr_dir", type=str, required=True)
    p.add_argument("--ranker_ckpt", type=str, required=True)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--batch_index", type=int, default=0, help="每个 idx{i}.pt 内 batches[k]")
    p.add_argument("--batch_size", type=int, default=32, help="须与预计算 batch 大小一致")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--n_init", type=int, default=1024)
    p.add_argument("--init_candidate_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--fk_resample", action="store_true")
    p.add_argument("--pick", type=str, default="sample", choices=("sample", "argmax"))
    p.add_argument("--imagenet_val", type=str, default="/root/autodl-tmp/imagenet/val")
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument("--clip_model_dir", type=Path, default=None)
    p.add_argument("--clip_batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()
    suite_root = Path(args.suite_image_root).expanduser().resolve()
    repr_d = Path(args.suite_repr_dir).expanduser().resolve()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_p = Path(args.ranker_ckpt).expanduser().resolve()
    ranker, selected_keys, listwise, ranker_meta = dsr.load_diffusion_ranker(
        ckpt_p, device
    )

    pool, pre_meta = dsr.load_precomputed_pool_from_rollout_suite(
        repr_d,
        suite_root,
        args.num_candidates,
        batch_index=args.batch_index,
    )
    dsr.verify_precomputed_pool_ranker_keys(
        pool, selected_keys, repr_dir_hint=repr_d
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    n_init = int(args.n_init)
    if n_init > 0:
        ic = int(args.init_candidate_idx)
        cand_dir = suite_root / f"candidate_{ic:04d}"
        n_copied = dsr.copy_init_images_from_rollout_candidate(
            output_dir, cand_dir, n_init
        )
        print(f"init: copied {n_copied} images from {cand_dir}")
    else:
        n_copied = 0

    pairs, trace = run_fk_steering_ranker_suite(
        ranker,
        selected_keys,
        listwise,
        device,
        pool,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        n_init=n_copied,
        tau=args.tau,
        fk_resample=args.fk_resample,
        pick=args.pick,
    )

    img_counter = n_copied
    for img, lab in pairs:
        img.save(output_dir / f"image_{img_counter:05d}_{lab:05d}.png")
        img_counter += 1

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cfg = {
        "verifier": "ranker_fk_steering",
        "suite_image_root": str(suite_root),
        "suite_repr_dir": str(repr_d),
        "ranker_ckpt": str(ckpt_p),
        "num_candidates": args.num_candidates,
        "batch_index": args.batch_index,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "n_init": n_copied,
        "init_candidate_idx": args.init_candidate_idx,
        "tau": args.tau,
        "fk_resample": args.fk_resample,
        "pick": args.pick,
        "pre_meta": pre_meta,
        "ranker_meta": ranker_meta,
    }
    (output_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False, default=str),
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
            output_dir,
            imagenet_val,
            dev_str,
            batch_size=args.metrics_batch_size,
        ).items()
    }
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(
        p for p in output_dir.iterdir() if p.suffix.lower() in exts and p.is_file()
    )
    labels_ls = [int(p.stem.split("_")[-1]) for p in paths]
    gen_images = _load_images_from_dir(output_dir)
    prompts = imagenet_label2text_for_clipscore(labels_ls)
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
    (output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
