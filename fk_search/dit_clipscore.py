#!/usr/bin/env python3
r"""
在 **多候选预生成图** 上做 **FK（Feynman–Kac）式 steering**， verifier 为 **CLIPScore**。

与 ``search/dit_random_search_clipscore.py`` 的数据布局相同（每个 candidate 目录一套按全局序号对齐的
``image_*_*.png``），区别是：

- **Random search**：每个 search batch **只看当前切片** 上各 candidate 的 mean CLIP，取 **argmax**。
- **FK steering**：在 candidate 维上维护 **累积 log-势** :math:`\ell_k \mathrel{+}= s_k/\tau`
  （:math:`s_k` 为本切片 mean CLIP），:math:`w=\mathrm{softmax}(\ell)`；可选 **多项式重采样**
  后用经验频率更新 :math:`w` 并把 :math:`\ell` 设为 :math:`\log w`（SMC 防退化）。最后按 :math:`w`
  **抽样或 argmax** 决定本切片写入 ``best_images`` 的 candidate（相对 random search **带跨切片记忆**）。

example::

python3 /root/diffusion_ranker/fk_search/dit_clipscore.py \
  --output_dir /root/autodl-tmp/fk_search_clipscore \
  --num_samples 50000 \
  --deno_steps 10 \
  --guidance_scale 1.0 \
  --num_candidates 16 \
  --tau 0.1 \
  --fk_resample \
  --pick sample
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "search"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from reward import load_clip, clipscore_reward, labels_1k
from utils import compute_fid_is_clipscore, set_device, set_seed


def save_best_images(best_path_ls: list[Path], output_dir: Path) -> None:
    for path in best_path_ls:
        copy_path = output_dir / path.name
        shutil.copy2(path, copy_path)
    print(f"saved {len(best_path_ls)} best images to {output_dir}")


def _fk_update_state(
    log_belief: torch.Tensor,
    scores: torch.Tensor,
    *,
    tau: float,
    do_resample: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在 candidate 维上 **累积** log-势：``log_belief[k] += scores[k] / tau``，再 softmax 得权重。
    可选：对权重做 K 次多项式重采样，并把 ``log_belief`` 同步为 ``log(w)``（SMC 常见写法）。

    返回 ``(log_belief_next, w)``。
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    log_belief = log_belief + scores / float(tau)
    w = torch.softmax(log_belief, dim=0)
    if not do_resample:
        return log_belief, w
    k = int(w.shape[0])
    idx = torch.multinomial(w, num_samples=k, replacement=True)
    counts = torch.bincount(idx, minlength=k).float()
    w = counts / counts.sum().clamp(min=1e-12)
    log_belief = torch.log(w.clamp(min=1e-20))
    return log_belief, w


def _pick_candidate(
    w: torch.Tensor,
    *,
    pick: str,
) -> int:
    if pick == "argmax":
        return int(torch.argmax(w).item())
    if pick == "sample":
        return int(torch.multinomial(w, num_samples=1).item())
    raise ValueError(f"pick must be argmax|sample, got {pick!r}")


@torch.inference_mode()
def run_fk_steering_clipscore(
    clip: CLIPModel,
    processor: CLIPProcessor,
    labels_1k: list[str],
    num_samples: int,
    search_batch_size: int,
    candidate_pathes_dict: dict[int, list[Path]],
    *,
    tau: float,
    fk_resample: bool,
    pick: str,
) -> tuple[list[Path], list[dict]]:
    """
    返回选中的路径列表，以及每步日志（权重、选中 candidate、各 candidate 分数）。
    """
    n_expected = num_samples
    start_idx = 0
    num_loops = (n_expected + search_batch_size - 1) // search_batch_size
    k_cand = len(candidate_pathes_dict)
    log_belief = torch.zeros(k_cand, dtype=torch.float64)

    best_path_ls: list[Path] = []
    trace: list[dict] = []

    for loop_i in tqdm(range(num_loops), desc="fk steer clipscore"):
        n_this_batch = min(search_batch_size, n_expected - start_idx)
        if n_this_batch <= 0:
            break

        scores = torch.empty(k_cand, dtype=torch.float64)
        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            batch_candidate_paths = candidate_path_ls[start_idx : start_idx + n_this_batch]
            r = clipscore_reward(batch_candidate_paths, clip, processor, labels_1k)
            scores[candidate_idx] = float(r)

        log_belief, w = _fk_update_state(
            log_belief, scores, tau=tau, do_resample=fk_resample
        )
        best_idx = _pick_candidate(w, pick=pick)

        win_paths = candidate_pathes_dict[best_idx][start_idx : start_idx + n_this_batch]
        best_path_ls.extend(win_paths)
        trace.append(
            {
                "loop": loop_i,
                "start_idx": start_idx,
                "n_batch": n_this_batch,
                "scores": {int(i): float(scores[i]) for i in range(k_cand)},
                "weights": [float(x) for x in w.tolist()],
                "picked_candidate": best_idx,
            }
        )
        start_idx += n_this_batch

    return best_path_ls, trace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FK steering over precomputed DiT candidates (CLIPScore)")
    p.add_argument("--output_dir", default="/root/autodl-tmp/fk_search_clipscore")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument(
        "--search_batch_size",
        type=int,
        default=32,
        help="每个 FK 步各 candidate 池的切片长度（与 random search 脚本一致）",
    )
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="FK 更新温度；越小越接近仅信当前切片 argmax（与均匀初值组合时首步类似 greedy）",
    )
    p.add_argument(
        "--fk_resample",
        action="store_true",
        help="在权重更新后做 K 次多项式重采样，得到新经验权重（SMC 防退化）",
    )
    p.add_argument(
        "--pick",
        type=str,
        default="sample",
        choices=("sample", "argmax"),
        help="本切片写入 best_images 时：按最终权重抽样 candidate，或取 argmax",
    )
    p.add_argument(
        "--rollout_root",
        type=str,
        default="",
        help="候选图根目录；默认与 random search 相同规则的路径",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.rollout_root.strip():
        rollout_root = Path(args.rollout_root).expanduser().resolve()
    else:
        rollout_root = Path(
            f"/root/autodl-tmp/dit_candidates_50k_cfg{args.guidance_scale}_deno{args.deno_steps}"
        )

    candidate_dir_ls = [
        rollout_root / f"candidate_{i:04d}" for i in range(args.num_candidates)
    ]

    set_seed(args.seed)
    device = set_device()

    clip, processor = load_clip(device)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        candidate_pathes_dict[i] = sorted(
            candidate_dir_ls[i].glob("image_*_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

    best_path_ls, trace = run_fk_steering_clipscore(
        clip,
        processor,
        labels_1k,
        args.num_samples,
        args.search_batch_size,
        candidate_pathes_dict,
        tau=args.tau,
        fk_resample=args.fk_resample,
        pick=args.pick,
    )

    best_dir = output_dir / "best_images"
    best_dir.mkdir(exist_ok=True, parents=True)
    save_best_images(best_path_ls, best_dir)

    cfg = {
        "verifier": "clipscore_fk_steering",
        "num_samples": args.num_samples,
        "search_batch_size": args.search_batch_size,
        "num_candidates": args.num_candidates,
        "deno_steps": args.deno_steps,
        "guidance_scale": args.guidance_scale,
        "tau": args.tau,
        "fk_resample": args.fk_resample,
        "pick": args.pick,
        "rollout_root": str(rollout_root),
    }
    (output_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "fk_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics = compute_fid_is_clipscore(best_dir, device=device)
    (best_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(metrics)


if __name__ == "__main__":
    main()
