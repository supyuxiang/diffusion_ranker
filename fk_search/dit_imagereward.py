#!/usr/bin/env python3
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
_SEARCH = _ROOT / "search"
for p in (_ROOT, _SEARCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import dit_random_search_ranker as dsr
from reward import labels_1k, load_imagereward
from utils import compute_fid_is_from_dirs, set_seed


def _update_fk_state(
    log_belief: torch.Tensor,
    scores: torch.Tensor,
    *,
    tau: float,
    potential: str,
    prev_scores: torch.Tensor | None,
    running_max: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if potential == "difference":
        inc = scores if prev_scores is None else (scores - prev_scores)
        log_belief = log_belief + float(tau) * inc
        prev_scores = scores.clone()
    else:
        running_max = scores.clone() if running_max is None else torch.maximum(running_max, scores)
        log_belief = float(tau) * running_max
    w = torch.softmax(log_belief, dim=0)
    return log_belief, w, prev_scores, running_max


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FK steering with ImageReward.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--suite_image_root", type=str, default="/root/autodl-tmp/dit_candidates_50k_cfg1.0_deno10")
    p.add_argument("--suite_repr_dir", type=str, default="/root/autodl-tmp/repr_candidate_deno10_cfg1.0/mean")
    p.add_argument("--num_candidates", type=int, default=10, help="parallel size")
    p.add_argument("--batch_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--search_batch_size", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--n_init", type=int, default=0)
    p.add_argument("--init_candidate_idx", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--potential", type=str, choices=("difference", "max"), default="difference")
    p.add_argument("--fk_resample", action="store_true")
    p.add_argument("--pick", type=str, choices=("sample", "argmax"), default="sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imagenet_val", type=str, default="/root/autodl-tmp/imagenet/val")
    p.add_argument("--metrics_batch_size", type=int, default=64)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_root = Path(args.suite_image_root).expanduser().resolve()
    repr_d = Path(args.suite_repr_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()

    bs = int(args.search_batch_size) if args.search_batch_size else int(args.batch_size)
    pool, pre_meta = dsr.load_precomputed_pool_from_rollout_suite(
        repr_d,
        suite_root,
        int(args.num_candidates),
        batch_index=int(args.batch_index),
        repr_keys=["s0_r0.6"],
        batch_size=bs,
    )
    if args.n_init > 0:
        cand_dir = suite_root / f"candidate_{int(args.init_candidate_idx):04d}"
        copied = dsr.copy_init_images_from_rollout_candidate(output_dir, cand_dir, int(args.n_init))
        print(f"init copied: {copied} from {cand_dir}")
    img_counter = int(args.n_init)

    rm = load_imagereward(device)
    k = len(pool)
    log_belief = torch.zeros(k, dtype=torch.float64)
    prev_scores: torch.Tensor | None = None
    running_max: torch.Tensor | None = None
    num_rounds = (int(args.num_samples) - img_counter + bs - 1) // bs
    trace: list[dict] = []

    for step in tqdm(range(max(num_rounds, 0)), desc="fk imagereward"):
        n_this = min(bs, int(args.num_samples) - img_counter)
        if n_this <= 0:
            break
        scores = torch.empty(k, dtype=torch.float64)
        round_entries = []
        for j in range(k):
            imgs_f, labels_f, _repr_f = pool[j]
            imgs_s = imgs_f[:n_this]
            labels_s = labels_f[:n_this]
            reward_ls = []
            for im, lab in zip(imgs_s, labels_s):
                prompt = f"a photo of a {labels_1k[int(lab)]}"
                reward_ls.append(float(rm.score(prompt, im)))
            scores[j] = float(sum(reward_ls) / max(len(reward_ls), 1))
            round_entries.append((imgs_s, labels_s))

        log_belief, w, prev_scores, running_max = _update_fk_state(
            log_belief,
            scores,
            tau=float(args.tau),
            potential=str(args.potential),
            prev_scores=prev_scores,
            running_max=running_max,
        )
        if args.fk_resample and args.pick == "sample":
            idx = int(torch.multinomial(w.float(), 1).item())
        else:
            idx = int(torch.argmax(w).item()) if args.pick == "argmax" else int(torch.multinomial(w.float(), 1).item())

        best_imgs, best_labels = round_entries[idx]
        for img, lab in zip(best_imgs, best_labels):
            if img_counter >= int(args.num_samples):
                break
            img.save(output_dir / f"image_{img_counter:05d}_{int(lab):05d}.png")
            img_counter += 1

        trace.append(
            {
                "step": int(step),
                "scores": [float(x) for x in scores.tolist()],
                "weights": [float(x) for x in w.tolist()],
                "picked_index": int(idx),
            }
        )

    (output_dir / "fk_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "reward": "imagereward",
                "potential": args.potential,
                "num_candidates": int(args.num_candidates),
                "batch_size": bs,
                "num_samples": int(args.num_samples),
                "pre_meta": pre_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if imagenet_val.is_dir():
        m = compute_fid_is_from_dirs(output_dir, imagenet_val, device, batch_size=int(args.metrics_batch_size))
        (output_dir / "final_metrics.json").write_text(
            json.dumps({k: float(v) for k, v in m.items()}, indent=2), encoding="utf-8"
        )
        print(f"FID={m['FID']:.2f}, IS={m['IS']:.4f}±{m['IS_STD']:.4f}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

