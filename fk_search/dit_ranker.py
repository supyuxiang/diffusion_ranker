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
from reward import ranker_mean_logit_score
from utils import compute_fid_is_from_dirs, set_seed


def load_prompt_table(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    t = obj["tensor"] if isinstance(obj, dict) and "tensor" in obj else obj
    if not isinstance(t, torch.Tensor) or t.dim() != 2:
        raise ValueError(f"prompt 表须为 2D Tensor，得到 {type(t)} shape={getattr(t, 'shape', None)}")
    return t.cpu()


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
    p = argparse.ArgumentParser(description="FK steering with ranker reward.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--suite_image_root", type=str, required=True)
    p.add_argument("--suite_repr_dir", type=str, required=True)
    p.add_argument("--ranker_ckpt", type=str, required=True)
    p.add_argument("--prompt_repr_pt", type=str, default="/root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt")
    p.add_argument("--num_candidates", type=int, default=10, help="parallel size")
    p.add_argument("--batch_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--n_init", type=int, default=0)
    p.add_argument("--init_candidate_idx", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--potential", type=str, choices=("difference", "max"), default="difference")
    p.add_argument("--fk_resample", action="store_true")
    p.add_argument("--pick", type=str, choices=("sample", "argmax"), default="sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imagenet_val", type=str, default="/root/autodl-tmp/imagenet/val")
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
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

    ranker, selected_keys, _listwise, _meta = dsr.load_diffusion_ranker(
        Path(args.ranker_ckpt).expanduser().resolve(),
        device,
        num_attention_heads=int(args.num_attention_heads),
        dropout=float(args.dropout),
    )
    repr_key = selected_keys[0]
    prompt_table = load_prompt_table(Path(args.prompt_repr_pt).expanduser().resolve())

    pool, pre_meta = dsr.load_precomputed_pool_from_rollout_suite(
        repr_d,
        suite_root,
        int(args.num_candidates),
        batch_index=int(args.batch_index),
        repr_keys=selected_keys,
        batch_size=int(args.batch_size),
    )
    dsr.verify_precomputed_pool_ranker_keys(pool, selected_keys, repr_dir_hint=repr_d)

    if args.n_init > 0:
        cand_dir = suite_root / f"candidate_{int(args.init_candidate_idx):04d}"
        copied = dsr.copy_init_images_from_rollout_candidate(output_dir, cand_dir, int(args.n_init))
        print(f"init copied: {copied} from {cand_dir}")
    img_counter = int(args.n_init)

    k = len(pool)
    log_belief = torch.zeros(k, dtype=torch.float64)
    prev_scores: torch.Tensor | None = None
    running_max: torch.Tensor | None = None
    num_rounds = (int(args.num_samples) - img_counter + int(args.batch_size) - 1) // int(args.batch_size)
    trace: list[dict] = []

    for step in tqdm(range(max(num_rounds, 0)), desc="fk ranker"):
        n_this = min(int(args.batch_size), int(args.num_samples) - img_counter)
        if n_this <= 0:
            break
        scores = torch.empty(k, dtype=torch.float64)
        round_entries = []
        for j in range(k):
            imgs_f, labels_f, repr_f = pool[j]
            imgs_s = imgs_f[:n_this]
            labels_s = labels_f[:n_this]
            repr_s = dsr.slice_repr_batch(repr_f, n_this) if n_this < len(labels_f) else repr_f
            s = ranker_mean_logit_score(
                ranker=ranker,
                repr_dict=repr_s,
                repr_key=repr_key,
                labels=[int(x) for x in labels_s],
                prompt_table=prompt_table,
                device=device,
            )
            scores[j] = float(s)
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
                "reward": "ranker",
                "repr_key": repr_key,
                "potential": args.potential,
                "num_candidates": int(args.num_candidates),
                "batch_size": int(args.batch_size),
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

