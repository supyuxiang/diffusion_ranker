#!/usr/bin/env python3


'''
python3 ranker/mid_val.py \
  --ranker_ckpt /root/autodl-tmp/ranker_mid_ckpt_pointwise_l0p6_s2/ranker_epoch10.pt \
  --prompt_repr_pt /root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt \
  --layer_ratio 0.6 \
  --step_ratio 0.25 \
  --output_dir /root/autodl-tmp/mid_val_random_search_l0p6_s2

'''
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SEARCH = _ROOT / "search"
for p in (_ROOT, _SEARCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from search.dit_random_search_ranker import load_ranker_from_ckpt
from search.reward import ranker_mean_logit_score
from utils import DitRolloutReprCapture, compute_fid_is_from_dirs, dit_loader, set_seed


def step_ratio_to_idx(step_ratio: float, num_inference_steps: int) -> int:
    t = int(num_inference_steps)
    if t <= 1:
        return 0
    return min(t - 1, max(0, round(float(step_ratio) * (t - 1))))


def load_prompt_table(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    t = obj["tensor"] if isinstance(obj, dict) and "tensor" in obj else obj
    if not isinstance(t, torch.Tensor) or t.dim() != 2:
        raise ValueError(f"prompt 表须为 2D Tensor，得到 {type(t)} shape={getattr(t, 'shape', None)}")
    return t.cpu()


@torch.inference_mode()
def rollout_candidate_with_repr(
    pipe,
    repr_capture: DitRolloutReprCapture,
    *,
    num_steps: int,
    n_this: int,
    device: torch.device,
    generator_seed: int,
    guidance_scale: float,
) -> tuple[list, list[int], dict]:
    labels = torch.randint(0, 1000, (n_this,), device=device).tolist()
    gen = torch.Generator(device=device).manual_seed(int(generator_seed))
    repr_capture.set_capturing(True)
    repr_capture.begin_batch(labels)
    out = pipe(
        class_labels=labels,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=gen,
    )
    repr_capture.end_batch(0, 0, generator_seed=int(generator_seed))
    rec = repr_capture.records.pop()
    repr_dict = rec["repr"]
    images = [im.copy() for im in out.images]
    del out, gen
    return images, labels, repr_dict


@torch.inference_mode()
def run_online_random_search_ranker(
    *,
    pipe,
    repr_capture: DitRolloutReprCapture,
    ranker,
    prompt_table: torch.Tensor,
    repr_key: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    num_steps: int,
    device: torch.device,
    seed: int,
    guidance_scale: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_counter = 0
    num_rounds = (num_samples + batch_size - 1) // batch_size

    for _ in range(num_rounds):
        n_this = min(batch_size, num_samples - img_counter)
        if n_this <= 0:
            break

        best_score = float("-inf")
        best_images = None
        best_labels = None

        for _k in range(num_candidates):
            cand_seed = int(rng.integers(0, 2**31 - 1))
            images, labels, repr_dict = rollout_candidate_with_repr(
                pipe,
                repr_capture,
                num_steps=num_steps,
                n_this=n_this,
                device=device,
                generator_seed=cand_seed,
                guidance_scale=guidance_scale,
            )
            score = ranker_mean_logit_score(
                ranker=ranker,
                repr_dict=repr_dict,
                repr_key=repr_key,
                labels=labels,
                prompt_table=prompt_table,
                device=device,
            )
            if score > best_score:
                best_score = score
                best_images = images
                best_labels = labels

        assert best_images is not None and best_labels is not None
        for img, lab in zip(best_images, best_labels):
            if img_counter >= num_samples:
                break
            img.save(output_dir / f"image_{img_counter:05d}_{int(lab):05d}.png")
            img_counter += 1


def main() -> None:
    p = argparse.ArgumentParser(description="Online random search with ranker reward.")
    p.add_argument("--ranker_ckpt", type=str, required=True)
    p.add_argument("--prompt_repr_pt", type=str, required=True)
    p.add_argument("--layer_ratio", type=float, required=True)
    p.add_argument("--step_ratio", type=float, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--num_candidates", type=int, default=10)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dit_path", type=str, default="/root/autodl-tmp/DiT-XL-2-256")
    p.add_argument("--imagenet_val", type=str, default="/root/autodl-tmp/imagenet/val")
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dtype", type=str, default="fp16")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()
    prompt_p = Path(args.prompt_repr_pt).expanduser().resolve()
    ckpt_p = Path(args.ranker_ckpt).expanduser().resolve()
    dit_path = Path(args.dit_path).expanduser().resolve()

    step_idx = step_ratio_to_idx(args.step_ratio, args.deno_steps)
    repr_key = f"s{step_idx}_r{float(args.layer_ratio):g}"
    print(f"repr_key={repr_key}")

    ranker = load_ranker_from_ckpt(
        ckpt_p, device, num_attention_heads=args.num_attention_heads, dropout=args.dropout
    )
    prompt_table = load_prompt_table(prompt_p)

    pipe = dit_loader(dit_path, device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    repr_capture = DitRolloutReprCapture(
        pipe,
        num_inference_steps=args.deno_steps,
        guidance_scale=args.guidance_scale,
        layer_ratios=(float(args.layer_ratio),),
        step_ratio_ls=[float(args.step_ratio)],
        store_dtype=torch.float16,
        rollout_base_seed=args.seed,
    )
    repr_capture.install()

    try:
        run_online_random_search_ranker(
            pipe=pipe,
            repr_capture=repr_capture,
            ranker=ranker,
            prompt_table=prompt_table,
            repr_key=repr_key,
            output_dir=output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_candidates=args.num_candidates,
            num_steps=args.deno_steps,
            device=device,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
        )
    finally:
        repr_capture.remove()

    cfg = {
        "ranker_ckpt": str(ckpt_p),
        "prompt_repr_pt": str(prompt_p),
        "repr_key": repr_key,
        "layer_ratio": float(args.layer_ratio),
        "step_ratio": float(args.step_ratio),
        "output_dir": str(output_dir),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "num_candidates": int(args.num_candidates),
        "deno_steps": int(args.deno_steps),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if imagenet_val.is_dir():
        print("computing FID / IS ...")
        metrics = compute_fid_is_from_dirs(
            output_dir,
            imagenet_val,
            device,
            batch_size=args.metrics_batch_size,
        )
        (output_dir / "final_metrics.json").write_text(
            json.dumps({k: float(v) for k, v in metrics.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"FID={metrics['FID']:.2f}, IS={metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}"
        )
    else:
        print(f"skip metrics: imagenet_val not found: {imagenet_val}")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

