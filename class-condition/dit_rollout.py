#!/usr/bin/env python3
'''

python /root/diffusion_ranker/class-condition/dit_rollout.py \
  --output_path /root/autodl-tmp/dit_hiddens_and_rewards_custom.pth \
  --target_step_indices 0,2,4,7,9 \
  --target_layer_ratios 0.3,0.6,0.9 \
  --num_samples 50000 \
  --batch_size 128

'''
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from search.reward import labels_1k, load_imagereward
from utils import DitRolloutReprCapture, dit_loader, set_seed


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _ratio_tag(x: float) -> str:
    return f"{float(x):g}".replace(".", "p")


@torch.inference_mode()
def _reward_imagereward_by_label(rm_model, images: list[Image.Image], labels: list[int]) -> torch.Tensor:
    vals: list[float] = []
    for img, lab in zip(images, labels):
        prompt = f"a photo of a {labels_1k[int(lab)]}"
        vals.append(float(rm_model.score(prompt, img)))
    return torch.tensor(vals, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DiT rollout for class-conditional hidden/reward shards")
    p.add_argument("--output_path", type=str, default="/root/autodl-tmp/dit_hiddens_and_rewards_s02479.pth")
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--pooling_mode", type=str, default="mean", choices=("mean", "last"))
    p.add_argument("--target_step_indices", type=str, default="0,2,4,7,9")
    p.add_argument("--target_layer_ratios", type=str, default="0.6")
    return p.parse_args()


@torch.inference_mode()
def rollout() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = Path(args.output_path).expanduser().resolve()
    shard_dir = save_path.with_suffix("")
    shard_dir = shard_dir.parent / f"{shard_dir.name}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    target_step_indices = sorted(set(_parse_int_list(args.target_step_indices)))
    target_layer_ratios = [float(x) for x in _parse_float_list(args.target_layer_ratios)]
    if not target_step_indices:
        raise ValueError("target_step_indices is empty")
    if not target_layer_ratios:
        raise ValueError("target_layer_ratios is empty")
    for s in target_step_indices:
        if s < 0 or s >= int(args.num_inference_steps):
            raise ValueError(
                f"step index out of range: {s}, num_inference_steps={args.num_inference_steps}"
            )
    step_ratios = [float(s) / float(max(1, args.num_inference_steps - 1)) for s in target_step_indices]

    pipe = dit_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    rm_model = load_imagereward(device)

    capture = DitRolloutReprCapture(
        pipe,
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        layer_ratios=tuple(target_layer_ratios),
        step_ratio_ls=step_ratios,
        store_dtype=torch.float16,
        rollout_base_seed=int(args.seed),
    )
    capture.install()

    hidden_shard_paths: dict[str, list[Path]] = {
        f"s{s}_r{r:g}": [] for s in target_step_indices for r in target_layer_ratios
    }
    reward_shard_paths: dict[int, list[Path]] = {s: [] for s in target_step_indices}
    label_shard_paths: list[Path] = []

    n_batches = (int(args.num_samples) + int(args.batch_size) - 1) // int(args.batch_size)
    img_counter = 0
    try:
        pbar = tqdm(range(n_batches), total=n_batches, desc="Running DiT rollout")
        for batch_i in pbar:
            batch_n = min(int(args.batch_size), int(args.num_samples) - batch_i * int(args.batch_size))
            if batch_n <= 0:
                break

            labels = torch.randint(0, 1000, (batch_n,), device=device).tolist()
            gen = torch.Generator(device=device).manual_seed(int(args.seed) + batch_i)

            capture.set_capturing(True)
            capture.begin_batch(labels)
            out = pipe(
                class_labels=labels,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.guidance_scale),
                generator=gen,
            )
            capture.end_batch(batch_i, img_counter, generator_seed=int(args.seed) + batch_i)
            if not capture.records:
                raise RuntimeError("repr capture failed: no records for current batch")
            rec = capture.records.pop()
            repr_dict = rec.get("repr", {})

            reward_t = _reward_imagereward_by_label(rm_model, out.images, labels)
            label_t = torch.tensor(labels, dtype=torch.long)
            lab_path = shard_dir / f"label_b{batch_i:05d}.pt"
            torch.save(label_t, lab_path)
            label_shard_paths.append(lab_path)

            for s in target_step_indices:
                r_path = shard_dir / f"reward_s{s}_b{batch_i:05d}.pt"
                torch.save(reward_t, r_path)
                reward_shard_paths[s].append(r_path)
                for r in target_layer_ratios:
                    key = f"s{s}_r{r:g}"
                    if key not in repr_dict:
                        raise KeyError(f"repr key {key!r} not found; available keys={list(repr_dict.keys())[:8]}")
                    h = repr_dict[key][str(args.pooling_mode)].to(torch.float16).cpu()
                    h_path = shard_dir / f"hidden_s{s}_r{_ratio_tag(r)}_b{batch_i:05d}.pt"
                    torch.save(h, h_path)
                    hidden_shard_paths[key].append(h_path)

            img_counter += batch_n
            del out, gen, reward_t, label_t, repr_dict
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        capture.remove()

    reward_store_tensor: dict[str, torch.Tensor] = {}
    for s in target_step_indices:
        parts = [torch.load(p, map_location="cpu", weights_only=False) for p in reward_shard_paths[s]]
        reward_store_tensor[f"s{s}"] = torch.cat(parts, dim=0) if parts else torch.empty(0)
        del parts
        gc.collect()

    hidden_store_target: dict[str, torch.Tensor] = {}
    for s in target_step_indices:
        for r in target_layer_ratios:
            key = f"s{s}_r{r:g}"
            parts = [torch.load(p, map_location="cpu", weights_only=False) for p in hidden_shard_paths[key]]
            hidden_store_target[key] = torch.cat(parts, dim=0) if parts else torch.empty(0)
            del parts
            gc.collect()

    label_parts = [torch.load(p, map_location="cpu", weights_only=False) for p in label_shard_paths]
    label_store = torch.cat(label_parts, dim=0) if label_parts else torch.empty(0, dtype=torch.long)
    del label_parts

    full_data = {
        "hidden_store_target": hidden_store_target,
        "reward_store": reward_store_tensor,
        "label_store": label_store,
        "meta": {
            "target_layer_ratios": target_layer_ratios,
            "target_step_indices": target_step_indices,
            "num_samples": int(args.num_samples),
            "batch_size": int(args.batch_size),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "seed": int(args.seed),
            "dtype": str(args.dtype),
            "pooling_mode": str(args.pooling_mode),
            "model_name": str(args.model_name),
            "reward_note": "reward_s* uses final-image ImageReward per sample (same labels/prompts for all steps).",
        },
    }
    torch.save(full_data, save_path)
    print(
        json.dumps(
            {
                "saved": str(save_path),
                "num_hidden_keys": len(hidden_store_target),
                "num_reward_keys": len(reward_store_tensor),
                "num_labels": int(label_store.numel()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    rollout()
