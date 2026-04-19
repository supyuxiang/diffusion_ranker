#!/usr/bin/env python3
"""
Multi-class ImageNet DiT rollout with mid-step hidden capture and
single-sample Inception confidence reward.

This script is the rollout/data-building counterpart of
`search/dit_random_search_is_reward.py`:
- generate ImageNet class-conditional DiT samples in class-id order
- capture hidden states at target steps / one transformer block
- score final images with single-sample IS-style reward:
    max softmax( InceptionV3 logits )

Default setup:
- per_samples = 64
- total_samples = 64000
- num_inference_steps = 10
- target_step_indices = [1, 6, 9]
- target_layer_idx = 16
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import CenterCrop, Compose, Resize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import INCEPTION_WEIGHTS_PATH, dit_loader, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ImageNet DiT rollout with single-sample IS reward")
    p.add_argument(
        "--save_path",
        type=str,
        default="/root/autodl-tmp/dit_imagenet_hiddens_and_is_rewards_s1_6_9_l16_per64.pth",
    )
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--per_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--target_step_indices", type=str, default="1,6,9")
    p.add_argument("--target_layer_idx", type=int, default=16)
    return p.parse_args()


def _parse_int_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    return sorted(set(vals))


def _build_labels_all(per_samples: int, num_classes: int = 1000) -> list[int]:
    return [c for c in range(num_classes) for _ in range(per_samples)]


def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        is_mps = latent_model_input.device.type == "mps"
        is_npu = latent_model_input.device.type == "npu"
        if isinstance(t, float):
            dt = torch.float32 if (is_mps or is_npu) else torch.float64
        else:
            dt = torch.int32 if (is_mps or is_npu) else torch.int64
        timesteps = torch.tensor([t], dtype=dt, device=latent_model_input.device)
    elif len(t.shape) == 0:
        timesteps = t[None].to(latent_model_input.device)
    else:
        timesteps = t
    return timesteps.expand(latent_model_input.shape[0])


def _decode_dit_latents_to_pil(pipe, x: torch.Tensor, guidance_scale: float) -> list[Image.Image]:
    if guidance_scale > 1:
        latents, _ = x.chunk(2, dim=0)
    else:
        latents = x
    latents = 1 / pipe.vae.config.scaling_factor * latents
    samples = pipe.vae.decode(latents).sample
    samples = (samples / 2 + 0.5).clamp(0, 1)

    out_images: list[Image.Image] = []
    for bi in range(samples.shape[0]):
        arr = samples[bi].detach().cpu().permute(1, 2, 0).float().numpy()
        out_images.append(Image.fromarray((arr * 255.0).round().clip(0, 255).astype("uint8")))
    return out_images


def _build_inception_logits_extractor(device: torch.device):
    from torch_fidelity.utils import create_feature_extractor

    return create_feature_extractor(
        "inception-v3-compat",
        ["logits_unbiased"],
        cuda=(device.type == "cuda"),
        verbose=False,
        feature_extractor_weights_path=INCEPTION_WEIGHTS_PATH,
    )


@torch.inference_mode()
def _reward_inception_confidence(images: list[Image.Image], feat_extractor, device: torch.device) -> torch.Tensor:
    from torch_fidelity.datasets import TransformPILtoRGBTensor

    transform = Compose([Resize(256), CenterCrop(256), TransformPILtoRGBTensor()])
    tensors = torch.stack([transform(img) for img in images], dim=0).to(
        device, non_blocking=(device.type == "cuda")
    )
    logits = feat_extractor(tensors)
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = F.softmax(logits.float(), dim=-1)
    return probs.max(dim=-1).values.detach().cpu().to(torch.float32)


@torch.inference_mode()
def rollout() -> None:
    args = parse_args()
    set_seed(args.seed)

    target_step_indices = _parse_int_list(args.target_step_indices)
    if not target_step_indices:
        raise ValueError("target_step_indices is empty")
    if min(target_step_indices) < 0 or max(target_step_indices) >= int(args.num_inference_steps):
        raise ValueError(
            f"target_step_indices={target_step_indices} must be in [0, {int(args.num_inference_steps) - 1}]"
        )
    if int(args.per_samples) <= 0:
        raise ValueError(f"per_samples must be positive, got {args.per_samples}")

    save_path = Path(args.save_path).expanduser().resolve()
    shard_dir = save_path.parent / f"{save_path.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    feat_extractor = _build_inception_logits_extractor(device)

    n_blocks = len(pipe.transformer.transformer_blocks)
    target_layer_idx = int(args.target_layer_idx)
    if target_layer_idx < 0 or target_layer_idx >= n_blocks:
        raise ValueError(f"target_layer_idx={target_layer_idx} out of range for n_blocks={n_blocks}")

    target_ratio = float(target_layer_idx) / float(max(1, n_blocks - 1))
    print(
        f"[info] transformer blocks={n_blocks}, target_layer_idx={target_layer_idx}, "
        f"equivalent_ratio={target_ratio:g}"
    )

    hidden_shard_paths: dict[str, list[Path]] = {
        f"s{s}_l{target_layer_idx}": [] for s in target_step_indices
    }
    reward_shard_paths: dict[int, list[Path]] = {s: [] for s in target_step_indices}
    label_shard_paths: list[Path] = []

    labels_all = _build_labels_all(int(args.per_samples))
    num_samples = len(labels_all)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    current_step_idx = {"value": -1}
    current_batch_size = {"value": 0}
    current_store_ref: dict[str, dict[int, list[torch.Tensor]] | None] = {"store": None}

    def _hook(_module, _inp, out) -> None:
        s = current_step_idx["value"]
        store = current_store_ref["store"]
        if store is None or s not in store:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        if h.dim() != 3:
            return
        if float(args.guidance_scale) > 1:
            h = h[: current_batch_size["value"]]
        store[s].append(h.detach().to(torch.float16).cpu())

    handle = pipe.transformer.transformer_blocks[target_layer_idx].register_forward_hook(_hook)

    try:
        n_batches = (num_samples + int(args.batch_size) - 1) // int(args.batch_size)
        pbar = tqdm(range(n_batches), total=n_batches, desc="ImageNet DiT rollout with IS reward")
        for batch_i in pbar:
            batch_start = batch_i * int(args.batch_size)
            batch_end = min(batch_start + int(args.batch_size), num_samples)
            batch_labels = labels_all[batch_start:batch_end]
            batch_n = len(batch_labels)
            if batch_n <= 0:
                break

            current_batch_size["value"] = batch_n
            label_t = torch.tensor(batch_labels, dtype=torch.long)
            lab_path = shard_dir / f"label_b{batch_i:05d}.pt"
            torch.save(label_t, lab_path)
            label_shard_paths.append(lab_path)

            gen = torch.Generator(device=device).manual_seed(int(args.seed) + batch_i)
            latents = torch.randn(
                (batch_n, latent_channels, latent_size, latent_size),
                generator=gen,
                device=device,
                dtype=pipe.transformer.dtype,
            )
            x = torch.cat([latents] * 2, dim=0) if float(args.guidance_scale) > 1 else latents

            cls = torch.tensor(batch_labels, device=device).reshape(-1)
            cls_null = torch.tensor([1000] * batch_n, device=device)
            class_labels_input = (
                torch.cat([cls, cls_null], dim=0) if float(args.guidance_scale) > 1 else cls
            )

            pipe.scheduler.set_timesteps(int(args.num_inference_steps))
            timesteps = pipe.scheduler.timesteps

            batch_hidden_store: dict[int, list[torch.Tensor]] = {s: [] for s in target_step_indices}
            current_store_ref["store"] = batch_hidden_store

            for st_i, t in enumerate(timesteps):
                current_step_idx["value"] = st_i
                if float(args.guidance_scale) > 1:
                    half = x[: len(x) // 2]
                    x = torch.cat([half, half], dim=0)
                x = pipe.scheduler.scale_model_input(x, t)
                timesteps_b = _broadcast_timesteps(t, x)
                noise_pred = pipe.transformer(
                    x,
                    timestep=timesteps_b,
                    class_labels=class_labels_input,
                ).sample

                if float(args.guidance_scale) > 1:
                    eps, tail = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    half_eps = uncond_eps + float(args.guidance_scale) * (cond_eps - uncond_eps)
                    eps = torch.cat([half_eps, half_eps], dim=0)
                    noise_pred = torch.cat([eps, tail], dim=1)

                if pipe.transformer.config.out_channels // 2 == latent_channels:
                    model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
                else:
                    model_output = noise_pred

                x = pipe.scheduler.step(model_output, t, x, generator=gen).prev_sample
                del noise_pred, model_output

            current_store_ref["store"] = None
            images = _decode_dit_latents_to_pil(pipe, x, float(args.guidance_scale))
            reward_t = _reward_inception_confidence(images, feat_extractor, device)

            for st_i in target_step_indices:
                h_chunks = batch_hidden_store[st_i]
                if h_chunks:
                    h_t = torch.cat(h_chunks, dim=0)
                    h_key = f"s{st_i}_l{target_layer_idx}"
                    h_path = shard_dir / f"hidden_{h_key}_b{batch_i:05d}.pt"
                    torch.save(h_t, h_path)
                    hidden_shard_paths[h_key].append(h_path)
                    del h_t

                r_path = shard_dir / f"reward_s{st_i}_b{batch_i:05d}.pt"
                torch.save(reward_t, r_path)
                reward_shard_paths[st_i].append(r_path)

            del latents, x, cls, cls_null, class_labels_input, batch_hidden_store, gen, label_t, images, reward_t
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    reward_store_tensor: dict[str, torch.Tensor] = {}
    for st_i in target_step_indices:
        parts = [torch.load(p, map_location="cpu", weights_only=False) for p in reward_shard_paths[st_i]]
        reward_store_tensor[f"s{st_i}"] = torch.cat(parts, dim=0) if parts else torch.empty(0)
        del parts
        gc.collect()

    hidden_store_target: dict[str, torch.Tensor] = {}
    for st_i in target_step_indices:
        key = f"s{st_i}_l{target_layer_idx}"
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
            "model_name": str(args.model_name),
            "target_layer_indices": [target_layer_idx],
            "target_step_indices": target_step_indices,
            "num_samples": int(num_samples),
            "per_samples": int(args.per_samples),
            "batch_size": int(args.batch_size),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "seed": int(args.seed),
            "dtype": str(args.dtype),
            "label_order_note": (
                "labels are generated in ascending class-id order, each class repeated exactly per_samples times"
            ),
            "reward_note": "reward_s* uses single-sample Inception max-softmax confidence on final images",
            "verifier": "inception_max_softmax_confidence",
            "layer_index_ratio": float(target_ratio),
        },
    }
    torch.save(full_data, save_path)
    print(
        json.dumps(
            {
                "saved": str(save_path),
                "shard_dir": str(shard_dir),
                "num_hidden_keys": len(hidden_store_target),
                "num_reward_keys": len(reward_store_tensor),
                "num_labels": int(label_store.numel()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    rollout()
