#!/usr/bin/env python3
"""
Rollout script for google/ddpm-cifar10-32 using pretrained CIFAR-10 classifier reward.

Reward: negative entropy from pretrained CIFAR-10 classifier.
  Higher reward = sharper class prediction = better image quality.

UNet structure: 22 ResnetBlock2D layers.
  layer_ratio ~0.6 => layer_idx=12, shape (B, 256, 8, 8)
  step_ratios [0.0, 0.25, 0.5, 0.75, 1.0] with num_inference_steps=50
    => step_indices [0, 12, 24, 37, 49]

Example:
  python /root/diffusion_ranker/uncondition_cifar10/cifar10_ddpm_rollout.py \
    --classifier_ckpt /root/autodl-tmp/cifar10_classifier/resnet18/classifier_best.pt
"""
from __future__ import annotations

import copy
import gc
import json
import sys
from pathlib import Path

import argparse

import torch
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uncondition_cifar10.cifar10_classifier_reward import (
    load_cifar10_classifier,
    reward_neg_entropy_cifar10,
)

# ---- Hard-coded targets for CIFAR-10 DDPM (22 ResnetBlock2D, 50 steps) ----
# layer ratio 0.6 -> layer index 12  (shape: B,256,8,8)
# step ratios [0.0, 0.25, 0.5, 0.75, 1.0] -> step indices [0, 25, 50, 75, 100]
_TARGET_LAYER_INDICES = [12]
_TARGET_STEP_INDICES = [0, 100, 200, 300, 400, 500]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CIFAR10 DDPM rollout with classifier reward")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/google_ddpm-cifar10-32")
    p.add_argument(
        "--classifier_ckpt",
        type=str,
        default="/root/autodl-tmp/cifar10_classifier/resnet18/classifier_best.pt",
    )
    p.add_argument("--save_path", type=str, default="/root/autodl-tmp/cifar10_ddpm_hiddens_and_rewards.pth")
    p.add_argument("--num_samples", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_inference_steps", type=int, default=500)
    p.add_argument("--mc_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _latents_to_pil(x: torch.Tensor) -> list[Image.Image]: # 意义是将latents转换为PIL图像
    x = (x / 2 + 0.5).clamp(0, 1)
    arr = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype("uint8")
    return [Image.fromarray(im) for im in arr]


@torch.inference_mode()
def mc(
    step_idx: int,
    batch_middle_x: torch.Tensor,
    pipe: DDPMPipeline,
    classifier,
    mc_samples: int,
    base_seed: int,
) -> torch.Tensor:
    device = batch_middle_x.device
    timesteps = pipe.scheduler.timesteps
    reward_ls: list[torch.Tensor] = []
    for j in range(mc_samples):
        sched = copy.deepcopy(pipe.scheduler)
        x = batch_middle_x.clone()
        gen = torch.Generator(device=device).manual_seed(base_seed + j)
        for t in timesteps[step_idx + 1 :]:
            eps = pipe.unet(x, t).sample
            x = sched.step(eps, t, x, generator=gen).prev_sample
            del eps
        images = _latents_to_pil(x)
        reward_ls.append(reward_neg_entropy_cifar10(classifier, images, device))
        del images, x, gen, sched
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.stack(reward_ls, dim=0).mean(dim=0)


@torch.inference_mode()
def rollout() -> None:
    args = parse_args()
    ddpm_path = Path(args.model_path).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = DDPMPipeline.from_pretrained(
        str(ddpm_path),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    classifier = load_cifar10_classifier(args.classifier_ckpt, device)

    num_samples = int(args.num_samples)
    batch_size = int(args.batch_size)
    num_inference_steps = int(args.num_inference_steps)
    mc_samples = int(args.mc_samples)
    seed = int(args.seed)

    generator = torch.Generator(device=device).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    resnet_blocks = [
        (name, mod)
        for name, mod in pipe.unet.named_modules()
        if mod.__class__.__name__ == "ResnetBlock2D"
    ]
    print(f"total ResnetBlock2D layers: {len(resnet_blocks)}")

    target_layer_indices = sorted(set(int(x) for x in _TARGET_LAYER_INDICES))
    target_step_indices = sorted(set(int(x) for x in _TARGET_STEP_INDICES))

    save_path = Path(args.save_path).expanduser().resolve()
    shard_dir = save_path.parent / f"{save_path.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    hidden_shard_paths: dict[str, list[Path]] = {
        f"s{s}_l{li}": [] for s in target_step_indices for li in target_layer_indices
    }
    reward_shard_paths: dict[int, list[Path]] = {s: [] for s in target_step_indices}

    current_step_idx = {"value": -1}
    capture_enabled = {"value": True}
    current_store_ref: dict[str, dict[int, dict[int, list[torch.Tensor]]] | None] = {"store": None}

    def _make_hook(layer_idx: int):
        def _hook(_module, _inp, _out):
            s = current_step_idx["value"]
            store = current_store_ref["store"]
            if (not capture_enabled["value"]) or store is None or s < 0 or s not in store:
                return
            t = _out[0] if isinstance(_out, (tuple, list)) else _out
            t = t.detach().float()
            store[s][layer_idx].append(t.to(torch.float16).cpu())
        return _hook

    handles = [
        resnet_blocks[li][1].register_forward_hook(_make_hook(li))
        for li in target_layer_indices
    ]

    n_batches = (num_samples + batch_size - 1) // batch_size
    pbar = tqdm(range(n_batches), total=n_batches, desc="CIFAR-10 DDPM rollout")
    for batch_i in pbar:
        batch_n = min(batch_size, num_samples - batch_i * batch_size)
        if batch_n <= 0:
            break
        x = torch.randn(
            (
                batch_n,
                pipe.unet.config.in_channels,
                pipe.unet.config.sample_size,
                pipe.unet.config.sample_size,
            ),
            generator=generator,
            device=device,
            dtype=next(pipe.unet.parameters()).dtype,
        )

        batch_hidden_store: dict[int, dict[int, list[torch.Tensor]]] = {
            s: {li: [] for li in target_layer_indices} for s in target_step_indices
        }
        batch_reward_store: dict[int, list[torch.Tensor]] = {s: [] for s in target_step_indices}
        current_store_ref["store"] = batch_hidden_store

        for st_i, t in enumerate(timesteps):
            current_step_idx["value"] = st_i
            eps = pipe.unet(x, t).sample
            x = pipe.scheduler.step(eps, t, x, generator=generator).prev_sample
            del eps
            if st_i in target_step_indices:
                capture_enabled["value"] = False
                batch_reward_store[st_i].append(
                    mc(
                        step_idx=st_i,
                        batch_middle_x=x,
                        pipe=pipe,
                        classifier=classifier,
                        mc_samples=mc_samples,
                        base_seed=seed + batch_i * 1000 + st_i * 10,
                    )
                )
                capture_enabled["value"] = True
        current_store_ref["store"] = None

        for st_i in target_step_indices:
            r_chunks = batch_reward_store[st_i]
            if r_chunks:
                r_t = torch.cat(r_chunks, dim=0)
                r_path = shard_dir / f"reward_s{st_i}_b{batch_i:05d}.pt"
                torch.save(r_t, r_path)
                reward_shard_paths[st_i].append(r_path)
                del r_t
            for li in target_layer_indices:
                h_chunks = batch_hidden_store[st_i][li]
                if h_chunks:
                    h_t = torch.cat(h_chunks, dim=0)
                    h_key = f"s{st_i}_l{li}"
                    h_path = shard_dir / f"hidden_{h_key}_b{batch_i:05d}.pt"
                    torch.save(h_t, h_path)
                    hidden_shard_paths[h_key].append(h_path)
                    del h_t

        del x, batch_hidden_store, batch_reward_store
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    reward_store_tensor: dict[str, torch.Tensor] = {}
    for st_i in target_step_indices:
        parts = [torch.load(p, map_location="cpu", weights_only=False) for p in reward_shard_paths[st_i]]
        reward_store_tensor[f"s{st_i}"] = torch.cat(parts, dim=0) if parts else torch.empty(0)
        del parts
        gc.collect()

    hidden_store_target: dict[str, torch.Tensor] = {}
    for st_i in target_step_indices:
        for li in target_layer_indices:
            key = f"s{st_i}_l{li}"
            parts = [torch.load(p, map_location="cpu", weights_only=False) for p in hidden_shard_paths[key]]
            hidden_store_target[key] = torch.cat(parts, dim=0) if parts else torch.empty(0)
            del parts
            gc.collect()

    full_data = {
        "hidden_store_target": hidden_store_target,
        "reward_store": reward_store_tensor,
        "meta": {
            "model": "google/ddpm-cifar10-32",
            "classifier_ckpt": str(Path(args.classifier_ckpt).expanduser().resolve()),
            "target_layer_indices": target_layer_indices,
            "target_step_indices": target_step_indices,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "num_inference_steps": num_inference_steps,
            "mc_samples": mc_samples,
            "seed": seed,
        },
    }

    torch.save(full_data, save_path)

    for key, t in hidden_store_target.items():
        print(f"  hidden[{key}] shape={tuple(t.shape)}")
    for key, t in reward_store_tensor.items():
        print(f"  reward[{key}] shape={tuple(t.shape)}")
    print(json.dumps({"saved": str(save_path), "shard_dir": str(shard_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    rollout()
