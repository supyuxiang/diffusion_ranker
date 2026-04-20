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
- per_samples = 128
- total_samples = 128 * 1000 = 128000
- num_inference_steps = 10
- target_step_indices = [1, 6, 9]
- target_layer_idx = 16
"""

from __future__ import annotations

import argparse
import copy
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
sys.path.insert(0, str(_ROOT))

from utils import INCEPTION_WEIGHTS_PATH, dit_loader, set_seed


# ------------------------------- I love Peking University -------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ImageNet DiT rollout with single-sample IS reward")
    p.add_argument(
        "--save_path",
        type=str,
        default="/root/autodl-tmp/dit_imagenet_hiddens_and_is_rewards_s1_6_9_l16_per128.pth",
    )
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--per_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--target_step_indices", type=str, default="1,6,9")
    p.add_argument("--target_layer_idx", type=int, default=16)
    p.add_argument("--mc", type=int, default=5)
    return p.parse_args()


def _build_labels_all(per_samples: int, num_classes: int = 1000) -> list[int]:
    return [c for c in range(num_classes) for _ in range(per_samples)]


def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    # 把时间步 t 统一成模型可用的 batch 形状时间向量: (batch_size,)
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
    # images: list[PIL.Image], length = B
    tensors = torch.stack([transform(img) for img in images], dim=0).to(
        device, non_blocking=(device.type == "cuda")
    )  # (B, 3, 256, 256), uint8 on device
    logits = feat_extractor(tensors)  # tuple/list with first item shaped (B, 1000)
    logits = logits[0]  # (B, 1008)
    probs = F.softmax(logits.float(), dim=-1)  # (B, 1008)
    # reward per sample: max class confidence -> (B,)
    return probs.max(dim=-1).values.detach().cpu().to(torch.float32)


@torch.inference_mode()
def mc_for_estimate_reward(
    pipe,
    x_start: torch.Tensor,
    start_step_idx: int,
    timesteps: torch.Tensor,
    scheduler_snapshot,
    class_labels_input: torch.Tensor,
    latent_channels: int,
    guidance_scale: float,
    feat_extractor,
    device: torch.device,
    seed_base: int,
    mc: int = 5,
) -> torch.Tensor:
    """
    Estimate reward from a target-step latent by MC continuation rollout.
    Returns per-sample reward mean with shape (B,).
    """

    if start_step_idx >= len(timesteps):
        images = _decode_dit_latents_to_pil(pipe, x_start, guidance_scale)
        return _reward_inception_confidence(images, feat_extractor, device)

    rewards_ls: list[torch.Tensor] = []
    for i in range(int(mc)):
        scheduler = copy.deepcopy(scheduler_snapshot)
        x = x_start.detach().clone()
        gen = torch.Generator(device=device).manual_seed(int(seed_base) + i)

        for st_i in range(int(start_step_idx), int(len(timesteps))):
            t = timesteps[st_i]
            if guidance_scale > 1:
                half = x[: len(x) // 2]
                x = torch.cat([half, half], dim=0)

            x_in = scheduler.scale_model_input(x, t)
            timesteps_b = _broadcast_timesteps(t, x_in)
            noise_pred = pipe.transformer(
                x_in,
                timestep=timesteps_b,
                class_labels=class_labels_input,
            ).sample

            if guidance_scale > 1:
                eps, tail = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                noise_pred = torch.cat([eps, tail], dim=1)

            if pipe.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            x = scheduler.step(model_output, t, x, generator=gen).prev_sample
            del noise_pred, model_output, x_in

        images = _decode_dit_latents_to_pil(pipe, x, guidance_scale)
        reward_t = _reward_inception_confidence(images, feat_extractor, device)  # (B,)
        rewards_ls.append(reward_t)
        del x, images, reward_t
        gc.collect()
        torch.cuda.empty_cache()

    return torch.stack(rewards_ls, dim=0).mean(dim=0).to(torch.float32)  # (B,)



@torch.inference_mode()
def rollout() -> None:
    args = parse_args()
    set_seed(args.seed)

    target_step_indices = [
        int(s)
        for s in args.target_step_indices.strip().split(',') if s.strip()
    ] # [1, 6, 9]

    save_path = Path(args.save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    feat_extractor = _build_inception_logits_extractor(device)

    n_blocks = len(pipe.transformer.transformer_blocks) # 28, idx:0->27
    target_layer_idx = int(args.target_layer_idx) # 16: 0.6 * 28 -1 -> 16

    target_ratio = float(target_layer_idx) / float(max(1, n_blocks - 1))
    print(
        f"[info] transformer blocks={n_blocks}, target_layer_idx={target_layer_idx}, "
        f"equivalent_ratio={target_ratio:g}"
    )

    # In-memory accumulation, then save one big pt per (step, layer).
    hidden_chunks_by_key: dict[str, list[torch.Tensor]] = {
        f"s{s}_l{target_layer_idx}": [] for s in target_step_indices
    }
    reward_chunks_by_key: dict[str, list[torch.Tensor]] = {
        f"s{s}_l{target_layer_idx}": [] for s in target_step_indices
    }
    label_chunks: list[torch.Tensor] = []

    labels_all = _build_labels_all(int(args.per_samples)) # 128 * 1000, 1
    num_samples = len(labels_all) # 128 * 1000
    latent_size = pipe.transformer.config.sample_size # 
    latent_channels = pipe.transformer.config.in_channels # 

     # 定义全局变量，以便在 hook 函数中访问
    current_step_idx = {"value": -1}
    current_batch_size = {"value": 0}
    current_store_ref: dict[str, dict[int, list[torch.Tensor]] | None] = {"store": None}
    hook_enabled = {"value": True}

    def _hook(_module, _inp, out) -> None:
        if not hook_enabled["value"]:
            return
        s = current_step_idx["value"]
        store = current_store_ref["store"]
        if store is None or s not in store:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
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
            label_t = torch.tensor(batch_labels, dtype=torch.long) # sample labels, batch_size
            label_chunks.append(label_t.clone()) # store sample labels, batch_size

            gen = torch.Generator(device=device).manual_seed(int(args.seed) + batch_i) # build gen
            latents = torch.randn(
                (batch_n, latent_channels, latent_size, latent_size),
                generator=gen,
                device=device,
                dtype=pipe.transformer.dtype,
            ) # sample initial noisy: batch_n, latent_channels, latent_size, latent_size
            x = torch.cat([latents] * 2, dim=0) if float(args.guidance_scale) > 1 else latents # cfg>1则需要考虑uncod和cond的加权求和,于是需要cat两个latents
            # x: batch_n * 2, latent_channels, latent_size, latent_size

            cls = torch.tensor(batch_labels, device=device).reshape(-1) # ensure shape is (batch_n,)
            cls_null = torch.tensor([1000] * batch_n, device=device) # 1000: null class for cfg>1
            class_labels_input = (
                torch.cat([cls, cls_null], dim=0) if float(args.guidance_scale) > 1 else cls
            ) # shape: (batch_n * 2,) if cfg>1 else (batch_n,)

            pipe.scheduler.set_timesteps(int(args.num_inference_steps)) # 10， set 调度器timesteps: 10, [0, 9]
            timesteps = pipe.scheduler.timesteps

            batch_hidden_store: dict[int, list[torch.Tensor]] = {s: [] for s in target_step_indices}
            current_store_ref["store"] = batch_hidden_store # current store <- batch_hidden_store
            batch_snapshot_x: dict[int, torch.Tensor] = {}
            batch_snapshot_scheduler: dict[int, object] = {}

            for st_i, t in enumerate(timesteps):
                current_step_idx["value"] = st_i # update current step idx
                if float(args.guidance_scale) > 1:
                    half = x[: len(x) // 2]
                    x = torch.cat([half, half], dim=0) # copy half 
                x = pipe.scheduler.scale_model_input(x, t) # scale model input
                timesteps_b = _broadcast_timesteps(t, x) # 
                noise_pred = pipe.transformer(
                    x,
                    timestep=timesteps_b,
                    class_labels=class_labels_input,
                ).sample # predict noise

                if float(args.guidance_scale) > 1:
                    # eps: 2*batch_n, latent_channels, latent_size, latent_size
                    # tail: 2*batch_n, latent_channels, latent_size, latent_size
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

                if st_i in target_step_indices:
                    # snapshot latent/scheduler at this target step, then continue from st_i+1 in MC rollout
                    batch_snapshot_x[st_i] = x.detach().clone()
                    batch_snapshot_scheduler[st_i] = copy.deepcopy(pipe.scheduler)

            current_store_ref["store"] = None # clear current store
            hook_enabled["value"] = False  # disable hook during MC continuation rollout
            batch_reward_by_step: dict[int, torch.Tensor] = {}
            for st_i in target_step_indices:
                if st_i not in batch_snapshot_x:
                    raise RuntimeError(f"target step {st_i} not captured for current batch")
                batch_reward_by_step[st_i] = mc_for_estimate_reward(
                    pipe=pipe,
                    x_start=batch_snapshot_x[st_i],
                    start_step_idx=st_i + 1,
                    timesteps=timesteps,
                    scheduler_snapshot=batch_snapshot_scheduler[st_i],
                    class_labels_input=class_labels_input,
                    latent_channels=latent_channels,
                    guidance_scale=float(args.guidance_scale),
                    feat_extractor=feat_extractor,
                    device=device,
                    seed_base=int(args.seed) + batch_i * 10000 + st_i * 100,
                    mc=int(args.mc),
                )
            hook_enabled["value"] = True

            for st_i in target_step_indices:
                h_chunks = batch_hidden_store[st_i] # get hidden chunks for current step
                h_key = f"s{st_i}_l{target_layer_idx}"
                if h_chunks:
                    h_t = torch.cat(h_chunks, dim=0) # cat hidden chunks
                    hidden_chunks_by_key[h_key].append(h_t) # append to hidden chunks by key
                    del h_t # clear h_t

                # reward is computed on final images; one reward vector aligns with this batch order.
                reward_chunks_by_key[h_key].append(batch_reward_by_step[st_i].clone()) # append to reward chunks by key

            del latents, x, cls, cls_null, class_labels_input, batch_hidden_store, batch_reward_by_step, batch_snapshot_x, batch_snapshot_scheduler, gen, label_t # clear variables
            gc.collect() # collect garbage
            if device.type == "cuda":
                torch.cuda.empty_cache() # empty cache
    finally:
        handle.remove() # remove hook

    label_store = torch.cat(label_chunks, dim=0) if label_chunks else torch.empty(0, dtype=torch.long) # cat label chunks
    del label_chunks 

    saved_paths: list[Path] = []
    for st_i in target_step_indices:
        key = f"s{st_i}_l{target_layer_idx}"
        hidden_t = (
            torch.cat(hidden_chunks_by_key[key], dim=0)
            if hidden_chunks_by_key[key]
            else torch.empty(0)
        )
        reward_t = (
            torch.cat(reward_chunks_by_key[key], dim=0)
            if reward_chunks_by_key[key]
            else torch.empty(0, dtype=torch.float32)
        )
        if hidden_t.shape[0] != reward_t.shape[0]:
            raise RuntimeError(
                f"{key}: rows mismatch hidden={hidden_t.shape[0]} reward={reward_t.shape[0]}"
            )
        if hidden_t.shape[0] != label_store.shape[0]:
            raise RuntimeError(
                f"{key}: rows mismatch hidden={hidden_t.shape[0]} labels={label_store.shape[0]}"
            )

        out_path = save_path.parent / f"{save_path.stem}_{key}.pt"
        torch.save(
            {
                "hiddens": hidden_t, # hidden states at target step
                "reward": reward_t, # reward at target step
                "labels": label_store, # sample labels
                "meta": {
                    "model_name": str(args.model_name),
                    "target_step_index": int(st_i),
                    "target_layer_index": int(target_layer_idx),
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
                    "reward_note": "reward uses MC-mean Inception max-softmax confidence from target-step latent continuation",
                    "reward_mc": int(args.mc),
                    "verifier": "inception_max_softmax_confidence",
                    "layer_index_ratio": float(target_ratio),
                },
            },
            out_path,
        )
        saved_paths.append(out_path)
        del hidden_t, reward_t
        gc.collect()

    print(
        json.dumps(
            {
                "saved": [str(p) for p in saved_paths],
                "num_labels": int(label_store.numel()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    rollout()
