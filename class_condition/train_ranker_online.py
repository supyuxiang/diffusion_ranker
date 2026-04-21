#!/usr/bin/env python3
"""
Online train ranker on ImageNet DiT rollout.

Pipeline per class group:
1) DiT rollout (same class, different noise)
2) capture one mid-step hidden state from one transformer block
3) mean-pool token sequence -> candidate feature (K, D)
4) compute teacher reward from decoded images (Inception max-softmax confidence)
5) train ranker immediately on this group
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from class_condition.ranker import DiffusionRanker_Transformer
from class_condition.utils import RankerLoss
from utils import INCEPTION_WEIGHTS_PATH, dit_loader, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Online train ranker with DiT rollout")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/ranker_online")
    p.add_argument("--run_name", type=str, default="dit_online_listwise")
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--num_classes_per_epoch", type=int, default=1000)
    p.add_argument("--group_size", type=int, default=8, help="K candidates per class group")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--target_step_idx", type=int, default=6)
    p.add_argument("--target_layer_idx", type=int, default=16)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--normalize_reward_per_group", action="store_true")

    p.add_argument("--input_dim_candidate", type=int, default=1152)
    p.add_argument("--input_dim_prompt", type=int, default=1000)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--half", action="store_true", default=True)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--kl_type", type=str, default="kl", choices=("kl", "js"))
    p.add_argument("--save_interval", type=int, default=1)
    return p.parse_args()


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
    images: list[Image.Image] = []
    for i in range(samples.shape[0]):
        arr = samples[i].detach().cpu().permute(1, 2, 0).float().numpy()
        images.append(Image.fromarray((arr * 255.0).round().clip(0, 255).astype("uint8")))
    return images


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
    x = torch.stack([transform(img) for img in images], dim=0).to(device)
    logits = feat_extractor(x)[0]
    probs = F.softmax(logits.float(), dim=-1)
    return probs.max(dim=-1).values.to(torch.float32)


@torch.inference_mode()
def rollout_one_class_group(
    *,
    pipe,
    feat_extractor,
    class_id: int,
    group_size: int,
    num_inference_steps: int,
    target_step_idx: int,
    target_layer_idx: int,
    guidance_scale: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((group_size, latent_channels, latent_size, latent_size), generator=gen, device=device, dtype=pipe.transformer.dtype)
    x = torch.cat([latents, latents], dim=0) if guidance_scale > 1 else latents

    cls = torch.full((group_size,), int(class_id), device=device, dtype=torch.long)
    cls_null = torch.full((group_size,), 1000, device=device, dtype=torch.long)
    class_labels_input = torch.cat([cls, cls_null], dim=0) if guidance_scale > 1 else cls

    hidden_capture: dict[str, torch.Tensor | None] = {"h": None}
    current_step = {"idx": -1}

    def _hook(_module, _inp, out):
        if current_step["idx"] != target_step_idx:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        if guidance_scale > 1:
            h = h[:group_size]
        hidden_capture["h"] = h.detach()

    handle = pipe.transformer.transformer_blocks[target_layer_idx].register_forward_hook(_hook)
    try:
        pipe.scheduler.set_timesteps(int(num_inference_steps))
        for st_i, t in enumerate(pipe.scheduler.timesteps):
            current_step["idx"] = st_i
            if guidance_scale > 1:
                half = x[: len(x) // 2]
                x = torch.cat([half, half], dim=0)
            x_in = pipe.scheduler.scale_model_input(x, t)
            timesteps_b = _broadcast_timesteps(t, x_in)
            noise_pred = pipe.transformer(x_in, timestep=timesteps_b, class_labels=class_labels_input).sample

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
            x = pipe.scheduler.step(model_output, t, x, generator=gen).prev_sample
    finally:
        handle.remove()

    h = hidden_capture["h"]
    if h is None:
        raise RuntimeError(f"hidden not captured at target_step_idx={target_step_idx}")
    pooled_h = h.float().mean(dim=1)  # (K, D)

    images = _decode_dit_latents_to_pil(pipe, x, guidance_scale)
    reward = _reward_inception_confidence(images, feat_extractor, device)  # (K,)
    return pooled_h, reward


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    feat_extractor = _build_inception_logits_extractor(device)

    ranker = DiffusionRanker_Transformer(
        input_dim_candidate=args.input_dim_candidate,
        input_dim_prompt=args.input_dim_prompt,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_layers=args.num_layers,
        half=args.half,
    ).to(device)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = RankerLoss(alpha=args.alpha, kl_type=args.kl_type)

    all_classes = torch.arange(1000, dtype=torch.long)
    for epoch in range(args.epochs):
        ranker.train()
        epoch_loss = 0.0

        g = torch.Generator().manual_seed(args.seed + epoch)
        perm = torch.randperm(1000, generator=g)
        class_ids = all_classes[perm[: args.num_classes_per_epoch]].tolist()

        pbar = tqdm(class_ids, desc=f"epoch {epoch+1}/{args.epochs}", leave=True)
        for ci, class_id in enumerate(pbar):
            pooled_h, reward = rollout_one_class_group(
                pipe=pipe,
                feat_extractor=feat_extractor,
                class_id=int(class_id),
                group_size=args.group_size,
                num_inference_steps=args.num_inference_steps,
                target_step_idx=args.target_step_idx,
                target_layer_idx=args.target_layer_idx,
                guidance_scale=args.guidance_scale,
                seed=args.seed + epoch * 100000 + ci * 1000,
                device=device,
            )

            if args.normalize_reward_per_group:
                reward = (reward - reward.mean()) / (reward.std() + 1e-8)

            cand = pooled_h.to(device=device, dtype=next(ranker.parameters()).dtype).unsqueeze(0)  # (1, K, D)
            target = reward.to(device=device, dtype=cand.dtype).unsqueeze(0)  # (1, K)

            pred = ranker(cand, None)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().item())
            pbar.set_postfix(loss=f"{float(loss.detach().item()):.4f}")

        epoch_loss /= max(1, len(class_ids))
        print(f"epoch {epoch} loss={epoch_loss:.6f}")

        if (epoch + 1) % int(args.save_interval) == 0:
            ckpt = out_dir / f"ranker_online_epoch{epoch+1}.pt"
            torch.save(
                {
                    "model": ranker.state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                },
                ckpt,
            )

    with open(out_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

