#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import dit_loader, set_seed
from class_condition.prob.dataset import HiddenRewardDataset
from class_condition.prob.prob import RewardProb
from class_condition.prob.utils import (
    broadcast_timesteps,
    decode_dit_latents_to_pil,
    load_imagenet_class_names,
    pearsonr_torch,
)

_DEFAULT_CLIP_DIR = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")


def _clipscore_rewards(
    images,
    prompt: str,
    clip: CLIPModel,
    processor: CLIPProcessor,
) -> torch.Tensor:
    prompts = [prompt] * len(images)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(clip.device) for k, v in inputs.items()}
    out = clip(**inputs)
    image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text_features = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    sim = (image_features * text_features).sum(dim=-1)
    return sim.detach().float().cpu()


@torch.inference_mode()
def _mc_clipscore_from_snapshot(
    *,
    pipe,
    x_start: torch.Tensor,
    prompt: str,
    start_step_idx: int,
    timesteps: torch.Tensor,
    scheduler_snapshot,
    class_labels_input: torch.Tensor,
    latent_channels: int,
    guidance_scale: float,
    seed_base: int,
    mc: int,
    clip: CLIPModel,
    processor: CLIPProcessor,
) -> torch.Tensor:
    if start_step_idx >= len(timesteps):
        return _clipscore_rewards(
            decode_dit_latents_to_pil(pipe, x_start, guidance_scale),
            prompt,
            clip,
            processor,
        )

    rewards_ls: list[torch.Tensor] = []
    for i in range(int(mc)):
        scheduler = copy.deepcopy(scheduler_snapshot)
        x = x_start.detach().clone()
        gen = torch.Generator(device=x.device).manual_seed(int(seed_base) + i)
        for st_i in range(int(start_step_idx), int(len(timesteps))):
            t = timesteps[st_i]
            if guidance_scale > 1:
                half = x[: len(x) // 2]
                x = torch.cat([half, half], dim=0)
            x_in = scheduler.scale_model_input(x, t)
            timesteps_b = broadcast_timesteps(t, x_in)
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
            x = scheduler.step(model_output, t, x, generator=gen).prev_sample

        rewards_ls.append(
            _clipscore_rewards(
                decode_dit_latents_to_pil(pipe, x, guidance_scale),
                prompt,
                clip,
                processor,
            )
        )
    return torch.stack(rewards_ls, dim=0).mean(dim=0)


@torch.inference_mode()
def collect_single_class_data(
    *,
    output_dir: Path,
    class_idx: int,
    num_samples: int = 5000,
    target_step_indices: tuple[int, ...] = (1, 5, 10, 15, 20, 24),
    target_layer_idx: int = 16,
    mc: int = 8,
    model_name: str = "dit-xl-2-256",
    dtype: str = "bf16",
    num_inference_steps: int = 25,
    batch_size: int = 32,
    guidance_scale: float = 1.0,
    seed: int = 42,
    clip_model_dir: str = str(_DEFAULT_CLIP_DIR),
) -> dict[int, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (0 <= int(class_idx) < 1000):
        raise ValueError(f"class_idx must be in [0, 999], got {class_idx}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(model_name, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    clip = CLIPModel.from_pretrained(str(clip_model_dir), local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(clip_model_dir), local_files_only=True)
    class_names = load_imagenet_class_names()
    prompt = f"A photo of a {class_names[int(class_idx)]}"

    n_blocks = len(pipe.transformer.transformer_blocks)
    if not (0 <= int(target_layer_idx) < n_blocks):
        raise ValueError(f"target_layer_idx must be in [0, {n_blocks - 1}]")
    target_steps = sorted(set(int(s) for s in target_step_indices))
    if not target_steps:
        raise ValueError("target_step_indices cannot be empty")
    for s in target_steps:
        if not (0 <= s < int(num_inference_steps)):
            raise ValueError(f"target step {s} out of range [0, {int(num_inference_steps) - 1}]")

    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels
    n_batches = (int(num_samples) + int(batch_size) - 1) // int(batch_size)
    all_hidden: dict[int, list[torch.Tensor]] = {s: [] for s in target_steps}
    all_reward: dict[int, list[torch.Tensor]] = {s: [] for s in target_steps}
    all_seed: list[int] = []

    current_step = {"value": -1}
    current_batch = {"value": 0}
    hidden_buf: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out) -> None:
        if current_step["value"] not in target_steps:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        if guidance_scale > 1:
            h = h[: current_batch["value"]]
        hidden_buf[str(current_step["value"])] = h.mean(dim=1).detach().float().cpu()

    handle = pipe.transformer.transformer_blocks[int(target_layer_idx)].register_forward_hook(_hook)
    try:
        pbar = tqdm(range(n_batches), desc="collect hidden/reward (CLIPScore)")
        for bi in pbar:
            st = bi * int(batch_size)
            ed = min(st + int(batch_size), int(num_samples))
            batch_n = ed - st
            if batch_n <= 0:
                break
            current_batch["value"] = batch_n
            batch_seed = int(seed) + bi
            gen = torch.Generator(device=device).manual_seed(batch_seed)
            latents = torch.randn(
                (batch_n, latent_channels, latent_size, latent_size),
                generator=gen,
                device=device,
                dtype=pipe.transformer.dtype,
            )
            x = torch.cat([latents] * 2, dim=0) if guidance_scale > 1 else latents
            cls = torch.full((batch_n,), int(class_idx), device=device, dtype=torch.long)
            cls_null = torch.full((batch_n,), 1000, device=device, dtype=torch.long)
            class_labels_input = torch.cat([cls, cls_null], dim=0) if guidance_scale > 1 else cls

            pipe.scheduler.set_timesteps(int(num_inference_steps))
            timesteps = pipe.scheduler.timesteps
            hidden_buf.clear()
            snapshot_x_by_step: dict[int, torch.Tensor] = {}
            snapshot_scheduler_by_step: dict[int, object] = {}

            for st_i, t in enumerate(timesteps):
                current_step["value"] = st_i
                if guidance_scale > 1:
                    half = x[: len(x) // 2]
                    x = torch.cat([half, half], dim=0)
                x_in = pipe.scheduler.scale_model_input(x, t)
                timesteps_b = broadcast_timesteps(t, x_in)
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
                if st_i in target_steps:
                    snapshot_x_by_step[st_i] = x.detach().clone()
                    snapshot_scheduler_by_step[st_i] = copy.deepcopy(pipe.scheduler)

            for s in target_steps:
                k = str(s)
                if k not in hidden_buf:
                    raise RuntimeError(f"Failed to capture hidden at step {s}")
                if s not in snapshot_x_by_step or s not in snapshot_scheduler_by_step:
                    raise RuntimeError(f"Failed to snapshot x/scheduler at step {s}")
                reward = _mc_clipscore_from_snapshot(
                    pipe=pipe,
                    x_start=snapshot_x_by_step[s],
                    prompt=prompt,
                    start_step_idx=s + 1,
                    timesteps=timesteps,
                    scheduler_snapshot=snapshot_scheduler_by_step[s],
                    class_labels_input=class_labels_input,
                    latent_channels=latent_channels,
                    guidance_scale=float(guidance_scale),
                    seed_base=int(seed) + bi * 10_007 + s * 97,
                    mc=int(mc),
                    clip=clip,
                    processor=processor,
                )
                all_hidden[s].append(hidden_buf[k])
                all_reward[s].append(reward)
            all_seed.extend([batch_seed + i for i in range(batch_n)])

            del latents, x, cls, cls_null, class_labels_input, snapshot_x_by_step, snapshot_scheduler_by_step
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    seeds = torch.tensor(all_seed, dtype=torch.long)
    out_by_step: dict[int, Path] = {}
    for s in target_steps:
        hiddens = torch.cat(all_hidden[s], dim=0).float()
        rewards = torch.cat(all_reward[s], dim=0).float()
        if hiddens.shape[0] != int(num_samples) or rewards.shape[0] != int(num_samples):
            raise RuntimeError(f"rows mismatch at step {s}: hidden={hiddens.shape[0]} reward={rewards.shape[0]} expected={num_samples}")
        out = output_dir / f"class_clipscore_hs_only_c{class_idx}_n{num_samples}_s{s}_l{target_layer_idx}_mc{mc}.pt"
        torch.save(
            {
                "hiddens": hiddens,
                "reward": rewards,
                "labels": torch.full((int(num_samples),), int(class_idx), dtype=torch.long),
                "seed": seeds,
                "meta": {
                    "class_idx": int(class_idx),
                    "class_name": class_names[int(class_idx)],
                    "prompt": prompt,
                    "num_samples": int(num_samples),
                    "target_step_idx": int(s),
                    "target_step_indices": target_steps,
                    "target_layer_idx": int(target_layer_idx),
                    "num_inference_steps": int(num_inference_steps),
                    "guidance_scale": float(guidance_scale),
                    "mc": int(mc),
                    "model_name": str(model_name),
                    "dtype": str(dtype),
                    "reward_type": "clipscore",
                    "prob_input": "hidden_only",
                },
            },
            out,
        )
        out_by_step[s] = out
    return out_by_step


def train_prob_hidden_only(
    *,
    data_pt: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.9,
    train_batch_size: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout_prob: float = 0.1,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    obj = torch.load(data_pt, map_location="cpu", weights_only=False)
    hiddens = obj["hiddens"].float()
    rewards_raw = obj["reward"].view(-1).float()
    n = int(hiddens.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    n_train = min(max(int(round(n * float(train_ratio))), 1), n - 1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_h = hiddens[train_idx]
    val_h = hiddens[val_idx]
    train_r_raw = rewards_raw[train_idx]
    val_r_raw = rewards_raw[val_idx]

    r_mean = float(train_r_raw.mean().item())
    r_std = float(train_r_raw.std().item())
    denom = r_std + 1e-8
    train_r = (train_r_raw - r_mean) / denom
    val_r = (val_r_raw - r_mean) / denom

    train_loader = DataLoader(
        HiddenRewardDataset(train_h, train_r),
        batch_size=int(train_batch_size),
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        HiddenRewardDataset(val_h, val_r),
        batch_size=int(train_batch_size),
        shuffle=False,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardProb(
        input_dim=int(train_h.shape[1]),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout_prob=float(dropout_prob),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "prob_best.pt"
    history = []
    best_val = float("inf")
    best_epoch = -1

    @torch.no_grad()
    def _eval_val():
        model.eval()
        total = 0.0
        preds: list[torch.Tensor] = []
        gts: list[torch.Tensor] = []
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)
            pred = model(x)
            total += float(loss_fn(pred, y).item())
            preds.append(pred.detach().cpu().view(-1))
            gts.append(y.detach().cpu().view(-1))
        return total / max(1, len(val_loader)), torch.cat(preds), torch.cat(gts)

    for ep in range(int(epochs)):
        model.train()
        tr_total = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_total += float(loss.detach().item())
        tr_loss = tr_total / max(1, len(train_loader))
        va_loss, _, _ = _eval_val()
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"epoch {ep:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = ep
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": va_loss}, best_ckpt)

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()
    val_loss_norm, val_pred_norm, val_gt_norm = _eval_val()
    val_pred_raw = val_pred_norm * denom + r_mean
    val_gt_raw = val_gt_norm * denom + r_mean
    val_pearson = pearsonr_torch(val_pred_raw, val_gt_raw)

    result = {
        "data_pt": str(data_pt),
        "best_ckpt": str(best_ckpt),
        "best_epoch": int(best_epoch),
        "best_val_loss_norm": float(best_val),
        "final_val_loss_norm": float(val_loss_norm),
        "val_pearson_raw": float(val_pearson),
        "num_train": int(train_idx.numel()),
        "num_val": int(val_idx.numel()),
        "epochs": int(epochs),
        "lr": float(lr),
        "reward_mean_train": float(r_mean),
        "reward_std_train": float(r_std),
        "reward_type": "clipscore",
        "prob_input": "hidden_only",
    }
    (output_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Class-conditional DiT hidden-only prob trainer (CLIPScore labels)")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/class_prob_clipscore_hidden_only")
    p.add_argument("--data_pt", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--class_idx", type=int, default=256)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--collect_batch_size", type=int, default=32)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--target_step_indices", type=str, default="1,5,10,15,20,24")
    p.add_argument("--train_step_idx", type=int, default=24)
    p.add_argument("--target_layer_idx", type=int, default=16)
    p.add_argument("--mc", type=int, default=8)
    p.add_argument("--clip_model_dir", type=str, default=str(_DEFAULT_CLIP_DIR))
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--prob_num_layers", type=int, default=2)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_dir = output_dir / "data"
    run_dir = output_dir / "train_run"
    target_steps = tuple(sorted(set(int(x.strip()) for x in str(args.target_step_indices).split(",") if x.strip())))
    if int(args.train_step_idx) not in target_steps:
        raise ValueError(f"train_step_idx {args.train_step_idx} must be in target_step_indices={target_steps}")

    if str(args.data_pt).strip():
        data_pt = Path(args.data_pt).expanduser().resolve()
    else:
        data_pt_by_step = collect_single_class_data(
            output_dir=data_dir,
            class_idx=int(args.class_idx),
            num_samples=int(args.num_samples),
            target_step_indices=target_steps,
            target_layer_idx=int(args.target_layer_idx),
            mc=int(args.mc),
            model_name=str(args.model_name),
            dtype=str(args.dtype),
            num_inference_steps=int(args.num_inference_steps),
            batch_size=int(args.collect_batch_size),
            guidance_scale=float(args.guidance_scale),
            seed=int(args.seed),
            clip_model_dir=str(args.clip_model_dir),
        )
        data_pt = data_pt_by_step[int(args.train_step_idx)]

    result = train_prob_hidden_only(
        data_pt=data_pt,
        output_dir=run_dir,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        train_batch_size=int(args.train_batch_size),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.prob_num_layers),
        dropout_prob=float(args.dropout_prob),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

