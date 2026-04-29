#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Resize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import INCEPTION_WEIGHTS_PATH, dit_loader, set_seed


_HIDDEN_POOL_SEQ: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": lambda h: h.mean(dim=1),
    "last": lambda h: h[:, -1, :],
}


class ProbIS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_prob: float,
        half: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.LayerNorm(4 * hidden_dim),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                )
                for _ in range(num_layers)
            ]
        )
        self.reward_head = nn.Linear(hidden_dim, 1)
        self._init_weights()
        if half:
            self.to(torch.float16)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.reward_head(x)

    def count_param(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HiddenRewardDataset(Dataset):
    def __init__(self, hiddens: torch.Tensor, rewards: torch.Tensor):
        self.hiddens = hiddens.float()
        self.rewards = rewards.view(-1, 1).float()
        if self.hiddens.shape[0] != self.rewards.shape[0]:
            raise ValueError(
                f"rows mismatch: hiddens={self.hiddens.shape[0]}, rewards={self.rewards.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.hiddens.shape[0])

    def __getitem__(self, idx: int):
        return self.hiddens[idx], self.rewards[idx]


def _parse_step_indices(s: str) -> list[int]:
    out = [int(x) for x in s.strip().split(",") if x.strip()]
    if not out:
        raise ValueError("target_step_indices cannot be empty")
    return sorted(list(set(out)))


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
        arr = samples[bi].detach().cpu().permute(1, 2, 0).float().numpy()  # (C, H, W) -> (H, W, C)
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
    )  # (B, 3, 256, 256), uint8
    logits = feat_extractor(tensors)[0]  # (B, 1000)
    probs = F.softmax(logits.float(), dim=-1)
    return probs.max(dim=-1).values.detach().cpu().to(torch.float32)  # (B,)


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
            del noise_pred, model_output, x_in

        images = _decode_dit_latents_to_pil(pipe, x, guidance_scale)
        reward_t = _reward_inception_confidence(images, feat_extractor, device)
        rewards_ls.append(reward_t)
        del x, images, reward_t
        gc.collect()
        torch.cuda.empty_cache()

    return torch.stack(rewards_ls, dim=0).mean(dim=0).to(torch.float32)  # (B,)


@torch.inference_mode()
def rollout_training_data(
    *,
    save_prefix: Path,
    model_name: str,
    per_samples: int,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    dtype: str,
    target_step_indices: list[int],
    target_layer_idx: int,
    hidden_pool: str,
    mc: int,
) -> list[Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(model_name, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    feat_extractor = _build_inception_logits_extractor(device)

    n_blocks = len(pipe.transformer.transformer_blocks)
    if not (0 <= int(target_layer_idx) < n_blocks):
        raise ValueError(f"target_layer_idx must be in [0, {n_blocks - 1}]")

    pool_hidden_fn = _HIDDEN_POOL_SEQ[str(hidden_pool)]
    hidden_chunks_by_key: dict[str, list[torch.Tensor]] = {
        f"s{s}_l{target_layer_idx}": [] for s in target_step_indices
    }
    reward_chunks_by_key: dict[str, list[torch.Tensor]] = {
        f"s{s}_l{target_layer_idx}": [] for s in target_step_indices
    }
    label_chunks: list[torch.Tensor] = []

    labels_all = [c for c in range(1000) for _ in range(int(per_samples))]
    num_samples = len(labels_all)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    current_step_idx = {"value": -1}
    current_batch_size = {"value": 0}
    current_store_ref: dict[str, dict[int, list[torch.Tensor]] | None] = {"store": None}
    hook_enabled = {"value": True}

    def _hook(_module, _inp, out):
        if not hook_enabled["value"]:
            return
        s = current_step_idx["value"]
        store = current_store_ref["store"]
        if store is None or s not in store:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        if guidance_scale > 1:
            h = h[: current_batch_size["value"]]
        h = pool_hidden_fn(h)
        store[s].append(h.detach().to(torch.float16).cpu())

    handle = pipe.transformer.transformer_blocks[target_layer_idx].register_forward_hook(_hook)

    try:
        n_batches = (num_samples + int(batch_size) - 1) // int(batch_size)
        pbar = tqdm(range(n_batches), total=n_batches, desc="rollout for ProbIS (IS reward)")
        for batch_i in pbar:
            st = batch_i * int(batch_size)
            ed = min(st + int(batch_size), num_samples)
            batch_labels = labels_all[st:ed]
            batch_n = len(batch_labels)
            if batch_n <= 0:
                break

            current_batch_size["value"] = batch_n
            label_t = torch.tensor(batch_labels, dtype=torch.long)
            label_chunks.append(label_t.clone())

            gen = torch.Generator(device=device).manual_seed(int(seed) + batch_i)
            latents = torch.randn(
                (batch_n, latent_channels, latent_size, latent_size),
                generator=gen,
                device=device,
                dtype=pipe.transformer.dtype,
            )
            x = torch.cat([latents] * 2, dim=0) if guidance_scale > 1 else latents

            cls = torch.tensor(batch_labels, device=device).reshape(-1)
            cls_null = torch.tensor([1000] * batch_n, device=device)
            class_labels_input = torch.cat([cls, cls_null], dim=0) if guidance_scale > 1 else cls

            pipe.scheduler.set_timesteps(int(num_inference_steps))
            timesteps = pipe.scheduler.timesteps

            batch_hidden_store: dict[int, list[torch.Tensor]] = {s: [] for s in target_step_indices}
            current_store_ref["store"] = batch_hidden_store
            batch_snapshot_x: dict[int, torch.Tensor] = {}
            batch_snapshot_scheduler: dict[int, object] = {}

            for st_i, t in enumerate(timesteps):
                current_step_idx["value"] = st_i
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
                del noise_pred, model_output, x_in

                if st_i in target_step_indices:
                    batch_snapshot_x[st_i] = x.detach().clone()
                    batch_snapshot_scheduler[st_i] = copy.deepcopy(pipe.scheduler)

            current_store_ref["store"] = None
            hook_enabled["value"] = False
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
                    guidance_scale=float(guidance_scale),
                    feat_extractor=feat_extractor,
                    device=device,
                    seed_base=int(seed) + batch_i * 10000 + st_i * 100,
                    mc=int(mc),
                )
            hook_enabled["value"] = True

            for st_i in target_step_indices:
                h_chunks = batch_hidden_store[st_i]
                key = f"s{st_i}_l{target_layer_idx}"
                if h_chunks:
                    h_t = torch.cat(h_chunks, dim=0)
                    hidden_chunks_by_key[key].append(h_t)
                    del h_t
                reward_chunks_by_key[key].append(batch_reward_by_step[st_i].clone())

            del (
                latents,
                x,
                cls,
                cls_null,
                class_labels_input,
                batch_hidden_store,
                batch_reward_by_step,
                batch_snapshot_x,
                batch_snapshot_scheduler,
                gen,
                label_t,
            )
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        handle.remove()

    label_store = torch.cat(label_chunks, dim=0) if label_chunks else torch.empty(0, dtype=torch.long)
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for st_i in target_step_indices:
        key = f"s{st_i}_l{target_layer_idx}"
        hidden_t = torch.cat(hidden_chunks_by_key[key], dim=0) if hidden_chunks_by_key[key] else torch.empty(0)
        reward_t = (
            torch.cat(reward_chunks_by_key[key], dim=0) if reward_chunks_by_key[key] else torch.empty(0, dtype=torch.float32)
        )

        if hidden_t.shape[0] != reward_t.shape[0]:
            raise RuntimeError(f"{key}: rows mismatch hidden={hidden_t.shape[0]} reward={reward_t.shape[0]}")
        if hidden_t.shape[0] != label_store.shape[0]:
            raise RuntimeError(f"{key}: rows mismatch hidden={hidden_t.shape[0]} labels={label_store.shape[0]}")

        out_path = save_prefix.parent / f"{save_prefix.stem}_{key}.pt"
        torch.save(
            {
                "hiddens": hidden_t,  # (num_samples, hidden_dim)
                "reward": reward_t,  # (num_samples,)
                "labels": label_store,
                "meta": {
                    "model_name": str(model_name),
                    "target_step_index": int(st_i),
                    "target_layer_index": int(target_layer_idx),
                    "num_samples": int(num_samples),
                    "per_samples": int(per_samples),
                    "batch_size": int(batch_size),
                    "num_inference_steps": int(num_inference_steps),
                    "guidance_scale": float(guidance_scale),
                    "seed": int(seed),
                    "dtype": str(dtype),
                    "reward_note": "MC rollout + inception max softmax confidence",
                    "reward_mc": int(mc),
                    "hidden_pool": str(hidden_pool),
                },
            },
            out_path,
        )
        saved_paths.append(out_path)
        del hidden_t, reward_t
        gc.collect()
        torch.cuda.empty_cache()

    return saved_paths


def _pearsonr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.view(-1).float()
    y = y.view(-1).float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx.pow(2).sum() + eps) * (vy.pow(2).sum() + eps))
    return float((vx * vy).sum().item() / denom.item())


def train_prob_on_data(
    *,
    data_path: Path,
    output_dir: Path,
    seed: int,
    train_ratio: float,
    train_batch_size: int,
    hidden_dim: int,
    num_layers: int,
    dropout_prob: float,
    half: bool,
    optim_name: str,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    momentum: float,
    scheduler_name: str,
    t_max: int,
    eta_min: float,
    epochs: int,
    num_workers: int,
) -> dict:
    obj = torch.load(data_path, map_location="cpu", weights_only=False)
    hiddens = obj["hiddens"].float()
    rewards = (obj["reward"] if "reward" in obj else obj["rewards"]).view(-1).float()

    # check
    if hiddens.dim() != 2:
        raise ValueError(f"hiddens must be 2D, got {tuple(hiddens.shape)}")
    if hiddens.shape[0] != rewards.shape[0]:
        raise ValueError(f"rows mismatch: h={hiddens.shape[0]} r={rewards.shape[0]}")

    n = int(hiddens.shape[0])
    if n < 2:
        raise ValueError(f"need >=2 rows, got {n}")
    # check end

    # split train and val
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    n_train = min(max(int(round(n * float(train_ratio))), 1), n - 1)
    tr_idx = perm[:n_train]
    va_idx = perm[n_train:]

    tr_h = hiddens[tr_idx]
    va_h = hiddens[va_idx]
    tr_r_raw = rewards[tr_idx]
    va_r_raw = rewards[va_idx]

    # normalize reward
    r_mean = float(tr_r_raw.mean().item())
    r_std = float(tr_r_raw.std().item())
    denom = r_std + 1e-8
    tr_r = (tr_r_raw - r_mean) / denom
    va_r = (va_r_raw - r_mean) / denom

    tr_ds = HiddenRewardDataset(tr_h, tr_r)
    va_ds = HiddenRewardDataset(va_h, va_r)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=int(train_batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=int(train_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProbIS(
        input_dim=int(tr_h.shape[1]),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout_prob=float(dropout_prob),
        half=bool(half),
    ).to(device)

    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            betas=(float(beta1), float(beta2)),
            eps=float(eps),
        )
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
    else:
        raise NotImplementedError(f"unsupported optim_name={optim_name}")

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(t_max), 1),
            eta_min=float(eta_min),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise NotImplementedError(f"unsupported scheduler_name={scheduler_name}")

    loss_fn = nn.MSELoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "prob_is_best.pt"
    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1

    @torch.no_grad()
    def _eval():
        model.eval()
        total = 0.0
        preds, gts = [], []
        for x, y in va_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)
            pred = model(x)
            total += float(loss_fn(pred, y).item())
            preds.append(pred.detach().cpu().view(-1))
            gts.append(y.detach().cpu().view(-1))
        return total / max(1, len(va_loader)), torch.cat(preds), torch.cat(gts)

    for ep in range(int(epochs)):
        model.train()
        tr_total = 0.0
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_total += float(loss.item())
        tr_loss = tr_total / max(1, len(tr_loader))

        va_loss, _, _ = _eval()
        if scheduler is not None:
            scheduler.step()

        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"[{data_path.name}] epoch={ep:03d} train={tr_loss:.6f} val={va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = ep
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "val_loss": va_loss},
                best_ckpt,
            )

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()
    val_loss_norm, val_pred_norm, val_gt_norm = _eval()
    val_pred_raw = val_pred_norm * denom + r_mean
    val_gt_raw = val_gt_norm * denom + r_mean
    pearson = _pearsonr_torch(val_pred_raw, val_gt_raw)

    metrics = {
        "data_path": str(data_path),
        "best_ckpt": str(best_ckpt),
        "best_epoch": int(best_epoch),
        "best_val_loss_norm": float(best_val),
        "final_val_loss_norm": float(val_loss_norm),
        "val_pearson_raw": float(pearson),
        "reward_mean_train": float(r_mean),
        "reward_std_train": float(r_std),
        "num_train": int(tr_idx.numel()),
        "num_val": int(va_idx.numel()),
    }
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DiT hidden -> ProbIS (golden reward by MC + Inception confidence)")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/prob_is")
    p.add_argument("--rollout_path", type=str, default="")
    p.add_argument("--collect_only", action="store_true")

    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--per_samples", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--target_step_indices", type=str, default="1,6,9")
    p.add_argument("--target_layer_idx", type=int, default=16)
    p.add_argument("--hidden_pool", type=str, default="mean", choices=("mean", "last"))
    p.add_argument("--mc", type=int, default=8)

    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=300)

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    p.add_argument("--half", action="store_true", default=False)

    p.add_argument("--optim_name", type=str, default="adamw", choices=("adamw", "sgd"))
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--scheduler_name", type=str, default="cosine", choices=("cosine", "none"))
    p.add_argument("--t_max", type=int, default=100)
    p.add_argument("--eta_min", type=float, default=1e-6)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(int(args.seed))

    out_dir = Path(args.output_dir).expanduser().resolve()
    rollout_dir = out_dir / "rollout"
    train_root = out_dir / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.rollout_path).strip():
        rollout_files = [Path(args.rollout_path).expanduser().resolve()]
    else:
        target_steps = _parse_step_indices(str(args.target_step_indices))
        save_prefix = rollout_dir / "dit_hs_is"
        rollout_files = rollout_training_data(
            save_prefix=save_prefix,
            model_name=str(args.model_name),
            per_samples=int(args.per_samples),
            batch_size=int(args.batch_size),
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=float(args.guidance_scale),
            seed=int(args.seed),
            dtype=str(args.dtype),
            target_step_indices=target_steps,
            target_layer_idx=int(args.target_layer_idx),
            hidden_pool=str(args.hidden_pool),
            mc=int(args.mc),
        )
        print(json.dumps({"rollout_files": [str(x) for x in rollout_files]}, ensure_ascii=False, indent=2))

    if args.collect_only:
        return

    all_metrics = []
    for f in rollout_files:
        run_dir = train_root / f.stem
        metrics = train_prob_on_data(
            data_path=f,
            output_dir=run_dir,
            seed=int(args.seed),
            train_ratio=float(args.train_ratio),
            train_batch_size=int(args.train_batch_size),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            dropout_prob=float(args.dropout_prob),
            half=bool(args.half),
            optim_name=str(args.optim_name),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            beta1=float(args.beta1),
            beta2=float(args.beta2),
            eps=float(args.eps),
            momentum=float(args.momentum),
            scheduler_name=str(args.scheduler_name),
            t_max=int(args.t_max),
            eta_min=float(args.eta_min),
            epochs=int(args.epochs),
            num_workers=int(args.num_workers),
        )
        all_metrics.append(metrics)

    (out_dir / "summary_metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

