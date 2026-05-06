#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import dit_loader, set_seed

DEFAULT_STEP_IDXS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
DEFAULT_LAYER_IDXS = (2, 5, 7, 16, 19, 21, 24)


def rgb_reward_single(image: Image.Image) -> torch.Tensor:
    """
    Return per-image mean RGB in 0~255 scale.
    Shape: (3,)
    """
    img = image.convert("RGB")
    x = np.asarray(img, dtype=np.float32)  # H,W,3
    rgb = x.reshape(-1, 3).mean(axis=0) # (3,)
    return torch.tensor(rgb, dtype=torch.float32)


def _select_channel(rgb: torch.Tensor, channel: str) -> torch.Tensor:
    """
    Select target from RGB tensor.
    rgb shape: (..., 3)
    return:
      - channel='rgb' -> (..., 3)
      - channel in {'r','g','b'} -> (..., 1)
    """
    c = str(channel).lower().strip()
    if c == "rgb":
        return rgb
    idx_map = {"r": 0, "g": 1, "b": 2}
    if c not in idx_map:
        raise ValueError(f"Invalid channel={channel!r}; choose from rgb/r/g/b")
    idx = idx_map[c]
    if rgb.dim() == 1:
        return rgb[idx : idx + 1]
    return rgb[..., idx : idx + 1]


class RGBProb(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, hidden_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                )
                for _ in range(num_layers)
            ]
        )
        self.reward_head = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

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


class RGBDataset(Dataset):
    def __init__(self, hiddens: torch.Tensor, rewards: torch.Tensor):
        if hiddens.dim() != 2:
            raise ValueError(f"hiddens must be 2D, got {tuple(hiddens.shape)}")
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if rewards.dim() != 2:
            raise ValueError(f"rewards must be 1D or 2D, got {tuple(rewards.shape)}")
        if hiddens.shape[0] != rewards.shape[0]:
            raise ValueError(f"rows mismatch: hiddens={hiddens.shape[0]} rewards={rewards.shape[0]}")
        self.hiddens = hiddens.float()
        self.rewards = rewards.float()

    def __len__(self) -> int:
        return int(self.hiddens.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hiddens[idx], self.rewards[idx]


def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        dt = torch.float64 if isinstance(t, float) else torch.int64
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
    for bi in range(samples.shape[0]):
        arr = samples[bi].detach().cpu().permute(1, 2, 0).float().numpy()
        images.append(Image.fromarray((arr * 255.0).round().clip(0, 255).astype("uint8")))
    return images


@torch.inference_mode()
def _mc_rgb_reward_from_snapshot(
    *,
    pipe,
    x_start: torch.Tensor,
    start_step_idx: int,
    timesteps: torch.Tensor,
    scheduler_snapshot,
    class_labels_input: torch.Tensor,
    latent_channels: int,
    guidance_scale: float,
    seed_base: int,
    mc: int,
) -> torch.Tensor:
    if start_step_idx >= len(timesteps):
        imgs = _decode_dit_latents_to_pil(pipe, x_start, guidance_scale)
        return torch.stack([rgb_reward_single(im) for im in imgs], dim=0)  # (B,3)

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

        imgs = _decode_dit_latents_to_pil(pipe, x, guidance_scale)
        rewards_ls.append(torch.stack([rgb_reward_single(im) for im in imgs], dim=0))  # (B,3)

    return torch.stack(rewards_ls, dim=0).mean(dim=0)  # (B,3)


@torch.inference_mode()
def collect_training_data(
    *,
    output_dir: Path,
    num_samples_per_setting: int = 5000,
    target_step_indices: tuple[int, ...] = DEFAULT_STEP_IDXS,
    target_layer_indices: tuple[int, ...] = DEFAULT_LAYER_IDXS,
    mc: int = 5,
    model_name: str = "dit-xl-2-256",
    dtype: str = "bf16",
    num_inference_steps: int = 10,
    batch_size: int = 32,
    guidance_scale: float = 1.0,
    seed: int = 42,
    channel: str = "rgb",
) -> dict[tuple[int, int], Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(model_name, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels
    n_batches = (int(num_samples_per_setting) + int(batch_size) - 1) // int(batch_size)
    target_steps = tuple(sorted(set(int(s) for s in target_step_indices)))
    target_layers = tuple(sorted(set(int(l) for l in target_layer_indices)))

    hidden_store: dict[tuple[int, int], list[torch.Tensor]] = {
        (layer_idx, step_idx): []
        for layer_idx in target_layers
        for step_idx in target_steps
    }
    reward_store: dict[tuple[int, int], list[torch.Tensor]] = {
        (layer_idx, step_idx): []
        for layer_idx in target_layers
        for step_idx in target_steps
    }
    label_store: list[torch.Tensor] = []

    current_step_idx = {"value": -1}
    current_batch_size = {"value": 0}
    current_store_ref: dict[str, dict[tuple[int, int], list[torch.Tensor]] | None] = {"store": None}

    def _make_hook(layer_idx: int):
        def _hook(_module, _inp, out) -> None:
            s = current_step_idx["value"]
            store = current_store_ref["store"]
            if store is None or s not in target_steps:
                return
            key = (int(layer_idx), int(s))
            if key not in store:
                return
            h = out[0] if isinstance(out, (tuple, list)) else out
            if guidance_scale > 1:
                h = h[: current_batch_size["value"]]
            h = h.mean(dim=1).detach().to(torch.float16).cpu()
            store[key].append(h)

        return _hook

    handles = [
        pipe.transformer.transformer_blocks[int(layer_idx)].register_forward_hook(_make_hook(int(layer_idx)))
        for layer_idx in target_layers
    ]
    try:
        pbar = tqdm(range(n_batches), desc="collect rgb hidden/reward (all layers+steps)")
        for batch_i in pbar:
            st = batch_i * int(batch_size)
            ed = min(st + int(batch_size), int(num_samples_per_setting))
            batch_n = ed - st
            if batch_n <= 0:
                break
            current_batch_size["value"] = batch_n

            g_cls = torch.Generator().manual_seed(int(seed) + 1234 + batch_i)
            batch_labels = torch.randint(0, 1000, (batch_n,), generator=g_cls, dtype=torch.long)
            label_store.append(batch_labels.clone())

            gen = torch.Generator(device=device).manual_seed(int(seed) + batch_i)
            latents = torch.randn(
                (batch_n, latent_channels, latent_size, latent_size),
                generator=gen,
                device=device,
                dtype=pipe.transformer.dtype,
            )
            x = torch.cat([latents] * 2, dim=0) if guidance_scale > 1 else latents

            cls = batch_labels.to(device=device)
            cls_null = torch.full((batch_n,), 1000, device=device, dtype=torch.long)
            class_labels_input = torch.cat([cls, cls_null], dim=0) if guidance_scale > 1 else cls

            pipe.scheduler.set_timesteps(int(num_inference_steps))
            timesteps = pipe.scheduler.timesteps

            batch_hidden_store: dict[tuple[int, int], list[torch.Tensor]] = {
                (layer_idx, step_idx): []
                for layer_idx in target_layers
                for step_idx in target_steps
            }
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

                if st_i in target_steps:
                    batch_snapshot_x[st_i] = x.detach().clone()
                    batch_snapshot_scheduler[st_i] = copy.deepcopy(pipe.scheduler)

            current_store_ref["store"] = None

            batch_reward_by_step: dict[int, torch.Tensor] = {}
            for st_i in target_steps:
                batch_reward_by_step[st_i] = _mc_rgb_reward_from_snapshot(
                    pipe=pipe,
                    x_start=batch_snapshot_x[st_i],
                    start_step_idx=st_i + 1,
                    timesteps=timesteps,
                    scheduler_snapshot=batch_snapshot_scheduler[st_i],
                    class_labels_input=class_labels_input,
                    latent_channels=latent_channels,
                    guidance_scale=float(guidance_scale),
                    seed_base=int(seed) + batch_i * 1000 + st_i * 100,
                    mc=int(mc),
                )
            for layer_idx in target_layers:
                for st_i in target_steps:
                    reward_t = _select_channel(batch_reward_by_step[st_i], channel)
                    key = (int(layer_idx), int(st_i))
                    h_chunks = batch_hidden_store[key]
                    if not h_chunks:
                        raise RuntimeError(f"hidden chunks missing for layer={layer_idx}, step={st_i}")
                    h_t = torch.cat(h_chunks, dim=0)
                    if h_t.shape[0] != reward_t.shape[0]:
                        raise RuntimeError(
                            f"rows mismatch for layer={layer_idx}, step={st_i}: hidden={h_t.shape[0]} reward={reward_t.shape[0]}"
                        )
                    hidden_store[key].append(h_t)
                    reward_store[key].append(reward_t)

            del latents, x, cls, cls_null, class_labels_input
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    labels = torch.cat(label_store, dim=0)
    out_paths: dict[tuple[int, int], Path] = {}
    for layer_idx in target_layers:
        for st_i in target_steps:
            key = (int(layer_idx), int(st_i))
            h = torch.cat(hidden_store[key], dim=0).float()
            r = torch.cat(reward_store[key], dim=0).float()
            out = output_dir / f"rgb_{channel}_hs_reward_s{st_i}_l{layer_idx}_n{num_samples_per_setting}_mc{mc}.pt"
            torch.save(
                {
                    "hiddens": h,
                    "reward": r,
                    "labels": labels,
                    "meta": {
                        "target_step_idx": int(st_i),
                        "target_layer_idx": int(layer_idx),
                        "num_samples": int(num_samples_per_setting),
                        "mc": int(mc),
                        "reward_name": "rgb_mean",
                        "target_channel": str(channel),
                    },
                },
                out,
            )
            out_paths[key] = out
    return out_paths


def _pearsonr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.view(-1).float()
    y = y.view(-1).float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx.pow(2).sum() + eps) * (vy.pow(2).sum() + eps))
    return float((vx * vy).sum().item() / denom.item())


class RGBTrainer:
    def __init__(
        self,
        *,
        train_path: Path,
        val_path: Path,
        output_dir: Path,
        seed: int = 42,
        batch_size: int = 64,
        num_epochs: int = 100,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_prob: float = 0.1,
        channel: str = "rgb",
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_prob = float(dropout_prob)
        self.channel = str(channel).lower().strip()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_data()
        self._build_model()

    def _load_pt_dataset(self, p: Path) -> RGBDataset:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        return RGBDataset(obj["hiddens"], obj["reward"])

    def _split_9_1(self, ds: Dataset) -> tuple[Dataset, Dataset]:
        n = len(ds)
        n_train = min(max(int(round(0.9 * n)), 1), n - 1)
        n_val = n - n_train
        g = torch.Generator().manual_seed(self.seed)
        return random_split(ds, [n_train, n_val], generator=g)

    def _build_data(self):
        ds_train_full = self._load_pt_dataset(self.train_path)
        ds_val_full = self._load_pt_dataset(self.val_path)
        if self.train_path.resolve() == self.val_path.resolve():
            train_subset, val_subset = self._split_9_1(ds_train_full)
            train_indices = torch.tensor(train_subset.indices, dtype=torch.long)
            val_indices = torch.tensor(val_subset.indices, dtype=torch.long)
            train_h = ds_train_full.hiddens[train_indices]
            val_h = ds_train_full.hiddens[val_indices]
            train_r_raw = ds_train_full.rewards[train_indices]
            val_r_raw = ds_train_full.rewards[val_indices]
        else:
            train_subset, _ = self._split_9_1(ds_train_full)
            _, val_subset = self._split_9_1(ds_val_full)
            train_indices = torch.tensor(train_subset.indices, dtype=torch.long)
            val_indices = torch.tensor(val_subset.indices, dtype=torch.long)
            train_h = ds_train_full.hiddens[train_indices]
            val_h = ds_val_full.hiddens[val_indices]
            train_r_raw = ds_train_full.rewards[train_indices]
            val_r_raw = ds_val_full.rewards[val_indices]

        # Channel-wise normalization using train split stats.
        self.reward_mean = train_r_raw.mean(dim=0)
        self.reward_std = train_r_raw.std(dim=0)
        denom = self.reward_std + 1e-8
        train_r = (train_r_raw - self.reward_mean) / denom
        val_r = (val_r_raw - self.reward_mean) / denom

        train_ds = RGBDataset(train_h, train_r)
        val_ds = RGBDataset(val_h, val_r)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.reward_dim = int(train_r.shape[1])
        x0, _ = ds_train_full[0]
        self.input_dim = int(x0.numel())

    def _build_model(self):
        self.model = RGBProb(
            input_dim=self.input_dim,
            output_dim=self.reward_dim,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            dropout_prob=self.dropout_prob,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = nn.MSELoss()
        self.best_ckpt = self.output_dir / "rgb_predictor_best.pt"

    def _run_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach().item())
        return total / max(1, len(self.train_loader))

    @torch.no_grad()
    def _eval_loss_and_preds(self) -> tuple[float, torch.Tensor, torch.Tensor]:
        self.model.eval()
        total = 0.0
        preds, gts = [], []
        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            total += float(loss.detach().item())
            preds.append(pred.detach().cpu())
            gts.append(y.detach().cpu())
        return total / max(1, len(self.val_loader)), torch.cat(preds, dim=0), torch.cat(gts, dim=0)

    def train_and_test(self) -> dict:
        best_val = float("inf")
        hist = []
        for ep in range(self.num_epochs):
            tr = self._run_epoch()
            va, _, _ = self._eval_loss_and_preds()
            hist.append({"epoch": ep, "train_loss": tr, "val_loss": va})
            print(f"epoch {ep:03d} | train_loss={tr:.6f} | val_loss={va:.6f}")
            if va < best_val:
                best_val = va
                torch.save({"model": self.model.state_dict(), "epoch": ep, "val_loss": va}, self.best_ckpt)

        ckpt = torch.load(self.best_ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(self.device).eval()
        val_loss, pred, gt = self._eval_loss_and_preds()
        out = {
            "best_ckpt": str(self.best_ckpt),
            "best_val_loss_during_training": float(best_val),
            "final_val_loss_with_best_ckpt": float(val_loss),
            "num_val_samples": int(gt.shape[0]),
            "train_data": str(self.train_path),
            "val_data": str(self.val_path),
            "epochs": int(self.num_epochs),
            "reward_norm_mean": [float(v) for v in self.reward_mean.tolist()],
            "reward_norm_std": [float(v) for v in self.reward_std.tolist()],
            "target_channel": str(self.channel),
        }
        if self.reward_dim == 3:
            p_r = _pearsonr_torch(pred[:, 0], gt[:, 0])
            p_g = _pearsonr_torch(pred[:, 1], gt[:, 1])
            p_b = _pearsonr_torch(pred[:, 2], gt[:, 2])
            out["val_pearson_r"] = float(p_r)
            out["val_pearson_g"] = float(p_g)
            out["val_pearson_b"] = float(p_b)
            out["val_pearson_mean"] = float((p_r + p_g + p_b) / 3.0)
        else:
            out["val_pearson"] = float(_pearsonr_torch(pred[:, 0], gt[:, 0]))
        (self.output_dir / "train_history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "final_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("RGB predictor data collection + training")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/rgb_predictor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--collect_batch_size", type=int, default=32)
    p.add_argument("--num_samples_per_setting", type=int, default=5000)
    p.add_argument("--mc", type=int, default=5)
    p.add_argument("--target_layer_indices", type=str, default="2,5,7,16,19,21,24")
    p.add_argument("--target_step_indices", type=str, default="1,2,3,4,5,6,7,8,9")
    p.add_argument("--channel", type=str, default="rgb", choices=("rgb", "r", "g", "b"))
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    step_indices = tuple(sorted(set(int(s) for s in str(args.target_step_indices).split(",") if s.strip())))
    layer_indices = tuple(sorted(set(int(s) for s in str(args.target_layer_indices).split(",") if s.strip())))
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_dir = output_dir / "data"
    run_root = output_dir / "train_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    data_paths = collect_training_data(
        output_dir=data_dir,
        num_samples_per_setting=int(args.num_samples_per_setting),
        target_step_indices=step_indices,
        target_layer_indices=layer_indices,
        mc=int(args.mc),
        model_name=str(args.model_name),
        dtype=str(args.dtype),
        num_inference_steps=int(args.num_inference_steps),
        batch_size=int(args.collect_batch_size),
        guidance_scale=float(args.guidance_scale),
        seed=int(args.seed),
        channel=str(args.channel),
    )
    results: list[dict] = []
    for layer_idx in layer_indices:
        for step_idx in step_indices:
            key = (int(layer_idx), int(step_idx))
            train_path = data_paths[key]
            run_dir = run_root / f"l{layer_idx}_s{step_idx}"
            trainer = RGBTrainer(
                train_path=train_path,
                val_path=train_path,
                output_dir=run_dir,
                seed=int(args.seed),
                batch_size=int(args.train_batch_size),
                num_epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                hidden_dim=int(args.hidden_dim),
                num_layers=int(args.num_layers),
                dropout_prob=float(args.dropout_prob),
                channel=str(args.channel),
            )
            m = trainer.train_and_test()
            m["layer_idx"] = int(layer_idx)
            m["step_idx"] = int(step_idx)
            if "val_pearson_mean" in m:
                m["score_for_grid"] = float(m["val_pearson_mean"])
            else:
                m["score_for_grid"] = float(m["val_pearson"])
            results.append(m)

    summary_json = output_dir / "grid_metrics.json"
    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = output_dir / "grid_metrics.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "layer_idx",
            "step_idx",
            "target_channel",
            "score_for_grid",
            "best_val_loss_during_training",
            "best_ckpt",
            "train_data",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["layer_idx"], x["step_idx"])):
            w.writerow(
                {
                    "layer_idx": r["layer_idx"],
                    "step_idx": r["step_idx"],
                    "target_channel": r.get("target_channel", ""),
                    "score_for_grid": r["score_for_grid"],
                    "best_val_loss_during_training": r["best_val_loss_during_training"],
                    "best_ckpt": r["best_ckpt"],
                    "train_data": r["train_data"],
                }
            )

    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "summary_csv": str(summary_csv),
                "num_settings": len(results),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()






















