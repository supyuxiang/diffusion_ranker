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
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import dit_loader, set_seed

_DEFAULT_CLIP_DIR = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")
_LABELS_JSON = ROOT / "imagenet-simple-labels.json"


def _load_class_names() -> list[str]:
    labels = json.loads(_LABELS_JSON.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or len(labels) != 1000:
        raise ValueError(f"Invalid ImageNet labels at {_LABELS_JSON}")
    return [str(x) for x in labels]


def _clipscore_rewards(
    images: list[Image.Image],
    labels: torch.Tensor,
    clip: CLIPModel,
    processor: CLIPProcessor,
    class_names: list[str],
) -> torch.Tensor:
    prompts = [f"A photo of a {class_names[int(lab)]}" for lab in labels.tolist()]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(clip.device) for k, v in inputs.items()}
    out = clip(**inputs)
    image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text_features = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    sim = (image_features * text_features).sum(dim=-1)
    return sim.detach().float().cpu()


class ClipScoreProb(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int, dropout_prob: float = 0.1):
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
        self.reward_head = nn.Linear(hidden_dim, 1)
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


class ClipScoreDataset(Dataset):
    def __init__(self, hiddens: torch.Tensor, rewards: torch.Tensor):
        if hiddens.dim() != 2:
            raise ValueError(f"hiddens must be 2D, got {tuple(hiddens.shape)}")
        rewards = rewards.view(-1).float()
        if hiddens.shape[0] != rewards.shape[0]:
            raise ValueError(f"rows mismatch: hiddens={hiddens.shape[0]} rewards={rewards.shape[0]}")
        self.hiddens = hiddens.float()
        self.rewards = rewards

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
def _mc_clipscore_reward_from_snapshot(
    *,
    pipe,
    x_start: torch.Tensor,
    labels: torch.Tensor,
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
    class_names: list[str],
) -> torch.Tensor:
    if start_step_idx >= len(timesteps):
        imgs = _decode_dit_latents_to_pil(pipe, x_start, guidance_scale)
        return _clipscore_rewards(imgs, labels, clip, processor, class_names)

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
        rewards_ls.append(_clipscore_rewards(imgs, labels, clip, processor, class_names))

    return torch.stack(rewards_ls, dim=0).mean(dim=0)


@torch.inference_mode()
def collect_training_data(
    *,
    output_dir: Path,
    per_class: int = 10,
    target_step_indices: tuple[int, ...] = (1, 6, 9),
    target_layer_idx: int = 16,
    mc: int = 5,
    model_name: str = "dit-xl-2-256",
    dtype: str = "bf16",
    num_inference_steps: int = 10,
    batch_size: int = 32,
    guidance_scale: float = 1.0,
    seed: int = 42,
) -> dict[int, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(model_name, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    clip = CLIPModel.from_pretrained(str(_DEFAULT_CLIP_DIR), local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(_DEFAULT_CLIP_DIR), local_files_only=True)
    class_names = _load_class_names()

    labels_all = torch.tensor([c for c in range(1000) for _ in range(int(per_class))], dtype=torch.long)
    num_samples = int(labels_all.numel())
    if num_samples != 10000:
        raise ValueError(f"Expected 10k samples (1000*10), got {num_samples}")

    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels
    n_batches = (num_samples + int(batch_size) - 1) // int(batch_size)
    target_steps = tuple(sorted(set(int(s) for s in target_step_indices)))

    hidden_store: dict[int, list[torch.Tensor]] = {s: [] for s in target_steps}
    reward_store: dict[int, list[torch.Tensor]] = {s: [] for s in target_steps}

    current_step_idx = {"value": -1}
    current_batch_size = {"value": 0}
    current_store_ref: dict[str, dict[int, list[torch.Tensor]] | None] = {"store": None}

    def _hook(_module, _inp, out) -> None:
        s = current_step_idx["value"]
        store = current_store_ref["store"]
        if store is None or s not in store:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        if guidance_scale > 1:
            h = h[: current_batch_size["value"]]
        h = h.mean(dim=1).detach().to(torch.float16).cpu()
        store[s].append(h)

    handle = pipe.transformer.transformer_blocks[int(target_layer_idx)].register_forward_hook(_hook)
    try:
        pbar = tqdm(range(n_batches), desc="collect clipscore hidden/reward")
        for batch_i in pbar:
            st = batch_i * int(batch_size)
            ed = min(st + int(batch_size), num_samples)
            batch_n = ed - st
            if batch_n <= 0:
                break
            current_batch_size["value"] = batch_n
            batch_labels = labels_all[st:ed].clone()

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

            batch_hidden_store: dict[int, list[torch.Tensor]] = {s: [] for s in target_steps}
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

            for st_i in target_steps:
                reward_t = _mc_clipscore_reward_from_snapshot(
                    pipe=pipe,
                    x_start=batch_snapshot_x[st_i],
                    labels=batch_labels,
                    start_step_idx=st_i + 1,
                    timesteps=timesteps,
                    scheduler_snapshot=batch_snapshot_scheduler[st_i],
                    class_labels_input=class_labels_input,
                    latent_channels=latent_channels,
                    guidance_scale=float(guidance_scale),
                    seed_base=int(seed) + batch_i * 1000 + st_i * 100,
                    mc=int(mc),
                    clip=clip,
                    processor=processor,
                    class_names=class_names,
                )
                h_t = torch.cat(batch_hidden_store[st_i], dim=0)
                if h_t.shape[0] != reward_t.shape[0]:
                    raise RuntimeError(f"rows mismatch for step {st_i}: hidden={h_t.shape[0]} reward={reward_t.shape[0]}")
                hidden_store[st_i].append(h_t)
                reward_store[st_i].append(reward_t)

            del latents, x, cls, cls_null, class_labels_input
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    out_paths: dict[int, Path] = {}
    for st_i in target_steps:
        h = torch.cat(hidden_store[st_i], dim=0).float()
        r = torch.cat(reward_store[st_i], dim=0).float()
        out = output_dir / f"clipscore_hs_reward_s{st_i}_l{target_layer_idx}_per{per_class}_mc{mc}.pt"
        torch.save(
            {
                "hiddens": h,
                "reward": r,
                "labels": labels_all,
                "meta": {
                    "target_step_idx": int(st_i),
                    "target_layer_idx": int(target_layer_idx),
                    "per_class": int(per_class),
                    "num_samples": int(num_samples),
                    "mc": int(mc),
                    "reward_name": "clipscore",
                    "prompt_template": "A photo of a {class_name}",
                },
            },
            out,
        )
        out_paths[int(st_i)] = out
    return out_paths


def _pearsonr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.view(-1).float()
    y = y.view(-1).float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx.pow(2).sum() + eps) * (vy.pow(2).sum() + eps))
    return float((vx * vy).sum().item() / denom.item())


class ClipScoreTrainer:
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_data()
        self._build_model()

    def _load_pt_dataset(self, p: Path) -> ClipScoreDataset:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        return ClipScoreDataset(obj["hiddens"], obj["reward"])

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
            train_ds, val_ds = self._split_9_1(ds_train_full)
        else:
            train_ds, _ = self._split_9_1(ds_train_full)
            _, val_ds = self._split_9_1(ds_val_full)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
        x0, _ = ds_train_full[0]
        self.input_dim = int(x0.numel())

    def _build_model(self):
        self.model = ClipScoreProb(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            dropout_prob=self.dropout_prob,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = nn.MSELoss()
        self.best_ckpt = self.output_dir / "clipscore_predictor_best.pt"

    def _run_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).view(-1, 1)
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
            y = y.to(self.device, non_blocking=True).view(-1, 1)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            total += float(loss.detach().item())
            preds.append(pred.detach().cpu().view(-1))
            gts.append(y.detach().cpu().view(-1))
        return total / max(1, len(self.val_loader)), torch.cat(preds), torch.cat(gts)

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
        pearson = _pearsonr_torch(pred, gt)
        out = {
            "best_ckpt": str(self.best_ckpt),
            "best_val_loss_during_training": float(best_val),
            "final_val_loss_with_best_ckpt": float(val_loss),
            "val_pearson": float(pearson),
            "num_val_samples": int(gt.numel()),
            "train_data": str(self.train_path),
            "val_data": str(self.val_path),
            "epochs": int(self.num_epochs),
        }
        (self.output_dir / "train_history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "final_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ClipScore predictor data collection + training")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/clipscore_predictor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--collect_batch_size", type=int, default=32)
    p.add_argument("--per_class", type=int, default=10)
    p.add_argument("--mc", type=int, default=5)
    p.add_argument("--target_layer_idx", type=int, default=16)
    p.add_argument("--target_step_indices", type=str, default="1,6,9")
    p.add_argument("--train_step_idx", type=int, default=6)
    p.add_argument("--val_step_idx", type=int, default=6)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    step_indices = tuple(int(s) for s in str(args.target_step_indices).split(",") if s.strip())
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_dir = output_dir / "data"
    run_dir = output_dir / "train_run"
    data_paths = collect_training_data(
        output_dir=data_dir,
        per_class=int(args.per_class),
        target_step_indices=step_indices,
        target_layer_idx=int(args.target_layer_idx),
        mc=int(args.mc),
        model_name=str(args.model_name),
        dtype=str(args.dtype),
        num_inference_steps=int(args.num_inference_steps),
        batch_size=int(args.collect_batch_size),
        guidance_scale=float(args.guidance_scale),
        seed=int(args.seed),
    )
    trainer = ClipScoreTrainer(
        train_path=data_paths[int(args.train_step_idx)],
        val_path=data_paths[int(args.val_step_idx)],
        output_dir=run_dir,
        seed=int(args.seed),
        batch_size=int(args.train_batch_size),
        num_epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout_prob=float(args.dropout_prob),
    )
    result = trainer.train_and_test()
    result["collected_paths"] = {str(k): str(v) for k, v in data_paths.items()}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

