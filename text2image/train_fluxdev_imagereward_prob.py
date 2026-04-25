#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from utils import ensure_image_reward_on_path, set_seed

_DEFAULT_IMAGEREWARD_DIR = Path("/root/autodl-tmp/ImageReward")
_DEFAULT_IMAGEREWARD_CKPT = _DEFAULT_IMAGEREWARD_DIR / "ImageReward.pt"
_DEFAULT_IMAGEREWARD_MED = _DEFAULT_IMAGEREWARD_DIR / "med_config.json"


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_imagereward(
    device: torch.device,
    ckpt_path: Path = _DEFAULT_IMAGEREWARD_CKPT,
    med_config: Path = _DEFAULT_IMAGEREWARD_MED,
):
    ensure_image_reward_on_path()
    import ImageReward as RM
    from transformers import BertTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _ = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    _orig = BertTokenizer.from_pretrained

    def _offline_from_pretrained(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        return _orig(*args, **kwargs)

    BertTokenizer.from_pretrained = _offline_from_pretrained
    model = RM.load(str(ckpt_path), device=device, med_config=str(med_config))
    model.eval()
    return model


def _imagereward_batch(images: list[Image.Image], prompt: str, model) -> torch.Tensor:
    vals = [float(model.score(prompt, img)) for img in images]
    return torch.tensor(vals, dtype=torch.float32)


def _decode_flux_latents_to_pil(
    pipe: FluxPipeline,
    latents_packed: torch.Tensor,
    height: int,
    width: int,
) -> list[Image.Image]:
    latents = pipe._unpack_latents(latents_packed, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def _setup_flux_timesteps(pipe: FluxPipeline, latents: torch.Tensor, num_inference_steps: int, device: torch.device):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    return timesteps


@torch.inference_mode()
def _mc_rollout_imagereward_flux(
    *,
    pipe: FluxPipeline,
    latents_start: torch.Tensor,
    start_step_idx: int,
    timesteps: torch.Tensor,
    scheduler_snapshot,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    guidance: torch.Tensor | None,
    do_true_cfg: bool,
    true_cfg_scale: float,
    negative_prompt_embeds: torch.Tensor | None,
    negative_pooled_prompt_embeds: torch.Tensor | None,
    negative_text_ids: torch.Tensor | None,
    seed_base: int,
    mc: int,
    prompt: str,
    rm_model,
    height: int,
    width: int,
) -> torch.Tensor:
    if start_step_idx >= len(timesteps):
        imgs = _decode_flux_latents_to_pil(pipe, latents_start, height=height, width=width)
        return _imagereward_batch(imgs, prompt=prompt, model=rm_model)

    rewards = []
    for i in range(int(mc)):
        scheduler = copy.deepcopy(scheduler_snapshot)
        latents = latents_start.detach().clone()

        for step_i in range(int(start_step_idx), int(len(timesteps))):
            t = timesteps[step_i]
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=pipe.joint_attention_kwargs,
                    return_dict=False,
                )[0]

            if do_true_cfg:
                with pipe.transformer.cache_context("uncond"):
                    neg_noise_pred = pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=pipe.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        imgs = _decode_flux_latents_to_pil(pipe, latents, height=height, width=width)
        rewards.append(_imagereward_batch(imgs, prompt=prompt, model=rm_model))

    return torch.stack(rewards, dim=0).mean(dim=0)


class RewardProb(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout_prob: float = 0.1):
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

    def _init_weights(self) -> None:
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


class HiddenRewardDataset(Dataset):
    def __init__(self, hiddens: torch.Tensor, rewards: torch.Tensor):
        self.hiddens = hiddens.float()
        self.rewards = rewards.view(-1).float()
        if self.hiddens.dim() != 2:
            raise ValueError(f"hiddens must be 2D, got {tuple(self.hiddens.shape)}")
        if self.hiddens.shape[0] != self.rewards.shape[0]:
            raise ValueError(f"row mismatch: {self.hiddens.shape[0]} vs {self.rewards.shape[0]}")

    def __len__(self) -> int:
        return int(self.hiddens.shape[0])

    def __getitem__(self, idx: int):
        return self.hiddens[idx], self.rewards[idx]


def _pearsonr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.view(-1).float()
    y = y.view(-1).float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx.pow(2).sum() + eps) * (vy.pow(2).sum() + eps))
    return float((vx * vy).sum().item() / denom.item())


@torch.inference_mode()
def collect_hidden_and_reward(
    *,
    model_path: str,
    prompt: str,
    negative_prompt: str,
    output_dir: Path,
    num_samples: int = 1000,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    true_cfg_scale: float = 1.0,
    target_step_idx: int = 14,
    target_block_type: str = "dual",
    target_layer_idx: int = 9,
    mc: int = 4,
    collect_batch_size: int = 2,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    dtype_name: str = "bf16",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _resolve_dtype(dtype_name, device)

    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    rm_model = load_imagereward(device=device)

    dual_blocks = pipe.transformer.transformer_blocks
    single_blocks = pipe.transformer.single_transformer_blocks
    if target_block_type == "dual":
        n_blocks = len(dual_blocks)
        if not (0 <= target_layer_idx < n_blocks):
            raise ValueError(f"target_layer_idx must in [0, {n_blocks - 1}] for dual blocks")
        target_block = dual_blocks[target_layer_idx]
    elif target_block_type == "single":
        n_blocks = len(single_blocks)
        if not (0 <= target_layer_idx < n_blocks):
            raise ValueError(f"target_layer_idx must in [0, {n_blocks - 1}] for single blocks")
        target_block = single_blocks[target_layer_idx]
    else:
        raise ValueError("target_block_type must be 'dual' or 'single'")

    if not (0 <= target_step_idx < num_inference_steps):
        raise ValueError(f"target_step_idx must in [0, {num_inference_steps - 1}]")

    all_hidden: list[torch.Tensor] = []
    all_reward: list[torch.Tensor] = []
    all_seed: list[int] = []

    cur_step = {"value": -1}
    hidden_buf: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out) -> None:
        if cur_step["value"] != target_step_idx:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        hidden_buf["h"] = h.mean(dim=1).detach().float().cpu()

    handle = target_block.register_forward_hook(_hook)
    try:
        for start in tqdm(range(0, num_samples, collect_batch_size), desc="collect"):
            batch_n = min(collect_batch_size, num_samples - start)
            batch_seed = seed + start
            generator = torch.Generator(device=device).manual_seed(batch_seed)

            prompts = [prompt] * batch_n
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )

            do_true_cfg = true_cfg_scale > 1.0 and len(negative_prompt.strip()) > 0
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            negative_text_ids = None
            if do_true_cfg:
                negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                    prompt=[negative_prompt] * batch_n,
                    prompt_2=None,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                )

            num_channels_latents = pipe.transformer.config.in_channels // 4
            latents, latent_image_ids = pipe.prepare_latents(
                batch_n,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents=None,
            )
            timesteps = _setup_flux_timesteps(pipe, latents, int(num_inference_steps), device=device)
            pipe.scheduler.set_begin_index(0)

            if pipe.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
            else:
                guidance = None

            snapshot_latents = None
            snapshot_scheduler = None
            hidden_buf.clear()

            for step_i, t in enumerate(timesteps):
                cur_step["value"] = step_i
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with pipe.transformer.cache_context("cond"):
                    noise_pred = pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=pipe.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with pipe.transformer.cache_context("uncond"):
                        neg_noise_pred = pipe.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=pipe.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if step_i == target_step_idx:
                    snapshot_latents = latents.detach().clone()
                    snapshot_scheduler = copy.deepcopy(pipe.scheduler)

            if "h" not in hidden_buf:
                raise RuntimeError(f"Failed to capture hidden at target step {target_step_idx}")
            if snapshot_latents is None or snapshot_scheduler is None:
                raise RuntimeError(f"Failed to snapshot latent/scheduler at target step {target_step_idx}")

            reward = _mc_rollout_imagereward_flux(
                pipe=pipe,
                latents_start=snapshot_latents,
                start_step_idx=target_step_idx + 1,
                timesteps=timesteps,
                scheduler_snapshot=snapshot_scheduler,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                latent_image_ids=latent_image_ids,
                guidance=guidance,
                do_true_cfg=do_true_cfg,
                true_cfg_scale=float(true_cfg_scale),
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                negative_text_ids=negative_text_ids,
                seed_base=int(seed) + start * 100 + target_step_idx * 13,
                mc=int(mc),
                prompt=prompt,
                rm_model=rm_model,
                height=height,
                width=width,
            )

            all_hidden.append(hidden_buf["h"])
            all_reward.append(reward)
            all_seed.extend([batch_seed + i for i in range(batch_n)])

            del latents, snapshot_latents, snapshot_scheduler, prompt_embeds, pooled_prompt_embeds
            del text_ids, latent_image_ids, reward
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    hiddens = torch.cat(all_hidden, dim=0).float()
    rewards = torch.cat(all_reward, dim=0).float()
    seeds = torch.tensor(all_seed, dtype=torch.long)

    if hiddens.shape[0] != num_samples or rewards.shape[0] != num_samples:
        raise RuntimeError(
            f"Unexpected rows, hidden={hiddens.shape[0]}, reward={rewards.shape[0]}, expected={num_samples}"
        )

    out_pt = (
        output_dir
        / f"fluxdev_ir_hs_reward_n{num_samples}_s{target_step_idx}_{target_block_type}{target_layer_idx}_mc{mc}.pt"
    )
    torch.save(
        {
            "hiddens": hiddens,
            "reward": rewards,
            "seed": seeds,
            "meta": {
                "model_path": str(model_path),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_samples": int(num_samples),
                "num_inference_steps": int(num_inference_steps),
                "target_step_idx": int(target_step_idx),
                "target_block_type": str(target_block_type),
                "target_layer_idx": int(target_layer_idx),
                "mc": int(mc),
                "guidance_scale": float(guidance_scale),
                "true_cfg_scale": float(true_cfg_scale),
                "height": int(height),
                "width": int(width),
            },
        },
        out_pt,
    )
    return out_pt


def train_prob_and_eval(
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
    val_interval: int = 1,
) -> dict:
    obj = torch.load(data_pt, map_location="cpu", weights_only=False)
    hiddens = obj["hiddens"].float()
    rewards_raw = obj["reward"].view(-1).float()
    n = int(hiddens.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    g = torch.Generator().manual_seed(seed)
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

    train_ds = HiddenRewardDataset(train_h, train_r)
    val_ds = HiddenRewardDataset(val_h, val_r)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=train_batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardProb(
        input_dim=int(train_h.shape[1]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "prob_best.pt"
    history = []
    best_val = float("inf")
    best_epoch = -1

    @torch.no_grad()
    def _eval_val():
        model.eval()
        total = 0.0
        preds = []
        gts = []
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

        if ep % int(val_interval) == 0:
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
    val_pearson = _pearsonr_torch(val_pred_raw, val_gt_raw)

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
        "val_interval": int(val_interval),
        "reward_mean_train": float(r_mean),
        "reward_std_train": float(r_std),
    }
    (output_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("FLUX.1-dev hidden -> ImageReward prob trainer")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/FLUX.1-dev")
    p.add_argument("--prompt", type=str, required=True, help="Unique prompt for all generated samples.")
    p.add_argument("--negative_prompt", type=str, default="blurry, low quality, watermark, text, deformed")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/text2image_fluxdev_ir_prob")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--collect_batch_size", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--true_cfg_scale", type=float, default=1.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--target_step_idx", type=int, default=14)
    p.add_argument("--target_block_type", type=str, default="dual", choices=("dual", "single"))
    p.add_argument("--target_layer_idx", type=int, default=9)
    p.add_argument("--mc", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--prob_num_layers", type=int, default=2)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    output_dir = Path(args.output_dir).expanduser().resolve()
    data_dir = output_dir / "data"
    run_dir = output_dir / "train_run"

    data_pt = collect_hidden_and_reward(
        model_path=str(args.model_path),
        prompt=str(args.prompt),
        negative_prompt=str(args.negative_prompt),
        output_dir=data_dir,
        num_samples=int(args.num_samples),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        true_cfg_scale=float(args.true_cfg_scale),
        target_step_idx=int(args.target_step_idx),
        target_block_type=str(args.target_block_type),
        target_layer_idx=int(args.target_layer_idx),
        mc=int(args.mc),
        collect_batch_size=int(args.collect_batch_size),
        height=int(args.height),
        width=int(args.width),
        seed=int(args.seed),
        dtype_name=str(args.dtype),
    )

    result = train_prob_and_eval(
        data_pt=data_pt,
        output_dir=run_dir,
        seed=int(args.seed),
        train_ratio=0.9,
        train_batch_size=int(args.train_batch_size),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.prob_num_layers),
        dropout_prob=float(args.dropout_prob),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_interval=int(args.val_interval),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
