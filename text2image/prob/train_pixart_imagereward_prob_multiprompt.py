#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from diffusers import PixArtAlphaPipeline
from utils import ensure_image_reward_on_path, set_seed

from text2image.utils import _resolve_dtype

_DEFAULT_IMAGEREWARD_DIR = Path("/root/autodl-tmp/ImageReward")
_DEFAULT_IMAGEREWARD_CKPT = _DEFAULT_IMAGEREWARD_DIR / "ImageReward.pt"
_DEFAULT_IMAGEREWARD_MED = _DEFAULT_IMAGEREWARD_DIR / "med_config.json"
_DEFAULT_VQASCORE_PROMPTS = Path("/root/diffusion_ranker/t2v_metrics-main/prompts/genai_image_1600.json")



def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        dt = torch.float64 if isinstance(t, float) else torch.int64
        t = torch.tensor([t], dtype=dt, device=latent_model_input.device)
    elif len(t.shape) == 0:
        t = t[None].to(latent_model_input.device)
    return t.expand(latent_model_input.shape[0])


def _build_added_cond_kwargs(
    *,
    height: int,
    width: int,
    batch: int,
    do_cfg: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    resolution = torch.tensor([height, width], device=device, dtype=dtype).repeat(batch, 1)
    aspect_ratio = torch.tensor([float(height / width)], device=device, dtype=dtype).repeat(batch, 1)
    if do_cfg:
        resolution = torch.cat([resolution, resolution], dim=0)
        aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)
    return {"resolution": resolution, "aspect_ratio": aspect_ratio}


def _decode_pixart_latents_to_pil(pipe: PixArtAlphaPipeline, latents: torch.Tensor) -> list[Image.Image]:
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image


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


def _load_prompts(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[str] = []
    if isinstance(obj, dict):
        # COCO-style captions json: {"annotations":[{"caption": "..."}], ...}
        if isinstance(obj.get("annotations"), list):
            for i, row in enumerate(obj["annotations"]):
                if not isinstance(row, dict):
                    continue
                p = str(row.get("caption", "")).strip()
                if p:
                    prompts.append(p)
            if len(prompts) == 0:
                raise ValueError(f"No valid caption found in annotations of {path}")
        else:
            # VQAScore-style json: {"00000": {"prompt": "..."}}
            for k in sorted(obj.keys()):
                row = obj[k]
                if not isinstance(row, dict):
                    continue
                p = str(row.get("prompt", "")).strip()
                if p:
                    prompts.append(p)
    elif isinstance(obj, list):
        for i, row in enumerate(obj):
            if isinstance(row, str):
                p = row.strip()
            elif isinstance(row, dict):
                # support both {"prompt": "..."} and {"caption": "..."}
                p = str(row.get("prompt", row.get("caption", ""))).strip()
            else:
                raise ValueError(f"Unsupported prompt type at index={i}: {type(row)}")
            if p:
                prompts.append(p)
    else:
        raise ValueError(f"Unsupported prompt json type: {type(obj)}")
    prompts = list(dict.fromkeys(prompts))
    if len(prompts) == 0:
        raise ValueError(f"No valid prompt found in {path}")
    return prompts


def _sample_and_split_prompts(prompts_pool: list[str], num_prompts: int, train_ratio: float, seed: int):
    if num_prompts <= 1:
        raise ValueError("num_prompts must be >= 2")
    if num_prompts > len(prompts_pool):
        raise ValueError(f"num_prompts={num_prompts} > available prompts={len(prompts_pool)}")
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(prompts_pool), generator=g).tolist()
    picked = [prompts_pool[i] for i in perm[: int(num_prompts)]]
    n_train = min(max(int(round(len(picked) * float(train_ratio))), 1), len(picked) - 1)
    train_prompts = picked[:n_train]
    val_prompts = picked[n_train:]
    return picked, train_prompts, val_prompts


def _sample_and_split_prompts_fixed_counts(
    prompts_pool: list[str], num_train_prompts: int, num_val_prompts: int, seed: int
):
    if num_train_prompts <= 0 or num_val_prompts <= 0:
        raise ValueError("num_train_prompts and num_val_prompts must be >= 1")
    total = int(num_train_prompts) + int(num_val_prompts)
    if total > len(prompts_pool):
        raise ValueError(f"requested total prompts={total} > available prompts={len(prompts_pool)}")
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(prompts_pool), generator=g).tolist()
    picked = [prompts_pool[i] for i in perm[:total]]
    train_prompts = picked[: int(num_train_prompts)]
    val_prompts = picked[int(num_train_prompts) :]
    return picked, train_prompts, val_prompts


def _imagereward_batch(images: list[Image.Image], prompt: str, model) -> torch.Tensor:
    vals = [float(model.score(prompt, img)) for img in images]
    return torch.tensor(vals, dtype=torch.float32)


@torch.inference_mode()
def _mc_rollout_imagereward(
    *,
    pipe: PixArtAlphaPipeline,
    latents_start: torch.Tensor,
    start_step_idx: int,
    timesteps: torch.Tensor,
    scheduler_snapshot,
    prompt_embeds_cfg: torch.Tensor,
    prompt_attention_mask_cfg: torch.Tensor,
    added_cond_kwargs: dict[str, torch.Tensor],
    guidance_scale: float,
    seed_base: int,
    mc: int,
    prompt: str,
    rm_model,
) -> torch.Tensor:
    if start_step_idx >= len(timesteps):
        imgs = _decode_pixart_latents_to_pil(pipe, latents_start)
        return _imagereward_batch(imgs, prompt=prompt, model=rm_model)

    rewards = []
    for i in range(int(mc)):
        scheduler = copy.deepcopy(scheduler_snapshot)
        latents = latents_start.detach().clone()
        generator = torch.Generator(device=latents.device).manual_seed(int(seed_base) + i)

        for step_i in range(int(start_step_idx), int(len(timesteps))):
            t = timesteps[step_i]
            latent_model_input = torch.cat([latents, latents], dim=0) if guidance_scale > 1.0 else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            t_b = _broadcast_timesteps(t, latent_model_input)

            noise_pred = pipe.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds_cfg,
                encoder_attention_mask=prompt_attention_mask_cfg,
                timestep=t_b,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if pipe.transformer.config.out_channels // 2 == pipe.transformer.config.in_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            latents = scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]

        imgs = _decode_pixart_latents_to_pil(pipe, latents)
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
        x = self.input_proj(x) # input_dim -> hidden_dim
        for layer in self.layers:
            x = x + layer(x) # hidden_dim -> hidden_dim
        return self.reward_head(x) # hidden_dim -> 1


class HiddenRewardDataset(Dataset):
    def __init__(self, hiddens: torch.Tensor, rewards: torch.Tensor):
        self.hiddens = hiddens.float()
        self.rewards = rewards.view(-1).float()
        # check 
        if self.hiddens.dim() != 2:
            raise ValueError(f"hiddens must be 2D, got {tuple(self.hiddens.shape)}")
        if self.hiddens.shape[0] != self.rewards.shape[0]:
            raise ValueError(f"row mismatch: {self.hiddens.shape[0]} vs {self.rewards.shape[0]}")
        # check end 

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
def collect_hidden_and_reward_for_prompts(
    *,
    model_path: str,
    prompts: list[str],
    split_name: str,
    negative_prompt: str,
    output_dir: Path,
    num_samples_per_prompt: int = 8,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    target_step_idx: int = 10,
    target_layer_idx: int = 14,
    mc: int = 4,
    collect_batch_size: int = 4,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    dtype_name: str = "fp16",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _resolve_dtype(dtype_name, device)
    do_cfg = guidance_scale > 1.0

    pipe = PixArtAlphaPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    rm_model = load_imagereward(device=device)

    n_blocks = len(pipe.transformer.transformer_blocks)
    # check 
    if not (0 <= target_layer_idx < n_blocks):
        raise ValueError(f"target_layer_idx must in [0, {n_blocks - 1}]")
    if not (0 <= target_step_idx < num_inference_steps):
        raise ValueError(f"target_step_idx must in [0, {num_inference_steps - 1}]")
    # check end 

    latent_channels = pipe.transformer.config.in_channels # C
    latent_size = pipe.transformer.config.sample_size # H,W = size, size

    all_hidden: list[torch.Tensor] = [] # list of hidden tensors
    all_reward: list[torch.Tensor] = [] # list of reward tensors
    all_seed: list[int] = [] # list of seed integers
    all_prompt_idx: list[int] = [] # list of prompt indices

    cur_step = {"value": -1}
    cur_batch = {"value": 0}
    hidden_buf: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out) -> None:
        if cur_step["value"] != target_step_idx:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        if do_cfg:
            h = h[: cur_batch["value"]]
        hidden_buf["h"] = h.mean(dim=1).detach().float().cpu()

    handle = pipe.transformer.transformer_blocks[target_layer_idx].register_forward_hook(_hook)
    try:
        outer = tqdm(range(len(prompts)), desc=f"collect {split_name}")
        for prompt_idx in outer:
            prompt = str(prompts[prompt_idx])
            for start in range(0, int(num_samples_per_prompt), int(collect_batch_size)):
                batch_n = min(int(collect_batch_size), int(num_samples_per_prompt) - start)
                cur_batch["value"] = batch_n
                batch_seed = int(seed) + prompt_idx * 100_000 + start # seed for each batch
                generator = torch.Generator(device=device).manual_seed(batch_seed)

                batch_prompts = [prompt] * batch_n
                prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
                    pipe.encode_prompt(
                        batch_prompts,
                        do_classifier_free_guidance=do_cfg,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=1,
                        device=device,
                        clean_caption=False,
                        max_sequence_length=120,
                    )
                )

                if do_cfg:
                    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    prompt_attention_mask_cfg = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
                else:
                    prompt_embeds_cfg = prompt_embeds
                    prompt_attention_mask_cfg = prompt_attention_mask

                added_cond_kwargs = _build_added_cond_kwargs(
                    height=height,
                    width=width,
                    batch=batch_n,
                    do_cfg=do_cfg,
                    device=device,
                    dtype=prompt_embeds.dtype,
                )

                latents = torch.randn(
                    (batch_n, latent_channels, latent_size, latent_size),
                    generator=generator,
                    device=device,
                    dtype=pipe.transformer.dtype,
                )
                pipe.scheduler.set_timesteps(int(num_inference_steps), device=device)
                timesteps = pipe.scheduler.timesteps

                snapshot_latents = None
                snapshot_scheduler = None
                hidden_buf.clear()

                for step_i, t in enumerate(timesteps):
                    cur_step["value"] = step_i
                    latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                    t_b = _broadcast_timesteps(t, latent_model_input)

                    noise_pred = pipe.transformer(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds_cfg,
                        encoder_attention_mask=prompt_attention_mask_cfg,
                        timestep=t_b,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    if do_cfg:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if pipe.transformer.config.out_channels // 2 == latent_channels:
                        noise_pred = noise_pred.chunk(2, dim=1)[0]

                    latents = pipe.scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]
                    if step_i == target_step_idx:
                        snapshot_latents = latents.detach().clone()
                        snapshot_scheduler = copy.deepcopy(pipe.scheduler)

                if "h" not in hidden_buf:
                    raise RuntimeError(f"Failed to capture hidden at target step {target_step_idx}")
                if snapshot_latents is None or snapshot_scheduler is None:
                    raise RuntimeError(f"Failed to snapshot latent/scheduler at target step {target_step_idx}")

                reward = _mc_rollout_imagereward(
                    pipe=pipe,
                    latents_start=snapshot_latents,
                    start_step_idx=target_step_idx + 1,
                    timesteps=timesteps,
                    scheduler_snapshot=snapshot_scheduler,
                    prompt_embeds_cfg=prompt_embeds_cfg,
                    prompt_attention_mask_cfg=prompt_attention_mask_cfg,
                    added_cond_kwargs=added_cond_kwargs,
                    guidance_scale=float(guidance_scale),
                    seed_base=batch_seed + target_step_idx * 13,
                    mc=int(mc),
                    prompt=prompt,
                    rm_model=rm_model,
                )

                all_hidden.append(hidden_buf["h"])
                all_reward.append(reward)
                all_seed.extend([batch_seed + i for i in range(batch_n)])
                all_prompt_idx.extend([prompt_idx] * batch_n)

                del prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
                del prompt_embeds_cfg, prompt_attention_mask_cfg, latents, snapshot_latents, snapshot_scheduler
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        handle.remove()

    hiddens = torch.cat(all_hidden, dim=0).float()
    rewards = torch.cat(all_reward, dim=0).float()
    seeds = torch.tensor(all_seed, dtype=torch.long)
    prompt_indices = torch.tensor(all_prompt_idx, dtype=torch.long)
    expected = int(len(prompts) * int(num_samples_per_prompt))
    if hiddens.shape[0] != expected or rewards.shape[0] != expected:
        raise RuntimeError(f"Unexpected rows, hidden={hiddens.shape[0]}, reward={rewards.shape[0]}, expected={expected}")

    out_pt = output_dir / f"pixart_ir_hs_reward_{split_name}_p{len(prompts)}_npp{num_samples_per_prompt}.pt"
    torch.save(
        {
            "hiddens": hiddens,
            "reward": rewards,
            "seed": seeds,
            "prompt_indices": prompt_indices,
            "prompts": prompts,
            "meta": {
                "split_name": split_name,
                "model_path": str(model_path),
                "negative_prompt": negative_prompt,
                "num_prompts": int(len(prompts)),
                "num_samples_per_prompt": int(num_samples_per_prompt),
                "num_inference_steps": int(num_inference_steps),
                "target_step_idx": int(target_step_idx),
                "target_layer_idx": int(target_layer_idx),
                "mc": int(mc),
                "guidance_scale": float(guidance_scale),
                "height": int(height),
                "width": int(width),
            },
        },
        out_pt,
    )
    return out_pt


def train_prob_on_split_data(
    *,
    train_pt: Path,
    val_pt: Path,
    output_dir: Path,
    train_batch_size: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout_prob: float = 0.1,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_interval: int = 1,
) -> dict:
    tr_obj = torch.load(train_pt, map_location="cpu", weights_only=False)
    va_obj = torch.load(val_pt, map_location="cpu", weights_only=False)
    train_h = tr_obj["hiddens"].float()
    val_h = va_obj["hiddens"].float()
    train_r_raw = tr_obj["reward"].view(-1).float()
    val_r_raw = va_obj["reward"].view(-1).float()
    if train_h.shape[0] < 2 or val_h.shape[0] < 1:
        raise ValueError("Train/val rows are too small.")

    r_mean = float(train_r_raw.mean().item())
    r_std = float(train_r_raw.std(unbiased=False).item())
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
        "train_pt": str(train_pt),
        "val_pt": str(val_pt),
        "best_ckpt": str(best_ckpt),
        "best_epoch": int(best_epoch),
        "best_val_loss_norm": float(best_val),
        "final_val_loss_norm": float(val_loss_norm),
        "val_pearson_raw": float(val_pearson),
        "num_train": int(train_h.shape[0]),
        "num_val": int(val_h.shape[0]),
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
    p = argparse.ArgumentParser("PixArt hidden -> ImageReward prob trainer (multi-prompt)")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/PixArt-XL-2-1024-MS")
    p.add_argument(
        "--train_prompts_json",
        type=str,
        default="/root/autodl-tmp/coco2017/prompts/coco2017_train_prompts_15000.json",
    )
    p.add_argument(
        "--val_prompts_json",
        type=str,
        default="/root/autodl-tmp/coco2017/prompts/coco2017_test_prompts_5000.json",
    )
    p.add_argument("--negative_prompt", type=str, default="blurry, low quality, watermark, text, deformed")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/text2image_pixart_ir_prob_multiprompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--num_samples_per_prompt", type=int, default=5)
    p.add_argument("--collect_batch_size", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--target_step_idx", type=int, default=10)
    p.add_argument("--target_layer_idx", type=int, default=14)
    p.add_argument("--mc", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
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
    output_dir.mkdir(parents=True, exist_ok=True)

    train_prompts = _load_prompts(Path(args.train_prompts_json).expanduser().resolve())
    val_prompts = _load_prompts(Path(args.val_prompts_json).expanduser().resolve())
    selected = train_prompts + val_prompts
    (output_dir / "selected_prompts.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "train_prompts.json").write_text(json.dumps(train_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "val_prompts.json").write_text(json.dumps(val_prompts, ensure_ascii=False, indent=2), encoding="utf-8")

    train_pt = collect_hidden_and_reward_for_prompts(
        model_path=str(args.model_path),
        prompts=train_prompts,
        split_name="train",
        negative_prompt=str(args.negative_prompt),
        output_dir=data_dir,
        num_samples_per_prompt=int(args.num_samples_per_prompt),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        target_step_idx=int(args.target_step_idx),
        target_layer_idx=int(args.target_layer_idx),
        mc=int(args.mc),
        collect_batch_size=int(args.collect_batch_size),
        height=int(args.height),
        width=int(args.width),
        seed=int(args.seed),
        dtype_name=str(args.dtype),
    )
    
    val_pt = collect_hidden_and_reward_for_prompts(
        model_path=str(args.model_path),
        prompts=val_prompts,
        split_name="val",
        negative_prompt=str(args.negative_prompt),
        output_dir=data_dir,
        num_samples_per_prompt=int(args.num_samples_per_prompt),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        target_step_idx=int(args.target_step_idx),
        target_layer_idx=int(args.target_layer_idx),
        mc=int(args.mc),
        collect_batch_size=int(args.collect_batch_size),
        height=int(args.height),
        width=int(args.width),
        seed=int(args.seed) + 777,
        dtype_name=str(args.dtype),
    )

    result = train_prob_on_split_data(
        train_pt=train_pt,
        val_pt=val_pt,
        output_dir=run_dir,
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
