#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from class_condition.ranker import DiffusionProb_SingleClass
from utils import dit_loader, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Single-class random search with trained single-class ranker")
    p.add_argument(
        "--ranker_ckpt",
        type=str,
        default="/root/autodl-tmp/ranker_ckpts/single_class/single_class_ranker_best.pth",
    )
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/class_condition/rs_single_class_ranker")
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))

    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--parallel_size", type=int, default=8, help="random search candidate count, i.e. ps")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--class_idx", type=int, default=None, help="if unset, read from ranker ckpt config")
    p.add_argument("--target_step_idx", type=int, default=None, help="if unset, read from ranker ckpt config")
    p.add_argument("--target_layer_idx", type=int, default=None, help="if unset, read from ranker ckpt config")
    p.add_argument("--hidden_pool", type=str, default="mean", choices=("mean", "last"))
    return p.parse_args()


def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        if isinstance(t, float):
            dt = torch.float64
        else:
            dt = torch.int64
        timesteps = torch.tensor([t], dtype=dt, device=latent_model_input.device)
    elif len(t.shape) == 0:
        timesteps = t[None].to(latent_model_input.device)
    else:
        timesteps = t
    return timesteps.expand(latent_model_input.shape[0])


def _decode_to_pil(pipe, x: torch.Tensor, guidance_scale: float) -> list[Image.Image]:
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


def _load_single_class_ranker(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[DiffusionProb_SingleClass, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    cfg = ckpt.get("config", {}) or {}

    hs_dim = int(ckpt.get("hidden_dim", state["linear_candidate.weight"].shape[1]))
    hidden_dim = int(cfg.get("hidden_dim", state["linear_candidate.weight"].shape[0]))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.1))
    half = bool(cfg.get("half", True))

    ranker = DiffusionProb_SingleClass(
        hs_dim=hs_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        half=half,
    )
    ranker.load_state_dict(state, strict=True)
    ranker.to(device)
    ranker.eval()
    return ranker, cfg


@torch.inference_mode()
def _sample_candidate_batch(
    *,
    pipe,
    device: torch.device,
    class_idx: int,
    batch_n: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed_this_candidate: int,
    target_step_idx: int,
    target_layer_idx: int,
    hidden_pool: str,
) -> tuple[list[Image.Image], torch.Tensor]:
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    hidden_holder: dict[str, torch.Tensor | None] = {"value": None}
    current_step = {"value": -1}
    current_batch = {"value": batch_n}

    def _hook(_module, _inp, out) -> None:
        if current_step["value"] != target_step_idx:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        if guidance_scale > 1:
            h = h[: current_batch["value"]]
        if hidden_pool == "mean":
            h = h.mean(dim=1)
        else:
            h = h[:, -1, :]
        hidden_holder["value"] = h.detach().to(torch.float32).cpu()

    handle = pipe.transformer.transformer_blocks[target_layer_idx].register_forward_hook(_hook)
    try:
        gen = torch.Generator(device=device).manual_seed(int(seed_this_candidate))
        latents = torch.randn(
            (batch_n, latent_channels, latent_size, latent_size),
            generator=gen,
            device=device,
            dtype=pipe.transformer.dtype,
        )
        x = torch.cat([latents, latents], dim=0) if guidance_scale > 1 else latents

        cls = torch.full((batch_n,), int(class_idx), device=device, dtype=torch.long)
        cls_null = torch.full((batch_n,), 1000, device=device, dtype=torch.long)
        class_labels_input = torch.cat([cls, cls_null], dim=0) if guidance_scale > 1 else cls

        pipe.scheduler.set_timesteps(int(num_inference_steps))
        timesteps = pipe.scheduler.timesteps
        for st_i, t in enumerate(timesteps):
            current_step["value"] = st_i
            if guidance_scale > 1:
                half = x[: len(x) // 2]
                x = torch.cat([half, half], dim=0)
            x_in = pipe.scheduler.scale_model_input(x, t)
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
            x = pipe.scheduler.step(model_output, t, x, generator=gen).prev_sample

        if hidden_holder["value"] is None:
            raise RuntimeError(
                f"target_step_idx={target_step_idx} hidden not captured; check num_inference_steps."
            )
        images = _decode_to_pil(pipe, x, guidance_scale)
        return images, hidden_holder["value"]
    finally:
        handle.remove()


@torch.inference_mode()
def run_random_search() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    output_dir = Path(args.output_dir).expanduser().resolve()
    best_dir = output_dir / "best_images"
    best_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)

    ranker_ckpt = Path(args.ranker_ckpt).expanduser().resolve()
    ranker, cfg = _load_single_class_ranker(ranker_ckpt, device)

    class_idx = int(args.class_idx if args.class_idx is not None else cfg.get("class_idx", 153))
    target_step_idx = int(args.target_step_idx if args.target_step_idx is not None else cfg.get("step_idx", 9))
    target_layer_idx = int(args.target_layer_idx if args.target_layer_idx is not None else cfg.get("layer_idx", 16))

    if target_step_idx < 0 or target_step_idx >= int(args.num_inference_steps):
        raise ValueError(
            f"target_step_idx={target_step_idx} should be in [0, {int(args.num_inference_steps)-1}]"
        )

    n_blocks = len(pipe.transformer.transformer_blocks)
    if target_layer_idx < 0 or target_layer_idx >= n_blocks:
        raise ValueError(f"target_layer_idx={target_layer_idx} out of range [0, {n_blocks-1}]")

    print(
        f"[info] class={class_idx} num_samples={args.num_samples} ps={args.parallel_size} "
        f"step={target_step_idx} layer={target_layer_idx} hidden_pool={args.hidden_pool}"
    )

    img_counter = 0
    num_rounds = (int(args.num_samples) + int(args.batch_size) - 1) // int(args.batch_size)
    pbar = tqdm(range(num_rounds), total=num_rounds, desc="random search single-class ranker")
    for round_i in pbar:
        n_this = min(int(args.batch_size), int(args.num_samples) - img_counter)
        if n_this <= 0:
            break

        cand_images: list[list[Image.Image]] = []
        cand_scores: list[torch.Tensor] = []
        for ci in range(int(args.parallel_size)):
            seed_this_candidate = int(args.seed) + round_i * 10007 + ci * 97
            images_i, hs_i = _sample_candidate_batch(
                pipe=pipe,
                device=device,
                class_idx=class_idx,
                batch_n=n_this,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.guidance_scale),
                seed_this_candidate=seed_this_candidate,
                target_step_idx=target_step_idx,
                target_layer_idx=target_layer_idx,
                hidden_pool=args.hidden_pool,
            )

            w_dtype = next(ranker.parameters()).dtype
            pred_i = ranker(hs_i.to(device=device, dtype=w_dtype)).reshape(-1).float().cpu()
            cand_images.append(images_i)
            cand_scores.append(pred_i)

        score_mat = torch.stack(cand_scores, dim=1)  # (B, ps)
        best_idx = score_mat.argmax(dim=1)  # (B,)
        for bi in range(n_this):
            winner = int(best_idx[bi].item())
            save_name = f"image_{img_counter:05d}_{class_idx:05d}.png"
            cand_images[winner][bi].save(best_dir / save_name)
            img_counter += 1

        mean_winner_score = float(score_mat.gather(1, best_idx.unsqueeze(1)).mean().item())
        pbar.set_postfix({"saved": img_counter, "winner_score": f"{mean_winner_score:.4f}"})

    cfg_dump = {
        "ranker_ckpt": str(ranker_ckpt),
        "model_name": args.model_name,
        "dtype": args.dtype,
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "parallel_size": int(args.parallel_size),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
        "class_idx": int(class_idx),
        "target_step_idx": int(target_step_idx),
        "target_layer_idx": int(target_layer_idx),
        "hidden_pool": str(args.hidden_pool),
        "saved_images": int(img_counter),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(cfg_dump, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(cfg_dump, ensure_ascii=False))


if __name__ == "__main__":
    run_random_search()
