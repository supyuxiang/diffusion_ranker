#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from class_condition.reward import labels_1k, load_clip, load_imagereward
from ranker.model import DiffusionRanker_Transformer
from utils import compute_fid_is_from_dirs, dit_loader, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ImageNet 50k random search with listwise ranker checkpoint")
    p.add_argument(
        "--ranker_ckpt",
        type=str,
        default="/root/autodl-tmp/ranker_ckpts/ranker_imagenet_dit_is_listwise_best.pth",
    )
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/class_condition/rs_ranker_imagenet_50k")
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--real_dir", type=str, default="/root/autodl-tmp/imagenet/val")

    # RS settings
    p.add_argument("--num_samples", type=int, default=50_000, help="Should be 50_000 for 50/class protocol")
    p.add_argument("--batch_size", type=int, default=2, help="RS batch size (bz)")
    p.add_argument("--parallel_size", type=int, default=8, help="RS candidate count (ps)")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Representation capture for ranker input
    p.add_argument("--target_step_idx", type=int, default=6)
    p.add_argument("--target_layer_idx", type=int, default=16)
    p.add_argument("--hidden_pool", type=str, default="mean", choices=("mean", "last"))

    # Prompt representation for ranker prompt branch (1024-D for this ckpt).
    p.add_argument(
        "--prompt_repr_pt",
        type=str,
        default="",
        help="Optional .pt tensor of shape (1000, D_prompt). If empty, compute BERT-large CLS embeddings online.",
    )
    p.add_argument(
        "--bert_model_dir",
        type=str,
        default="/root/autodl-tmp/bert-large-cased",
        help="Used only when --prompt_repr_pt is not provided.",
    )
    return p.parse_args()


def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        dt = torch.float64 if isinstance(t, float) else torch.int64
        timesteps = torch.tensor([t], dtype=dt, device=latent_model_input.device)
    elif len(t.shape) == 0:
        timesteps = t[None].to(latent_model_input.device)
    else:
        timesteps = t
    return timesteps.expand(latent_model_input.shape[0])


def _build_labels_50k_class_major(num_samples: int) -> list[int]:
    labels = [c for c in range(1000) for _ in range(50)]
    if num_samples != 50_000:
        if num_samples > len(labels):
            raise ValueError(f"num_samples={num_samples} > 50,000 class-major protocol size")
        labels = labels[:num_samples]
    return labels


def _load_or_build_prompt_table(args: argparse.Namespace, device: torch.device, d_prompt: int) -> torch.Tensor:
    if str(args.prompt_repr_pt).strip():
        pt = Path(args.prompt_repr_pt).expanduser().resolve()
        obj = torch.load(pt, map_location="cpu", weights_only=False)
        tab = obj["tensor"] if isinstance(obj, dict) and "tensor" in obj else obj
        if not isinstance(tab, torch.Tensor) or tab.dim() != 2:
            raise ValueError(f"Invalid prompt table at {pt}: expected 2D Tensor")
        if tab.shape[0] != 1000:
            raise ValueError(f"Prompt table rows should be 1000, got {tab.shape[0]}")
        if tab.shape[1] != d_prompt:
            raise ValueError(f"Prompt dim mismatch: table={tab.shape[1]} vs ckpt={d_prompt}")
        return tab.float().cpu()

    # Fallback: build from local BERT-large CLS embeddings (1000, 1024).
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(args.bert_model_dir).expanduser().resolve()
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    texts = [str(x) for x in labels_1k]
    with torch.inference_mode():
        enc = tok(texts, padding=True, truncation=True, max_length=16, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc).last_hidden_state[:, 0, :].float().cpu()
    if out.shape != (1000, d_prompt):
        raise ValueError(f"Built prompt table shape {tuple(out.shape)} does not match expected (1000, {d_prompt})")
    return out


def _load_ranker_ckpt(ckpt_path: Path, device: torch.device) -> tuple[DiffusionRanker_Transformer, int, int]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # This ckpt is a raw state_dict; also support dict with model_state_dict.
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported ckpt format at {ckpt_path}")

    if "linear_candidate.weight" not in state or "linear_prompt.weight" not in state:
        raise ValueError("Checkpoint does not look like DiffusionRanker_Transformer state_dict")

    hidden_dim = int(state["linear_candidate.weight"].shape[0])
    input_dim_candidate = int(state["linear_candidate.weight"].shape[1])
    input_dim_prompt = int(state["linear_prompt.weight"].shape[1])
    num_layers = max(int(k.split(".")[1]) for k in state if k.startswith("layers.")) + 1
    # Infer head count by testing common choices.
    num_heads = None
    for h in (2, 4, 8, 16):
        if hidden_dim % h == 0:
            num_heads = h
            break
    if num_heads is None:
        raise ValueError(f"Cannot infer num_heads for hidden_dim={hidden_dim}")

    ranker = DiffusionRanker_Transformer(
        input_dim_candidate=input_dim_candidate,
        input_dim_prompt=input_dim_prompt,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.1,
        num_layers=num_layers,
        half=True,
    )
    ranker.load_state_dict(state, strict=True)
    ranker.to(device).eval()
    return ranker, input_dim_candidate, input_dim_prompt


def _compile_module_if_needed(module: torch.nn.Module, module_name: str) -> torch.nn.Module:
    if not hasattr(torch, "compile"):
        print(f"[compile] {module_name}: torch.compile unavailable in current PyTorch, fallback to eager")
        return module

    # Auto mode: try compile, fallback to eager if compile fails.
    try:
        compiled = torch.compile(module, mode="max-autotune", fullgraph=False, dynamic=False)
        print(f"[compile] {module_name}: enabled")
        return compiled
    except Exception as e:
        print(f"[compile] {module_name}: failed ({type(e).__name__}: {e}), fallback to eager")
        return module


@torch.inference_mode()
def _sample_candidate_batch(
    *,
    pipe,
    device: torch.device,
    labels_this_batch: list[int],
    num_inference_steps: int,
    guidance_scale: float,
    seed_this_candidate: int,
    target_step_idx: int,
    target_layer_idx: int,
    hidden_pool: str,
) -> tuple[list[Image.Image], torch.Tensor]:
    bsz = len(labels_this_batch)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels
    hidden_holder: dict[str, torch.Tensor | None] = {"value": None}
    current_step = {"value": -1}

    def _hook(_module, _inp, out) -> None:
        if current_step["value"] != target_step_idx:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        if guidance_scale > 1:
            h = h[:bsz]
        h = h.mean(dim=1) if hidden_pool == "mean" else h[:, -1, :]
        hidden_holder["value"] = h.detach().to(torch.float32).cpu()

    handle = pipe.transformer.transformer_blocks[target_layer_idx].register_forward_hook(_hook)
    try:
        gen = torch.Generator(device=device).manual_seed(int(seed_this_candidate))
        latents = torch.randn(
            (bsz, latent_channels, latent_size, latent_size),
            generator=gen,
            device=device,
            dtype=pipe.transformer.dtype,
        )
        x = torch.cat([latents, latents], dim=0) if guidance_scale > 1 else latents
        cls = torch.tensor(labels_this_batch, device=device, dtype=torch.long)
        cls_null = torch.full((bsz,), 1000, device=device, dtype=torch.long)
        class_labels_input = torch.cat([cls, cls_null], dim=0) if guidance_scale > 1 else cls

        pipe.scheduler.set_timesteps(int(num_inference_steps))
        for st_i, t in enumerate(pipe.scheduler.timesteps):
            current_step["value"] = st_i
            if guidance_scale > 1:
                half = x[: len(x) // 2]
                x = torch.cat([half, half], dim=0)
            x_in = pipe.scheduler.scale_model_input(x, t)
            t_b = _broadcast_timesteps(t, x_in)
            noise_pred = pipe.transformer(x_in, timestep=t_b, class_labels=class_labels_input).sample
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
            raise RuntimeError("Hidden capture failed")

        lat = x.chunk(2, dim=0)[0] if guidance_scale > 1 else x
        lat = 1 / pipe.vae.config.scaling_factor * lat
        samples = pipe.vae.decode(lat).sample
        samples = (samples / 2 + 0.5).clamp(0, 1)
        images: list[Image.Image] = []
        for bi in range(samples.shape[0]):
            arr = samples[bi].detach().cpu().permute(1, 2, 0).float().numpy()
            images.append(Image.fromarray((arr * 255.0).round().clip(0, 255).astype("uint8")))
        return images, hidden_holder["value"]
    finally:
        handle.remove()


@torch.inference_mode()
def _compute_clipscore_from_dir(image_dir: Path, device: torch.device, batch_size: int = 64) -> dict[str, float]:
    paths = sorted(image_dir.glob("*.png"))
    clip_model, clip_processor = load_clip(device)
    vals: list[float] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="clipscore"):
        chunk = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB").copy() for p in chunk]
        labels = [int(p.stem.split("_")[-1]) for p in chunk]
        prompts = [f"A photo of a {labels_1k[lab]}." for lab in labels]
        inputs = clip_processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = clip_model(**inputs)
        img_f = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_f = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (img_f * txt_f).sum(dim=-1)
        vals.extend(sim.detach().float().cpu().tolist())
    arr = np.asarray(vals, dtype=np.float64)
    return {"clipscore_mean": float(arr.mean()), "clipscore_std": float(arr.std(ddof=0))}


@torch.inference_mode()
def _compute_imagereward_from_dir(image_dir: Path, device: torch.device) -> dict[str, float]:
    paths = sorted(image_dir.glob("*.png"))
    rm_model = load_imagereward(device)
    vals: list[float] = []
    for p in tqdm(paths, desc="imagereward"):
        lab = int(p.stem.split("_")[-1])
        prompt = f"A photo of a {labels_1k[lab]}."
        img = Image.open(p).convert("RGB").copy()
        vals.append(float(rm_model.score(prompt, img)))
    arr = np.asarray(vals, dtype=np.float64)
    return {"imagereward_mean": float(arr.mean()), "imagereward_std": float(arr.std(ddof=0))}


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    real_dir = Path(args.real_dir).expanduser().resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranker_ckpt = Path(args.ranker_ckpt).expanduser().resolve()
    ranker, d_cand, d_prompt = _load_ranker_ckpt(ranker_ckpt, device)
    prompt_table = _load_or_build_prompt_table(args, device, d_prompt)

    pipe = dit_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    # Compile the heaviest module first; keep ranker eager by default since it's lightweight.
    pipe.transformer = _compile_module_if_needed(pipe.transformer, "dit_transformer")

    labels_all = _build_labels_50k_class_major(int(args.num_samples))
    num_rounds = (len(labels_all) + int(args.batch_size) - 1) // int(args.batch_size)
    img_counter = 0

    pbar = tqdm(range(num_rounds), desc="random search ranker 50k")
    for r in pbar:
        st = r * int(args.batch_size)
        ed = min(st + int(args.batch_size), len(labels_all))
        labels_this = labels_all[st:ed]
        bsz = len(labels_this)
        if bsz <= 0:
            break

        cand_images: list[list[Image.Image]] = []
        cand_hidden: list[torch.Tensor] = []
        for c in range(int(args.parallel_size)):
            seed_this = int(args.seed) + r * 10007 + c * 97
            imgs_c, hid_c = _sample_candidate_batch(
                pipe=pipe,
                device=device,
                labels_this_batch=labels_this,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.guidance_scale),
                seed_this_candidate=seed_this,
                target_step_idx=int(args.target_step_idx),
                target_layer_idx=int(args.target_layer_idx),
                hidden_pool=str(args.hidden_pool),
            )
            if hid_c.shape[-1] != d_cand:
                raise RuntimeError(f"Hidden dim mismatch: got {hid_c.shape[-1]} expected {d_cand}")
            cand_images.append(imgs_c)
            cand_hidden.append(hid_c)  # (B,D)

        # (B, K, D)
        candidate_hstates = torch.stack(cand_hidden, dim=1)
        prompt_hstates = prompt_table[torch.tensor(labels_this, dtype=torch.long)].to(torch.float32)
        w = next(ranker.parameters())
        logits = ranker(
            candidate_hstates.to(device=device, dtype=w.dtype),
            prompt_hstates.to(device=device, dtype=w.dtype),
        ).float().cpu()  # (B,K)
        best_idx = logits.argmax(dim=1)  # (B,)

        for bi in range(bsz):
            ci = int(best_idx[bi].item())
            lab = int(labels_this[bi])
            img = cand_images[ci][bi]
            img.save(output_dir / f"image_{img_counter:05d}_{lab:05d}.png")
            img_counter += 1

        pbar.set_postfix({"saved": img_counter})

    if img_counter != len(labels_all):
        raise RuntimeError(f"saved {img_counter} images, expected {len(labels_all)}")

    # Evaluate metrics
    fid_is = compute_fid_is_from_dirs(output_dir, real_dir, device=device, batch_size=64)
    clip_m = _compute_clipscore_from_dir(output_dir, device=device, batch_size=64)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ir_m = _compute_imagereward_from_dir(output_dir, device=device)
    out = {
        "num_images": int(img_counter),
        "FID": float(fid_is["FID"]),
        "IS": float(fid_is["IS"]),
        "IS_STD": float(fid_is["IS_STD"]),
        **clip_m,
        **ir_m,
        "ranker_ckpt": str(ranker_ckpt),
        "output_dir": str(output_dir),
        "real_dir": str(real_dir),
    }
    (output_dir / "final_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
