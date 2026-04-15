#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path("/root/diffusion_ranker")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import model_loader, set_seed


def _parse_ratio_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty ratio list: {raw!r}")
    return vals


def _ratio_tag(r: float) -> str:
    return f"{float(r):g}"


def _pick_resnet_modules(unet, layer_ratios: list[float]) -> dict[str, tuple[str, torch.nn.Module]]:
    blocks: list[tuple[str, torch.nn.Module]] = []
    for name, mod in unet.named_modules():
        if mod.__class__.__name__ == "ResnetBlock2D":
            blocks.append((name, mod))
    if not blocks:
        raise RuntimeError("no ResnetBlock2D modules found in DDPM UNet")

    out: dict[str, tuple[str, torch.nn.Module]] = {}
    n = len(blocks)
    for r in layer_ratios:
        if not (0.0 <= float(r) <= 1.0):
            raise ValueError(f"layer ratio must be in [0,1], got {r}")
        idx = min(n - 1, max(0, round(float(r) * (n - 1))))
        out[_ratio_tag(r)] = blocks[idx]
    return out


def _latents_to_pil(latents: torch.Tensor) -> list[Image.Image]:
    x = (latents / 2 + 0.5).clamp(0, 1)
    arr = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype("uint8")
    return [Image.fromarray(im) for im in arr]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Rollout unconditional DDPM and capture layer/step ratio representations")
    p.add_argument("--model_name", type=str, default="google_ddpm-celebahq-256")
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--layer_ratios", type=str, default="0.6")
    p.add_argument("--step_ratios", type=str, default="1.0")
    p.add_argument(
        "--output_image_root",
        type=Path,
        default=Path("/root/autodl-tmp/ddpm_candidates_50k_deno10"),
    )
    p.add_argument(
        "--output_repr_root",
        type=Path,
        default=Path("/root/autodl-tmp/ddpm_repr_candidates_deno10"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--store_dtype", type=str, default="float16", choices=("float16", "float32", "bfloat16"))
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layer_ratios = _parse_ratio_list(args.layer_ratios)
    step_ratios = _parse_ratio_list(args.step_ratios)
    for sr in step_ratios:
        if not (0.0 <= float(sr) <= 1.0):
            raise ValueError(f"step ratio must be in [0,1], got {sr}")

    pipe = model_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    unet, scheduler = pipe.unet, pipe.scheduler

    ratio_to_module = _pick_resnet_modules(unet, layer_ratios)
    step_idx_to_ratios: dict[int, list[float]] = {}
    for sr in step_ratios:
        sidx = min(int(args.num_inference_steps) - 1, max(0, round(float(sr) * (int(args.num_inference_steps) - 1))))
        step_idx_to_ratios.setdefault(sidx, []).append(float(sr))

    all_repr_keys = sorted({f"s{sidx}_r{_ratio_tag(lr)}" for sidx in step_idx_to_ratios for lr in layer_ratios})
    store_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.store_dtype]

    args.output_image_root = args.output_image_root.expanduser().resolve()
    args.output_repr_root = args.output_repr_root.expanduser().resolve()
    args.output_image_root.mkdir(parents=True, exist_ok=True)
    args.output_repr_root.mkdir(parents=True, exist_ok=True)

    current_step = -1
    capture_bucket: dict[str, torch.Tensor] = {}

    def _make_hook(r_tag: str):
        def _hook(_module, _inp, out):
            if current_step < 0:
                return
            if current_step not in step_idx_to_ratios:
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t) or t.dim() != 4:
                return
            pooled = t.detach().float().mean(dim=(2, 3)).to("cpu", dtype=store_dtype)
            capture_bucket[f"s{current_step}_r{r_tag}"] = pooled

        return _hook

    handles = []
    module_manifest: dict[str, str] = {}
    for r_tag, (mname, mod) in ratio_to_module.items():
        module_manifest[r_tag] = mname
        handles.append(mod.register_forward_hook(_make_hook(r_tag)))

    manifest = {
        "model_name": str(args.model_name),
        "num_candidates": int(args.num_candidates),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "num_inference_steps": int(args.num_inference_steps),
        "layer_ratios": [float(x) for x in layer_ratios],
        "step_ratios": [float(x) for x in step_ratios],
        "repr_keys": all_repr_keys,
        "module_manifest": module_manifest,
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "store_dtype": str(args.store_dtype),
    }
    (args.output_repr_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for cand_idx in range(int(args.num_candidates)):
        cimg = args.output_image_root / f"candidate_{cand_idx:04d}"
        crepr = args.output_repr_root / f"candidate_{cand_idx:04d}"
        cimg.mkdir(parents=True, exist_ok=True)
        crepr.mkdir(parents=True, exist_ok=True)

        repr_chunks: dict[str, list[torch.Tensor]] = {k: [] for k in all_repr_keys}
        saved = 0
        n_batches = (int(args.num_samples) + int(args.batch_size) - 1) // int(args.batch_size)
        for bidx in tqdm(range(n_batches), desc=f"rollout cand {cand_idx:04d}", leave=False):
            n_this = min(int(args.batch_size), int(args.num_samples) - bidx * int(args.batch_size))
            if n_this <= 0:
                break

            scheduler.set_timesteps(int(args.num_inference_steps), device=device)
            gen = torch.Generator(device=device).manual_seed(int(args.seed) + cand_idx * 1_000_000 + bidx)
            latents = torch.randn(
                (n_this, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size),
                generator=gen,
                device=device,
                dtype=unet.dtype,
            )

            capture_bucket.clear()
            for st_i, t in enumerate(scheduler.timesteps):
                current_step = int(st_i)
                noise_pred = unet(latents, t).sample
                latents = scheduler.step(noise_pred, t, latents, generator=gen).prev_sample
                if st_i in step_idx_to_ratios:
                    for lr in layer_ratios:
                        k = f"s{st_i}_r{_ratio_tag(lr)}"
                        feat = capture_bucket.get(k)
                        if feat is None:
                            raise RuntimeError(f"missing feature capture for key={k} at candidate={cand_idx}, batch={bidx}")
                        repr_chunks[k].append(feat)
                    capture_bucket.clear()
            current_step = -1

            for im in _latents_to_pil(latents):
                if saved >= int(args.num_samples):
                    break
                if int(args.image_size) > 0 and im.size != (int(args.image_size), int(args.image_size)):
                    im = im.resize((int(args.image_size), int(args.image_size)), Image.LANCZOS)
                im.save(cimg / f"{saved:05d}.png")
                saved += 1

        for k, chunks in repr_chunks.items():
            if not chunks:
                raise RuntimeError(f"empty repr chunks for key={k}, candidate={cand_idx}")
            tensor = torch.cat(chunks, dim=0)
            if tensor.shape[0] > int(args.num_samples):
                tensor = tensor[: int(args.num_samples)]
            torch.save({"tensor": tensor, "meta": manifest}, crepr / f"{k}.pt")

    for h in handles:
        h.remove()

    print(
        json.dumps(
            {
                "output_image_root": str(args.output_image_root),
                "output_repr_root": str(args.output_repr_root),
                "num_candidates": int(args.num_candidates),
                "num_samples": int(args.num_samples),
                "repr_keys": all_repr_keys,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
