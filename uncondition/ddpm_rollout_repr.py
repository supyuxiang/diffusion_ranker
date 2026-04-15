#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import sys

REPO_ROOT = Path("/root/diffusion_ranker")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import ddpm_loader, set_seed


def _parse_ratio_list(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("ratio list is empty")
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"ratio must be in [0, 1], got {v}")
    return vals


def _dtype_from_str(dtype: str) -> torch.dtype:
    d = dtype.lower().strip()
    if d in ("fp16", "float16"):
        return torch.float16
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype {dtype}")


def _store_dtype_from_str(dtype: str) -> torch.dtype:
    d = dtype.lower().strip()
    if d in ("float16", "fp16"):
        return torch.float16
    if d in ("bfloat16", "bf16"):
        return torch.bfloat16
    if d in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"unsupported store_dtype {dtype}")


def _collect_resnet_modules(unet) -> list[tuple[str, torch.nn.Module]]:
    modules: list[tuple[str, torch.nn.Module]] = []
    for name, module in unet.named_modules():
        if module.__class__.__name__ == "ResnetBlock2D":
            modules.append((name, module))
    if not modules:
        raise RuntimeError("no ResnetBlock2D modules found in DDPM UNet")
    return modules


def _ratio_to_index(r: float, n: int) -> int:
    if n <= 1:
        return 0
    idx = int(round(r * (n - 1)))
    return max(0, min(n - 1, idx))


def _step_ratio_to_index(r: float, num_steps: int) -> int:
    if num_steps <= 1:
        return 0
    idx = int(round(r * (num_steps - 1)))
    return max(0, min(num_steps - 1, idx))


def _to_pil_images(pipe, image_tensor: torch.Tensor) -> list[Image.Image]:
    return pipe.image_processor.postprocess(image_tensor, output_type="pil")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DDPM rollout with layer-ratio and step-ratio representations")
    p.add_argument("--model_name", type=str, default="google_ddpm-celebahq-256")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--layer_ratios", type=str, default="0.1,0.3,0.6,0.9,1.0")
    p.add_argument("--step_ratios", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--output_image_dir", type=Path, required=True)
    p.add_argument("--output_repr_pt", type=Path, required=True)
    p.add_argument(
        "--output_repr_dir",
        type=Path,
        default=None,
        help="optional: save each repr key to separate *.pt under this directory",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16")
    p.add_argument("--store_dtype", type=str, default="float16")
    p.add_argument("--image_size", type=int, default=256)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_image_dir = args.output_image_dir.expanduser().resolve()
    output_repr_pt = args.output_repr_pt.expanduser().resolve()
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_repr_pt.parent.mkdir(parents=True, exist_ok=True)
    if args.output_repr_dir is not None:
        args.output_repr_dir = args.output_repr_dir.expanduser().resolve()
        args.output_repr_dir.mkdir(parents=True, exist_ok=True)

    layer_ratios = _parse_ratio_list(args.layer_ratios)
    step_ratios = _parse_ratio_list(args.step_ratios)
    store_dtype = _store_dtype_from_str(args.store_dtype)

    pipe = ddpm_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    unet = pipe.unet

    resnet_modules = _collect_resnet_modules(unet)
    selected_layers: dict[float, tuple[int, str, torch.nn.Module]] = {}
    for lr in layer_ratios:
        li = _ratio_to_index(lr, len(resnet_modules))
        lname, lmod = resnet_modules[li]
        selected_layers[lr] = (li, lname, lmod)

    selected_step_indices = sorted(
        {_step_ratio_to_index(sr, int(args.num_inference_steps)) for sr in step_ratios}
    )
    step_idx_to_ratio: dict[int, float] = {
        _step_ratio_to_index(sr, int(args.num_inference_steps)): sr for sr in step_ratios
    }

    layer_capture: dict[float, torch.Tensor | None] = {lr: None for lr in layer_ratios}
    capture_enabled = False

    def _make_hook(lr_key: float):
        def _hook(_module, _inputs, output):
            nonlocal capture_enabled
            if not capture_enabled:
                return
            out = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(out) or out.dim() < 4:
                raise RuntimeError(f"unexpected feature shape for layer ratio {lr_key}: {type(out)}")
            pooled = out.mean(dim=(2, 3)).detach().to("cpu", dtype=store_dtype)
            layer_capture[lr_key] = pooled

        return _hook

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for lr in layer_ratios:
        _li, _ln, lm = selected_layers[lr]
        handles.append(lm.register_forward_hook(_make_hook(lr)))

    repr_chunks: dict[str, list[torch.Tensor]] = {}
    for st in selected_step_indices:
        sr = step_idx_to_ratio[st]
        for lr in layer_ratios:
            key = f"s{st}_r{lr:g}"
            repr_chunks[key] = []

    sample_counter = 0
    n_batches = (int(args.num_samples) + int(args.batch_size) - 1) // int(args.batch_size)
    for bidx in tqdm(range(n_batches), desc="ddpm rollout + repr"):
        n_this = min(int(args.batch_size), int(args.num_samples) - sample_counter)
        if n_this <= 0:
            break
        gen = torch.Generator(device=device).manual_seed(int(args.seed) + bidx)

        image = torch.randn(
            (
                n_this,
                unet.config.in_channels,
                unet.config.sample_size,
                unet.config.sample_size,
            ),
            generator=gen,
            device=device,
            dtype=next(unet.parameters()).dtype,
        )
        pipe.scheduler.set_timesteps(int(args.num_inference_steps))
        timesteps = pipe.scheduler.timesteps

        for step_i, t in enumerate(timesteps):
            for lr in layer_ratios:
                layer_capture[lr] = None

            capture_enabled = step_i in selected_step_indices
            model_output = unet(image, t).sample
            image = pipe.scheduler.step(model_output, t, image, generator=gen).prev_sample
            capture_enabled = False

            if step_i in selected_step_indices:
                sr = step_idx_to_ratio[step_i]
                for lr in layer_ratios:
                    feat = layer_capture[lr]
                    if feat is None:
                        raise RuntimeError(f"missing capture at step={step_i}, layer_ratio={lr}")
                    repr_chunks[f"s{step_i}_r{lr:g}"].append(feat)

        pil_images = _to_pil_images(pipe, image)
        for im in pil_images:
            if sample_counter >= int(args.num_samples):
                break
            if args.image_size > 0 and im.size != (int(args.image_size), int(args.image_size)):
                im = im.resize((int(args.image_size), int(args.image_size)), Image.LANCZOS)
            im.save(output_image_dir / f"{sample_counter:05d}.png")
            sample_counter += 1

    for h in handles:
        h.remove()

    tensor_by_key: dict[str, torch.Tensor] = {}
    for k, ls in repr_chunks.items():
        if not ls:
            raise RuntimeError(f"empty repr list for key {k}")
        tensor_by_key[k] = torch.cat(ls, dim=0)
        if tensor_by_key[k].shape[0] > int(args.num_samples):
            tensor_by_key[k] = tensor_by_key[k][: int(args.num_samples)]

    payload = {
        "tensor_by_key": tensor_by_key,
        "meta": {
            "model_name": str(args.model_name),
            "num_samples": int(args.num_samples),
            "batch_size": int(args.batch_size),
            "num_inference_steps": int(args.num_inference_steps),
            "layer_ratios": [float(x) for x in layer_ratios],
            "step_ratios": [float(x) for x in step_ratios],
            "selected_step_indices": [int(x) for x in selected_step_indices],
            "seed": int(args.seed),
            "dtype": str(args.dtype),
            "store_dtype": str(args.store_dtype),
            "output_image_dir": str(output_image_dir),
            "saved_images": int(sample_counter),
            "layer_map": {
                f"{lr:g}": {
                    "module_index": int(selected_layers[lr][0]),
                    "module_name": str(selected_layers[lr][1]),
                }
                for lr in layer_ratios
            },
        },
    }
    torch.save(payload, output_repr_pt)

    if args.output_repr_dir is not None:
        for k, t in tensor_by_key.items():
            torch.save({"tensor": t, "meta": payload["meta"]}, args.output_repr_dir / f"{k}.pt")

    print(
        json.dumps(
            {
                "saved_repr_pt": str(output_repr_pt),
                "saved_images": int(sample_counter),
                "num_repr_keys": len(tensor_by_key),
                "repr_keys": sorted(tensor_by_key.keys()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
