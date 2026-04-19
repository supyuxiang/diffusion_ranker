#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMPipeline
from PIL import Image
from torchvision.datasets import CIFAR10
from tqdm import tqdm

_ROOT_ = Path(__file__).resolve().parent.parent
if str(_ROOT_) not in sys.path:
    sys.path.insert(0, str(_ROOT_))

from ranker import Ranker_Transformer
from utils import compute_fid, compute_is, set_device, set_seed


def _dtype_to_torch(dtype: str) -> torch.dtype:
    d = dtype.lower().strip()
    if d == "fp16":
        return torch.float16
    if d == "fp32":
        return torch.float32
    if d == "bf16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype: {dtype}")


def _latents_to_pil(x: torch.Tensor) -> list[Image.Image]:
    x = (x / 2 + 0.5).clamp(0, 1).float()
    arr = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype("uint8")
    return [Image.fromarray(im) for im in arr]


def _load_ranker(
    ckpt_path: Path,
    device: torch.device,
    *,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    input_channels: int,
    dropout: float,
) -> Ranker_Transformer:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model = Ranker_Transformer(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        input_channels=input_channels,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def _run_one_candidate(
    *,
    pipe: DDPMPipeline,
    base_scheduler,
    timesteps: torch.Tensor,
    target_layer_module,
    step_idx: int,
    batch_n: int,
    generator_seed: int,
    device: torch.device,
) -> tuple[list[Image.Image], torch.Tensor]:
    state = {"cur_step": -1, "hidden": None}

    def _hook(_m, _inp, _out):
        if state["cur_step"] != step_idx:
            return
        t = _out[0] if isinstance(_out, (tuple, list)) else _out
        state["hidden"] = t.detach().float()

    h = target_layer_module.register_forward_hook(_hook)
    try:
        sched = copy.deepcopy(base_scheduler)
        x = torch.randn(
            (
                batch_n,
                pipe.unet.config.in_channels,
                pipe.unet.config.sample_size,
                pipe.unet.config.sample_size,
            ),
            generator=torch.Generator(device=device).manual_seed(generator_seed),
            device=device,
            dtype=next(pipe.unet.parameters()).dtype,
        )
        for i, t in enumerate(timesteps):
            state["cur_step"] = i
            eps = pipe.unet(x, t).sample
            x = sched.step(eps, t, x, generator=None).prev_sample
            del eps
        images = _latents_to_pil(x)
        if state["hidden"] is None:
            raise RuntimeError(f"hook did not capture hidden at step_idx={step_idx}")
        return images, state["hidden"]
    finally:
        h.remove()
        gc.collect()
        torch.cuda.empty_cache()


def _load_cifar10_test_images(data_root: str | Path) -> torch.Tensor:
    ds = CIFAR10(root=str(Path(data_root).expanduser().resolve()), train=False, download=False)
    arr = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    return arr.to(torch.uint8)


def _load_images_from_dir_uint8(img_dir: str | Path) -> torch.Tensor:
    d = Path(img_dir).expanduser().resolve()
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)
    imgs: list[torch.Tensor] = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        imgs.append(t)
    if not imgs:
        raise FileNotFoundError(f"no images found under {d}")
    return torch.stack(imgs, dim=0).to(torch.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CIFAR-10 online random search with Ranker_Transformer")
    p.add_argument("--ranker_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/cifar10_random_search_ranker_trans")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/google_ddpm-cifar10-32")
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--parallel_size", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--layer_idx", type=int, default=12)
    p.add_argument("--step_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--input_channels", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.parallel_size <= 0:
        raise ValueError(f"parallel_size must be > 0, got {args.parallel_size}")

    set_seed(args.seed)
    device = set_device()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink()

    real_images = _load_cifar10_test_images(args.data_root)
    print(f"[info] loaded CIFAR10 test images: {tuple(real_images.shape)}")

    dtype = args.dtype
    if dtype == "bf16":
        print("[warn] bf16 is not supported in DDPMPipeline numpy conversion, fallback to fp16.")
        dtype = "fp16"
    pipe = DDPMPipeline.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=_dtype_to_torch(dtype),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    base_scheduler = copy.deepcopy(pipe.scheduler)
    timesteps = base_scheduler.timesteps

    resnet_blocks = [(n, m) for n, m in pipe.unet.named_modules() if m.__class__.__name__ == "ResnetBlock2D"]
    if args.layer_idx < 0 or args.layer_idx >= len(resnet_blocks):
        raise IndexError(f"layer_idx out of range: {args.layer_idx}, total={len(resnet_blocks)}")
    if args.step_idx < 0 or args.step_idx >= args.num_inference_steps:
        raise IndexError(f"step_idx out of range: {args.step_idx}, total_steps={args.num_inference_steps}")
    layer_name, layer_module = resnet_blocks[args.layer_idx]
    print(f"using layer_idx={args.layer_idx} name={layer_name}, step_idx={args.step_idx}")

    ranker = _load_ranker(
        Path(args.ranker_ckpt).expanduser().resolve(),
        device,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        input_channels=args.input_channels,
        dropout=args.dropout,
    )

    n_rounds = math.ceil(args.num_samples / args.batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    round_logs: list[dict[str, float | list[float]]] = []

    for ridx in tqdm(range(n_rounds), desc=f"online trans-ranker best-of-{args.parallel_size}"):
        batch_n = min(args.batch_size, args.num_samples - img_counter)
        if batch_n <= 0:
            break

        cand_images_all: list[list[Image.Image]] = []
        cand_hidden_all: list[torch.Tensor] = []
        cand_seed_all: list[int] = []

        for _k in range(args.parallel_size):
            cand_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            cand_images, cand_hidden = _run_one_candidate(
                pipe=pipe,
                base_scheduler=base_scheduler,
                timesteps=timesteps,
                target_layer_module=layer_module,
                step_idx=args.step_idx,
                batch_n=batch_n,
                generator_seed=cand_seed,
                device=device,
            )
            cand_images_all.append(cand_images)
            cand_hidden_all.append(cand_hidden)
            cand_seed_all.append(cand_seed)

        stacked_hidden = torch.stack(cand_hidden_all, dim=1).to(device, non_blocking=True)  # [B,K,C,H,W]
        pred = ranker(stacked_hidden)  # [B,K]
        mean_scores = pred.mean(dim=0).float().cpu()
        best_idx = int(torch.argmax(mean_scores).item())
        best_mean = float(mean_scores[best_idx].item())
        best_images = cand_images_all[best_idx]

        for im in best_images:
            if img_counter >= args.num_samples:
                break
            im.save(out_dir / f"{img_counter:05d}.png")
            img_counter += 1

        round_logs.append(
            {
                "round": float(ridx),
                "best_idx": float(best_idx),
                "best_batch_mean_reward": float(best_mean),
                "candidate_mean_scores": [float(v.item()) for v in mean_scores],
                "candidate_seeds": [float(s) for s in cand_seed_all],
            }
        )

        del stacked_hidden, pred, mean_scores
        for t in cand_hidden_all:
            del t
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gen_images = _load_images_from_dir_uint8(out_dir)
    metrics = {
        "fid": float(compute_fid(real_images, gen_images, device=device)),
    }
    is_mean, is_std = compute_is(gen_images, device=device)
    metrics["is"] = float(is_mean)
    metrics["is_std"] = float(is_std)
    metrics["num_generated"] = float(img_counter)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "round_log.json").write_text(json.dumps(round_logs, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
