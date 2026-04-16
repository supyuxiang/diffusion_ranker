#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm

_ROOT_ = Path(__file__).resolve().parent.parent
if str(_ROOT_) not in sys.path:
    sys.path.insert(0, str(_ROOT_))

from ranker import Probe_MLP
from search.reward import load_imagereward
from utils import compute_fid_is_clipscore_for_uncondition, set_device, set_seed


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


def _load_probe(
    ckpt_path: Path,
    device: torch.device,
    *,
    num_layers: int,
    hidden_dim: int,
    input_channels: int,
    dropout: float,
) -> Probe_MLP:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model = Probe_MLP(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
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
    pool_kernel: int,
    pool_stride: int,
) -> tuple[list[Image.Image], torch.Tensor]:
    state = {"cur_step": -1, "hidden": None}

    def _hook(_m, _inp, _out):
        if state["cur_step"] != step_idx:
            return
        t = _out[0] if isinstance(_out, (tuple, list)) else _out
        t = t.detach().float()
        if t.dim() == 4 and t.shape[-2] >= pool_kernel and t.shape[-1] >= pool_kernel:
            t = F.avg_pool2d(t, kernel_size=pool_kernel, stride=pool_stride)
        state["hidden"] = t

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
        hidden = state["hidden"]
        return images, hidden
    finally:
        h.remove()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _mean_imagereward_for_dir(gen_dir: Path, device: torch.device) -> float:
    rm = load_imagereward(device)
    prompt = "a photo of a human face"
    paths = sorted(gen_dir.glob("*.png"))
    vals: list[float] = []
    for p in tqdm(paths, desc="ImageReward(final)"):
        img = Image.open(p).convert("RGB")
        vals.append(float(rm.score(prompt, img)))
    return float(sum(vals) / max(1, len(vals)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Online random search with Probe reward")
    p.add_argument("--probe_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/uncond_random_search_probe_ps2")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/google_ddpm-celebahq-256")
    p.add_argument("--real_dir", type=str, default="/root/autodl-tmp/img_align_celeba_fid_ref_50k")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--parallel_size", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--layer_idx", type=int, default=19)
    p.add_argument("--step_idx", type=int, default=24)
    p.add_argument("--pool_kernel", type=int, default=2)
    p.add_argument("--pool_stride", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--input_channels", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip_model_dir", type=str, default="/root/autodl-tmp/openai_clip-vit-large-patch14")
    p.add_argument("--clip_batch_size", type=int, default=64)
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

    pipe = DDPMPipeline.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=_dtype_to_torch(args.dtype),
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

    probe = _load_probe(
        Path(args.probe_ckpt).expanduser().resolve(),
        device,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        input_channels=args.input_channels,
        dropout=args.dropout,
    )

    n_rounds = math.ceil(args.num_samples / args.batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    round_logs: list[dict[str, float]] = []
    for ridx in tqdm(range(n_rounds), desc=f"online best-of-{args.parallel_size}"):
        batch_n = min(args.batch_size, args.num_samples - img_counter)
        if batch_n <= 0:
            break

        best_mean = float("-inf")
        best_images: list[Image.Image] | None = None

        for k in range(args.parallel_size):
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
                pool_kernel=args.pool_kernel,
                pool_stride=args.pool_stride,
            )
            pred = probe(cand_hidden.to(device, non_blocking=True))
            mean_score = float(pred.mean().item())
            if mean_score > best_mean:
                best_mean = mean_score
                best_images = cand_images
            else:
                del cand_images
            del cand_hidden, pred
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        assert best_images is not None
        for im in best_images:
            if img_counter >= args.num_samples:
                break
            im.save(out_dir / f"{img_counter:05d}.png")
            img_counter += 1
        round_logs.append({"round": float(ridx), "best_batch_mean_reward": float(best_mean)})

    metrics = compute_fid_is_clipscore_for_uncondition(
        gen_dir=out_dir,
        real_dir=Path(args.real_dir).expanduser().resolve(),
        clip_model_dir=Path(args.clip_model_dir).expanduser().resolve(),
        clip_batch_size=args.clip_batch_size,
    )
    metrics["imagereward"] = _mean_imagereward_for_dir(out_dir, device)
    metrics["num_generated"] = float(img_counter)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "round_log.json").write_text(json.dumps(round_logs, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
