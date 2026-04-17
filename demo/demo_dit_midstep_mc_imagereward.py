#!/usr/bin/env python3
r"""
估计 **给定 step ratio** 处 latent 的 **ImageReward 期望效用**（蒙特卡洛样本均值）。

做法（与 ``DitRolloutReprCapture`` 的步下标一致）::

  ``k = round(step_ratio * (T - 1))`` 表示已完成 **k** 次 ``scheduler.step``（``k∈[0,T-1]``），
  在该 latent 上固定后，对剩余 ``T-k`` 步各做 **MC 次** 独立 ``variance_noise`` 延续（``DPMSolverMultistep``），
  解码成图后用 ImageReward 打分，**utility ≈ 这 MC 张图的平均分**。

示例::

  CUDA_VISIBLE_DEVICES=0 python3 demo/demo_dit_midstep_mc_imagereward.py \
    --dit_path /root/autodl-tmp/DiT-XL-2-256 \
    --step_ratio 0.5 \
    --mc 4 \
    --num_inference_steps 10 \
    --class_id 281 \
    --seed 42

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SEARCH = _REPO / "search"
if str(_SEARCH) not in sys.path:
    sys.path.insert(0, str(_SEARCH))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from diffusers.utils import randn_tensor
from PIL import Image

from utils import dit_loader, ensure_image_reward_on_path, set_device, set_seed


def step_ratio_to_k(step_ratio: float, num_inference_steps: int) -> int:
    t = int(num_inference_steps)
    if t <= 1:
        return 0
    return min(t - 1, max(0, round(float(step_ratio) * (t - 1))))


@torch.inference_mode()
def _dit_denoise_prefix(
    pipe,
    *,
    class_labels: list[int],
    guidance_scale: float,
    generator: torch.Generator,
    num_inference_steps: int,
    k_completed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """跑前 ``k_completed`` 个 scheduler 步，返回 ``(latent_model_input, class_labels_input, latent_channels, timesteps)``。"""
    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    latents = randn_tensor(
        shape=(batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=pipe._execution_device,
        dtype=pipe.transformer.dtype,
    )
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

    class_labels_t = torch.tensor(class_labels, device=pipe._execution_device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=pipe._execution_device)
    class_labels_input = (
        torch.cat([class_labels_t, class_null], 0) if guidance_scale > 1 else class_labels_t
    )

    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    k = min(k_completed, len(timesteps))
    for t in timesteps[:k]:
        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps_b = _broadcast_timesteps(t, latent_model_input)
        noise_pred = pipe.transformer(
            latent_model_input, timestep=timesteps_b, class_labels=class_labels_input
        ).sample

        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            noise_pred = torch.cat([eps, rest], dim=1)

        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        latent_model_input = pipe.scheduler.step(
            model_output, t, latent_model_input, generator=generator
        ).prev_sample

    return latent_model_input, class_labels_input, latent_channels, timesteps


def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        is_mps = latent_model_input.device.type == "mps"
        is_npu = latent_model_input.device.type == "npu"
        if isinstance(t, float):
            dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        else:
            dtype = torch.int32 if (is_mps or is_npu) else torch.int64
        timesteps = torch.tensor([t], dtype=dtype, device=latent_model_input.device)
    elif len(t.shape) == 0:
        timesteps = t[None].to(latent_model_input.device)
    else:
        timesteps = t
    return timesteps.expand(latent_model_input.shape[0])


@torch.inference_mode()
def _dit_denoise_suffix_mc(
    pipe,
    *,
    latent_model_input: torch.Tensor,
    class_labels_input: torch.Tensor,
    class_labels: list[int],
    guidance_scale: float,
    latent_channels: int,
    timesteps: torch.Tensor,
    k_completed: int,
    mc_index: int,
    base_seed: int,
) -> Image.Image:
    """从 ``timesteps[k_completed:]`` 继续解码；用独立 ``variance_noise`` 体现 MC 分叉。"""
    device = latent_model_input.device
    gen_mc = torch.Generator(device=device).manual_seed(int(base_seed) + 10_000 + mc_index * 9973)
    x = latent_model_input.clone()
    rest = timesteps[k_completed:]
    for t in rest:
        if guidance_scale > 1:
            half = x[: len(x) // 2]
            x = torch.cat([half, half], dim=0)
        x = pipe.scheduler.scale_model_input(x, t)
        timesteps_b = _broadcast_timesteps(t, x)
        noise_pred = pipe.transformer(
            x, timestep=timesteps_b, class_labels=class_labels_input
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

        vn = randn_tensor(
            x.shape,
            generator=gen_mc,
            device=device,
            dtype=x.dtype,
            layout=torch.strided,
        )
        x = pipe.scheduler.step(
            model_output, t, x, generator=gen_mc, variance_noise=vn
        ).prev_sample

    if guidance_scale > 1:
        latents, _ = x.chunk(2, dim=0)
    else:
        latents = x
    latents = 1 / pipe.vae.config.scaling_factor * latents
    samples = pipe.vae.decode(latents).sample
    samples = (samples / 2 + 0.5).clamp(0, 1)
    arr = samples[0].cpu().permute(1, 2, 0).float().numpy()
    return Image.fromarray((arr * 255.0).round().clip(0, 255).astype("uint8"))


def main() -> None:
    p = argparse.ArgumentParser(description="DiT mid-step ratio MC + ImageReward utility demo")
    p.add_argument("--dit_path", type=str, default=str(Path("/root/autodl-tmp/DiT-XL-2-256")))
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--step_ratio", type=float, default=0.5)
    p.add_argument("--mc", type=int, default=4, help="蒙特卡洛次数")
    p.add_argument("--class_id", type=int, default=281, help="ImageNet 类 id（0..999）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    dit_path = Path(args.dit_path).expanduser().resolve()
    if not dit_path.is_dir():
        raise SystemExit(f"DiT 目录不存在: {dit_path}")

    set_seed(args.seed)
    device = set_device()
    dtype_s = "fp16" if args.fp16 else "fp32"
    pipe = dit_loader(dit_path, device, dtype=dtype_s)
    pipe.set_progress_bar_config(disable=True)

    k = step_ratio_to_k(args.step_ratio, args.num_inference_steps)
    gen0 = torch.Generator(device=device).manual_seed(args.seed)

    latent_mid, class_in, latent_c, timesteps = _dit_denoise_prefix(
        pipe,
        class_labels=[args.class_id],
        guidance_scale=args.guidance_scale,
        generator=gen0,
        num_inference_steps=args.num_inference_steps,
        k_completed=k,
    )

    from reward import labels_1k, load_imagereward

    prompt = f"A photo of a {labels_1k[args.class_id]}."
    ensure_image_reward_on_path()
    rm = load_imagereward(device)

    scores: list[float] = []
    pil_images: list[Image.Image] = []
    for i in range(args.mc):
        pil = _dit_denoise_suffix_mc(
            pipe,
            latent_model_input=latent_mid,
            class_labels_input=class_in,
            class_labels=[args.class_id],
            guidance_scale=args.guidance_scale,
            latent_channels=latent_c,
            timesteps=timesteps,
            k_completed=k,
            mc_index=i,
            base_seed=args.seed,
        )
        pil_images.append(pil)
        scores.append(float(rm.score(prompt, pil)))

    mean_u = sum(scores) / len(scores)
    print(
        f"step_ratio={args.step_ratio}  ->  k_completed={k} / T={args.num_inference_steps}  "
        f"(与 DitRolloutReprCapture: round(r*(T-1)))"
    )
    print(f"class_id={args.class_id}  prompt={prompt!r}")
    print(f"MC={args.mc}  ImageReward scores={['%.4f' % s for s in scores]}")
    print(f"utility_MC_mean={mean_u:.6f}")


if __name__ == "__main__":
    main()
