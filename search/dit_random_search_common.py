"""
DiT Random Search 脚本共用：随机类条件 rollout、ImageNet 标签、init_random_samples。

- 默认 ``class_ids_from_generator=True``：class 与 noise 共用同一 ``torch.Generator``（与 ImageReward / CLIP / Qwen 等脚本一致）。
- ``class_ids_from_generator=False``：先 ``torch.randint``（全局 RNG），再对 latent 单独 ``Generator``（与 ``dit_random_search_fid_reward.py`` 历史行为一致）。
- ``rollout_dit_batch_with_pooled_hidden``：供 ``dit_random_search_ranker`` 在最后一步 hook 指定 block 并 GAP（与训练时 ranker 一致）。
"""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Callable

import torch
from diffusers import DiTPipeline
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import _prepare_pil_for_save, imagenet_simple_labels_list, model_loader

##########################################
IMAGENET_NUM_CLASSES = 1000
NULL_CLASS_ID = 1000
OUTPUT_IMAGE_SIZE = 256
INIT_BATCH_SEED_STRIDE = 10_000
CANDIDATE_SEED_HIGH = 2**31
##########################################



def load_imagenet_simple_labels(cache_dir: Path | None = None) -> list[str]:
    """与 ``utils.imagenet_simple_labels_list`` 同源（仓库 ``imagenet-simple-labels.json``）。

    ``cache_dir`` 已弃用，保留仅为兼容旧调用。
    """
    _ = cache_dir
    try:
        return imagenet_simple_labels_list()
    except (FileNotFoundError, ValueError) as e:
        print(f"警告: ImageNet 标签未就绪 ({e})，将使用占位名。请运行 scripts/build_imagenet_simple_labels.py")
        return [f"imagenet class {i}" for i in range(1000)]


def prompt_for_class(labels: list[str], class_id: int, template: str) -> str:
    assert 0 <= class_id < len(labels)
    name = labels[class_id]
    return template.format(name=name, id=class_id)


def _expand_timestep(t: torch.Tensor | int, batch_size: int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(t):
        return torch.tensor([t], dtype=torch.long, device=device).expand(batch_size)
    return (t.expand(batch_size) if t.dim() == 0 else t).to(device)


def _merge_cfg_noise_pred(
    noise_pred: torch.Tensor,
    latent_channels: int,
    guidance_scale: float,
) -> torch.Tensor:
    if guidance_scale <= 1:
        return noise_pred
    eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    merged = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    # Latents stay doubled through the denoise loop; scheduler.step expects model_output
    # batch size to match latents. Duplicate merged eps so it matches ``rest`` and latents.
    merged = torch.cat([merged, merged], dim=0)
    return torch.cat([merged, rest], dim=1)


def _scheduler_model_output(noise_pred: torch.Tensor, pipe: DiTPipeline, latent_channels: int) -> torch.Tensor:
    if pipe.transformer.config.out_channels // 2 == latent_channels:
        model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        return model_output
    return noise_pred


@contextmanager
def _forward_hook(module: torch.nn.Module, hook_fn: Callable) -> AbstractContextManager[None]:
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def _prepare_random_class_latents(
    pipe: DiTPipeline,
    batch_size: int,
    device: str,
    seed: int,
    guidance_scale: float,
    *,
    class_ids_from_generator: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size
    if class_ids_from_generator:
        gen = torch.Generator(device=device).manual_seed(seed)
        class_ids = torch.randint(
            0, IMAGENET_NUM_CLASSES, (batch_size,), device=device, generator=gen
        ).tolist()
        latents = randn_tensor(
            (batch_size, latent_channels, latent_size, latent_size),
            generator=gen,
            device=device,
            dtype=pipe.transformer.dtype,
        )
    else:
        class_ids = torch.randint(0, IMAGENET_NUM_CLASSES, (batch_size,), device=device).tolist()
        gen = torch.Generator(device=device).manual_seed(seed)
        latents = randn_tensor(
            (batch_size, latent_channels, latent_size, latent_size),
            generator=gen,
            device=device,
            dtype=pipe.transformer.dtype,
        )
    if guidance_scale > 1:
        latents = torch.cat([latents] * 2, dim=0)
    class_labels = torch.tensor(class_ids, device=device, dtype=torch.long).reshape(-1)
    class_null = torch.tensor([NULL_CLASS_ID] * batch_size, device=device)
    labels_in = torch.cat([class_labels, class_null], dim=0) if guidance_scale > 1 else class_labels
    return latents, labels_in, class_ids, latent_channels


def _dit_denoise_loop(
    pipe: DiTPipeline,
    num_steps: int,
    latents: torch.Tensor,
    class_labels_input: torch.Tensor,
    latent_channels: int,
    guidance_scale: float,
    *,
    step_forward_ctx: Callable[[int, int], AbstractContextManager[None]] | None = None,
) -> torch.Tensor:
    """``step_forward_ctx(step_i, n_steps)`` 包住该步 transformer 前向；``None`` 表示全程 ``nullcontext``。"""
    pipe.scheduler.set_timesteps(num_steps)
    timesteps = list(pipe.scheduler.timesteps)
    n_t = len(timesteps)
    for step_i, t in enumerate(timesteps):
        if guidance_scale > 1:
            half = latents[: len(latents) // 2]
            latents = torch.cat([half, half], dim=0)
        latent_input = pipe.scheduler.scale_model_input(latents, t)
        ts = _expand_timestep(t, latent_input.shape[0], latent_input.device)
        ctx = step_forward_ctx(step_i, n_t) if step_forward_ctx is not None else nullcontext()
        with ctx:
            noise_pred = pipe.transformer(
                latent_input, timestep=ts, class_labels=class_labels_input
            ).sample
        noise_pred = _merge_cfg_noise_pred(
            noise_pred,
            latent_channels,
            guidance_scale,
        )
        model_output = _scheduler_model_output(noise_pred, pipe, latent_channels)
        latents = pipe.scheduler.step(model_output, t, latents).prev_sample
    return latents


def _decode_latents_to_pil(
    pipe: DiTPipeline,
    latents: torch.Tensor,
    batch_size: int,
    guidance_scale: float,
) -> list[Image.Image]:
    to_decode = latents[:batch_size] if guidance_scale > 1 else latents
    to_decode = to_decode / pipe.vae.config.scaling_factor
    samples = pipe.vae.decode(to_decode).sample
    samples = (samples / 2 + 0.5).clamp(0, 1)
    samples_np = samples.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    images = pipe.numpy_to_pil(samples_np)
    return [_prepare_pil_for_save(img, OUTPUT_IMAGE_SIZE) for img in images]


def rollout_dit_batch(
    pipe: DiTPipeline,
    num_steps: int,
    batch_size: int,
    device: str,
    seed: int,
    guidance_scale: float = 1.0,
    *,
    class_ids_from_generator: bool = True,
) -> tuple[list[Image.Image], list[int]]:
    latents, labels_in, class_ids, lc = _prepare_random_class_latents(
        pipe, batch_size, device, seed, guidance_scale, class_ids_from_generator=class_ids_from_generator
    )
    with torch.no_grad():
        latents = _dit_denoise_loop(
            pipe,
            num_steps,
            latents,
            labels_in,
            lc,
            guidance_scale,
            step_forward_ctx=None,
        )
    images = _decode_latents_to_pil(pipe, latents, batch_size, guidance_scale)
    return images, class_ids


def rollout_dit_batch_with_pooled_hidden(
    pipe: DiTPipeline,
    num_steps: int,
    batch_size: int,
    device: str,
    seed: int,
    hidden_layer_idx: int,
    guidance_scale: float = 1.0,
) -> tuple[list[Image.Image], list[int], torch.Tensor]:
    """
    与 ``rollout_dit_batch`` 相同随机类与噪声；最后一步去噪时对 ``transformer_blocks[hidden_layer_idx]``
    注册 forward hook，对输出做 GAP，得到 ``pooled`` [B, D]（CPU float32）。
    """
    blocks = pipe.transformer.transformer_blocks
    if not (0 <= hidden_layer_idx < len(blocks)):
        raise ValueError(f"hidden_layer_idx={hidden_layer_idx} out of range for {len(blocks)} blocks")

    latents, labels_in, class_ids, lc = _prepare_random_class_latents(
        pipe, batch_size, device, seed, guidance_scale, class_ids_from_generator=True
    )
    capture: list[torch.Tensor | None] = [None]

    def hook_fn(_module, _inp, out):
        capture[0] = out.detach()

    def step_ctx(step_i: int, n_steps: int) -> AbstractContextManager[None]:
        if step_i == n_steps - 1:
            return _forward_hook(blocks[hidden_layer_idx], hook_fn)
        return nullcontext()

    with torch.no_grad():
        latents = _dit_denoise_loop(
            pipe,
            num_steps,
            latents,
            labels_in,
            lc,
            guidance_scale,
            step_forward_ctx=step_ctx,
        )

    hidden = capture[0]
    if hidden is None:
        raise RuntimeError("未能捕获指定 block 的表征（请确认 deno_steps>=1）。")
    if guidance_scale > 1:
        hidden = hidden[: hidden.shape[0] // 2]
    pooled = hidden.float().mean(dim=1).cpu()
    images = _decode_latents_to_pil(pipe, latents, batch_size, guidance_scale)
    return images, class_ids, pooled


def rollout_dit_batch_fid_images(
    pipe: DiTPipeline,
    num_steps: int,
    batch_size: int,
    device: str,
    seed: int,
    guidance_scale: float = 1.0,
) -> list[Image.Image]:
    """FID 版历史行为：类别来自全局 RNG，latent 来自 ``Generator``；仅返回图像。"""
    imgs, _ = rollout_dit_batch(
        pipe, num_steps, batch_size, device, seed, guidance_scale, class_ids_from_generator=False
    )
    return imgs


def init_random_samples(
    pipe: DiTPipeline,
    output_dir: Path,
    init_samples: int,
    batch_size: int,
    deno_steps: int,
    device: str,
    seed: int,
    guidance_scale: float = 1.0,
    *,
    class_ids_from_generator: bool = True,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    img_counter = 0
    num_batches = (init_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="随机初始化"):
        n_this = min(batch_size, init_samples - img_counter)
        if n_this <= 0:
            break
        sub_seed = seed + batch_idx * INIT_BATCH_SEED_STRIDE
        if class_ids_from_generator:
            batch_images, _ = rollout_dit_batch(
                pipe, deno_steps, n_this, device, sub_seed, guidance_scale=guidance_scale
            )
        else:
            batch_images = rollout_dit_batch_fid_images(
                pipe, deno_steps, n_this, device, sub_seed, guidance_scale=guidance_scale
            )
        for img in batch_images:
            if img_counter >= init_samples:
                break
            im = _prepare_pil_for_save(img, OUTPUT_IMAGE_SIZE)
            im.save(output_dir / f"{img_counter:06d}.png")
            img_counter += 1
    return img_counter


__all__ = [
    "CANDIDATE_SEED_HIGH",
    "INIT_BATCH_SEED_STRIDE",
    "IMAGENET_NUM_CLASSES",
    "NULL_CLASS_ID",
    "OUTPUT_IMAGE_SIZE",
    "init_random_samples",
    "model_loader",
    "load_imagenet_simple_labels",
    "prompt_for_class",
    "rollout_dit_batch",
    "rollout_dit_batch_fid_images",
    "rollout_dit_batch_with_pooled_hidden",
]
