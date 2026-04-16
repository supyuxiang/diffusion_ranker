#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# Hard-coded targets (precomputed from ratios for num_inference_steps=25 and current UNet block count):
# layer ratios [0.1, 0.3, 0.6, 0.9, 1.0] -> layer indices [3, 9, 19, 28, 31]
# step ratios  [0.0, 0.25, 0.5, 0.75, 1.0] -> step indices  [0, 6, 12, 18, 24]
_TARGET_LAYER_INDICES = [3, 9, 19, 28, 31]
_TARGET_STEP_INDICES = [0, 6, 12, 18, 24]


def build_clip(device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    clip_path = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")
    clip = CLIPModel.from_pretrained(
        str(clip_path),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    clip.eval()
    clip_processor = CLIPProcessor.from_pretrained(
        str(clip_path),
        local_files_only=True,
    )
    return clip, clip_processor


@torch.inference_mode()
def reward_clip(clip, clip_processor, images: list[Image.Image]) -> torch.Tensor:

    prompt = "a photo of a human face"
    prompts = [prompt] * len(images)
    device = next(clip.parameters()).device
    inputs = clip_processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = clip(**inputs)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True, p=2)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True, p=2)
    sim = (image_embeds * text_embeds).sum(dim=-1)
    return sim.float().cpu()


def _latents_to_pil(x: torch.Tensor) -> list[Image.Image]:
    x = (x / 2 + 0.5).clamp(0, 1)
    arr = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype("uint8")
    return [Image.fromarray(im) for im in arr]


@torch.inference_mode()
def mc(
    step_idx: int,
    batch_middle_x: torch.Tensor,
    pipe: DDPMPipeline,
    clip,
    clip_processor,
    mc_samples: int,
    base_seed: int,
) -> torch.Tensor:
    """
    Start from x at middle step and rollout to final step for MC reward.
    Returns per-sample average reward, shape [B].
    """
    
    device = batch_middle_x.device
    timesteps = pipe.scheduler.timesteps
    reward_ls: list[torch.Tensor] = []
    for j in range(mc_samples):
        # keep scheduler state isolated across MC runs
        sched = copy.deepcopy(pipe.scheduler)
        x = batch_middle_x.clone()
        gen = torch.Generator(device=device).manual_seed(base_seed + j)
        for t in timesteps[step_idx + 1 :]:
            eps = pipe.unet(x, t).sample
            x = sched.step(eps, t, x, generator=gen).prev_sample
        images = _latents_to_pil(x)
        reward_ls.append(reward_clip(clip, clip_processor, images))

    return torch.stack(reward_ls, dim=0).mean(dim=0)


@torch.inference_mode()
def rollout() -> None:
    ddpm_path = Path("/root/autodl-tmp/google_ddpm-celebahq-256")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = DDPMPipeline.from_pretrained(
        str(ddpm_path),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    clip, clip_processor = build_clip(device)

    num_samples = 5000
    batch_size = 32
    num_inference_steps = 25
    mc_samples = 4 
    seed = 42

    generator = torch.Generator(device=device).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    resnet_blocks = [
        (name, mod)
        for name, mod in pipe.unet.named_modules()
        if mod.__class__.__name__ == "ResnetBlock2D"
    ]
    if not resnet_blocks:
        raise RuntimeError("no ResnetBlock2D found")

    # Use hard-coded indices directly (no runtime ratio-to-index conversion).
    target_layer_indices = sorted(set(int(x) for x in _TARGET_LAYER_INDICES))
    target_step_indices = sorted(set(int(x) for x in _TARGET_STEP_INDICES))
    if any(li < 0 or li >= len(resnet_blocks) for li in target_layer_indices):
        raise ValueError(
            f"target_layer_indices out of range: {target_layer_indices}, "
            f"available ResnetBlock2D count={len(resnet_blocks)}"
        )
    if any(si < 0 or si >= num_inference_steps for si in target_step_indices):
        raise ValueError(
            f"target_step_indices out of range: {target_step_indices}, "
            f"num_inference_steps={num_inference_steps}"
        )

    # Capture ALL steps (for selected layers), then filter target steps later.
    all_hidden_store: dict[int, dict[int, list[torch.Tensor]]] = {
        s: {li: [] for li in target_layer_indices} for s in range(num_inference_steps)
    }

    current_step_idx = {"value": -1}

    def _make_hook(layer_idx: int):
        def _hook(_module, _inp, _out):
            s = current_step_idx["value"]
            if s < 0:
                return
            t = _out[0] if isinstance(_out, (tuple, list)) else _out
            # pool spatial dims -> [B, C] for compact storage
            pooled = t.detach().float().mean(dim=(2, 3)).cpu()
            all_hidden_store[s][layer_idx].append(pooled)

        return _hook

    handles = [
        resnet_blocks[li][1].register_forward_hook(_make_hook(li))
        for li in target_layer_indices
    ]

    reward_store: dict[int, list[torch.Tensor]] = {s: [] for s in target_step_indices}

    n_batches = num_samples // batch_size
    pbar = tqdm(range(n_batches), total=n_batches, desc="Running DDPM rollout")
    for batch_i in pbar:
        x = torch.randn(
            (
                batch_size,
                pipe.unet.config.in_channels,
                pipe.unet.config.sample_size,
                pipe.unet.config.sample_size,
            ),
            generator=generator,
            device=device,
            dtype=next(pipe.unet.parameters()).dtype,
        )

        for st_i, t in enumerate(timesteps):
            current_step_idx["value"] = st_i
            eps = pipe.unet(x, t).sample
            x = pipe.scheduler.step(eps, t, x, generator=generator).prev_sample
            if st_i in target_step_indices:
                reward_store[st_i].append(
                    mc(
                        step_idx=st_i,
                        batch_middle_x=x,
                        pipe=pipe,
                        clip=clip,
                        clip_processor=clip_processor,
                        mc_samples=mc_samples,
                        base_seed=seed + batch_i * 1000 + st_i * 10,
                    )
                )

    for h in handles:
        h.remove()

    # 1) rewards on target steps
    reward_store_tensor = {
        f"s{st_i}": torch.cat(reward_store[st_i], dim=0) if reward_store[st_i] else torch.empty(0)
        for st_i in target_step_indices
    }

    # 2) filter target step/layer from all-step hidden store
    hidden_store_target = {}
    for st_i in target_step_indices:
        for li in target_layer_indices:
            chunks = all_hidden_store[st_i][li]
            key = f"s{st_i}_l{li}"
            hidden_store_target[key] = torch.cat(chunks, dim=0) if chunks else torch.empty(0)

    full_data = {
        "hidden_store_target": hidden_store_target,
        "reward_store": reward_store_tensor,
        "meta": {
            "target_layer_indices": target_layer_indices,
            "target_step_indices": target_step_indices,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "num_inference_steps": num_inference_steps,
            "mc_samples": mc_samples,
            "seed": seed,
        },
    }

    save_path = Path("/root/autodl-tmp/ddpm_hidden_store_demo.pth")
    torch.save(full_data, save_path)
    print(json.dumps({"saved": str(save_path), "num_keys": len(hidden_store_target)}, ensure_ascii=False))


if __name__ == "__main__":
    rollout()
