#!/usr/bin/env python3

'''

rollout:
python "/root/diffusion_ranker/text2image/rollout_pixart_hiddens_rewards.py" \
  --model_path "/root/autodl-tmp/PixArt-XL-2-1024-MS" \
  --prompts_json "/root/autodl-tmp/prompt_datasets/genai_image_1600_list.json" \
  --output_root "/root/autodl-tmp/text2image_pixart_hs_reward_rollout_1600_v2" \
  --repeats_per_prompt 500 \
  --batch_size 4 \
  --num_inference_steps 20 \
  --guidance_scale 4.5 \
  --negative_prompt "blurry, low quality, watermark, text, deformed" \
  --height 1024 --width 1024 \
  --target_step_idx 10 --target_layer_idx 14 \
  --mc 5 --dtype fp16 --seed 42

train ranker:

python "/root/diffusion_ranker/text2image/train_pixart_prompt_ranker.py" \
  --hs_reward_path "/root/autodl-tmp/text2image_pixart_hs_reward_rollout_1600_v2/pixart_prompt_hs_reward_n4216_s10_l14_mc5.pt" \
  --model_path "/root/autodl-tmp/PixArt-XL-2-1024-MS" \
  --output_dir "/root/autodl-tmp/text2image_pixart_prompt_ranker_1600_s10_l14_mc5" \
  --seed 42 \
  --dtype fp16 \
  --train_ratio 0.9 \
  --ranker_type listwise \
  --hidden_dim 128 \
  --num_heads 4 \
  --num_layers 2 \
  --dropout 0.1 \
  --batch_size 32 \
  --parallel_size 8 \
  --num_bootstrap 8 \
  --num_epochs 100 \
  --eval_interval 1 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --alpha 0.8 \
  --kl_type kl

'''
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import ensure_image_reward_on_path, set_seed

_DEFAULT_IMAGEREWARD_DIR = Path("/root/autodl-tmp/ImageReward")
_DEFAULT_IMAGEREWARD_CKPT = _DEFAULT_IMAGEREWARD_DIR / "ImageReward.pt"
_DEFAULT_IMAGEREWARD_MED = _DEFAULT_IMAGEREWARD_DIR / "med_config.json"


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _safe_name(s: str, max_len: int = 64) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_\\-]", "", s)
    return (s[:max_len] or "prompt")


def _load_prompts(json_path: Path) -> list[str]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list) or len(obj) == 0:
        raise ValueError("prompts json must be a non-empty list.")
    prompts: list[str] = []
    for i, x in enumerate(obj):
        if isinstance(x, str):
            p = x.strip()
        elif isinstance(x, dict):
            p = str(x.get("prompt", "")).strip()
        else:
            raise ValueError(f"Unsupported prompt item at index={i}: {type(x)}")
        if not p:
            raise ValueError(f"Empty prompt at index={i}")
        prompts.append(p)
    return prompts


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
    return pipe.image_processor.postprocess(image, output_type="pil")


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


def _imagereward_batch(images: list[Image.Image], prompts: list[str], model) -> torch.Tensor:
    vals = [float(model.score(p, img)) for p, img in zip(prompts, images)]
    return torch.tensor(vals, dtype=torch.float32)


@torch.inference_mode()
def _mc_imagereward_from_snapshot(
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
    prompts: list[str],
    rm_model,
) -> torch.Tensor:
    if start_step_idx >= len(timesteps):
        imgs = _decode_pixart_latents_to_pil(pipe, latents_start)
        return _imagereward_batch(imgs, prompts, rm_model)

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
        rewards.append(_imagereward_batch(imgs, prompts, rm_model))

    return torch.stack(rewards, dim=0).mean(dim=0)


@torch.inference_mode()
def rollout_hiddens_rewards(args: argparse.Namespace) -> None:
    prompts = _load_prompts(Path(args.prompts_json).expanduser().resolve())
    expanded_prompts: list[str] = []
    expanded_prompt_idx: list[int] = []
    for idx, p in enumerate(prompts):
        for _ in range(args.repeats_per_prompt):
            expanded_prompts.append(p)
            expanded_prompt_idx.append(idx)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (output_root / "images").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _resolve_dtype(args.dtype, device)
    do_cfg = args.guidance_scale > 1.0

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    rm_model = load_imagereward(device=device)

    n_blocks = len(pipe.transformer.transformer_blocks)
    if not (0 <= args.target_layer_idx < n_blocks):
        raise ValueError(f"target_layer_idx must be in [0, {n_blocks-1}]")
    if not (0 <= args.target_step_idx < args.num_inference_steps):
        raise ValueError(f"target_step_idx must be in [0, {args.num_inference_steps-1}]")

    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size

    all_hidden: list[torch.Tensor] = []
    all_reward: list[torch.Tensor] = []
    all_seed: list[int] = []
    all_image_names: list[str] = []

    cur_step = {"value": -1}
    cur_batch = {"value": 0}
    hidden_buf: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out):
        if cur_step["value"] != args.target_step_idx:
            return
        h = out[0] if isinstance(out, (tuple, list)) else out
        if do_cfg:
            h = h[: cur_batch["value"]]
        hidden_buf["h"] = h.mean(dim=1).detach().float().cpu()

    handle = pipe.transformer.transformer_blocks[args.target_layer_idx].register_forward_hook(_hook)
    try:
        pbar = tqdm(range(0, len(expanded_prompts), args.batch_size), desc="rollout hs+reward")
        for st in pbar:
            ed = min(st + args.batch_size, len(expanded_prompts))
            batch_prompts = expanded_prompts[st:ed]
            bsz = len(batch_prompts)
            if bsz <= 0:
                continue
            cur_batch["value"] = bsz

            batch_seed = args.seed + st
            generator = torch.Generator(device=device).manual_seed(batch_seed)

            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
                pipe.encode_prompt(
                    batch_prompts,
                    do_classifier_free_guidance=do_cfg,
                    negative_prompt=args.negative_prompt,
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
                height=args.height,
                width=args.width,
                batch=bsz,
                do_cfg=do_cfg,
                device=device,
                dtype=prompt_embeds.dtype,
            )
            latents = torch.randn(
                (bsz, latent_channels, latent_size, latent_size),
                generator=generator,
                device=device,
                dtype=pipe.transformer.dtype,
            )
            pipe.scheduler.set_timesteps(int(args.num_inference_steps), device=device)
            timesteps = pipe.scheduler.timesteps
            hidden_buf.clear()

            snapshot_latents = None
            snapshot_scheduler = None
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
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                if pipe.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                latents = pipe.scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]
                if step_i == args.target_step_idx:
                    snapshot_latents = latents.detach().clone()
                    snapshot_scheduler = copy.deepcopy(pipe.scheduler)

            if "h" not in hidden_buf:
                raise RuntimeError(f"hook failed for prompts[{st}:{ed}]")
            if snapshot_latents is None or snapshot_scheduler is None:
                raise RuntimeError(f"snapshot failed at step={args.target_step_idx}")

            reward = _mc_imagereward_from_snapshot(
                pipe=pipe,
                latents_start=snapshot_latents,
                start_step_idx=args.target_step_idx + 1,
                timesteps=timesteps,
                scheduler_snapshot=snapshot_scheduler,
                prompt_embeds_cfg=prompt_embeds_cfg,
                prompt_attention_mask_cfg=prompt_attention_mask_cfg,
                added_cond_kwargs=added_cond_kwargs,
                guidance_scale=float(args.guidance_scale),
                seed_base=int(args.seed) + st * 100 + args.target_step_idx * 13,
                mc=int(args.mc),
                prompts=batch_prompts,
                rm_model=rm_model,
            )

            all_hidden.append(hidden_buf["h"][:bsz].clone())
            all_reward.append(reward.clone())
            all_seed.extend([batch_seed + i for i in range(bsz)])

            if args.save_images:
                imgs = _decode_pixart_latents_to_pil(pipe, latents)
                for j, img in enumerate(imgs):
                    global_idx = st + j
                    ptxt = expanded_prompts[global_idx]
                    name = f"image_{global_idx:06d}_p{expanded_prompt_idx[global_idx]:04d}_{_safe_name(ptxt,32)}.png"
                    img.save(output_root / "images" / name)
                    all_image_names.append(name)
    finally:
        handle.remove()

    hiddens = torch.cat(all_hidden, dim=0).float()
    rewards = torch.cat(all_reward, dim=0).float()
    prompt_idx_tensor = torch.tensor(expanded_prompt_idx, dtype=torch.long)
    seed_tensor = torch.tensor(all_seed, dtype=torch.long)

    if hiddens.shape[0] != len(expanded_prompts) or rewards.shape[0] != len(expanded_prompts):
        raise RuntimeError(
            f"rows mismatch: hidden={hiddens.shape[0]} reward={rewards.shape[0]} expected={len(expanded_prompts)}"
        )

    out_pt = output_root / (
        f"pixart_prompt_hs_reward_n{len(expanded_prompts)}_s{args.target_step_idx}_l{args.target_layer_idx}_mc{args.mc}.pt"
    )
    torch.save(
        {
            "hiddens": hiddens,
            "reward": rewards,
            "prompt_indices": prompt_idx_tensor,
            "prompts": expanded_prompts,
            "seed": seed_tensor,
            "image_files": all_image_names if args.save_images else [],
            "meta": {
                "model_path": str(args.model_path),
                "num_prompts": len(prompts),
                "repeats_per_prompt": int(args.repeats_per_prompt),
                "num_samples": len(expanded_prompts),
                "num_inference_steps": int(args.num_inference_steps),
                "target_step_idx": int(args.target_step_idx),
                "target_layer_idx": int(args.target_layer_idx),
                "mc": int(args.mc),
                "guidance_scale": float(args.guidance_scale),
                "height": int(args.height),
                "width": int(args.width),
            },
        },
        out_pt,
    )
    (output_root / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] saved: {out_pt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout PixArt hiddens and ImageReward labels from prompt JSON.")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/PixArt-XL-2-1024-MS")
    p.add_argument("--prompts_json", type=str, required=True, help="JSON list[str] or list[{prompt:...}]")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--repeats_per_prompt", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--negative_prompt", type=str, default="blurry, low quality, watermark, text, deformed")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--target_step_idx", type=int, default=10)
    p.add_argument("--target_layer_idx", type=int, default=14)
    p.add_argument("--mc", type=int, default=4)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_images", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rollout_hiddens_rewards(args)


if __name__ == "__main__":
    main()
