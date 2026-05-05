#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import set_seed


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        dt = torch.float64 if isinstance(t, float) else torch.int64
        t = torch.tensor([t], dtype=dt, device=latent_model_input.device)
    elif len(t.shape) == 0:
        t = t[None].to(latent_model_input.device)
    return t.expand(latent_model_input.shape[0])


def _safe_name(s: str, max_len: int = 80) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_\-]", "", s)
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


def _decode_pixart_latents_to_pil(pipe: PixArtAlphaPipeline, latents: torch.Tensor):
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


@torch.inference_mode()
def rollout_prompt_candidates(args: argparse.Namespace) -> None:
    prompts = _load_prompts(Path(args.prompts_json).expanduser().resolve())
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _resolve_dtype(args.dtype, device)
    do_cfg = args.guidance_scale > 1.0
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    n_blocks = len(pipe.transformer.transformer_blocks)
    if not (0 <= args.target_layer_idx < n_blocks):
        raise ValueError(f"target_layer_idx must be in [0, {n_blocks - 1}]")
    if not (0 <= args.target_step_idx < args.num_inference_steps):
        raise ValueError(f"target_step_idx must be in [0, {args.num_inference_steps - 1}]")
    target_block = pipe.transformer.transformer_blocks[args.target_layer_idx]

    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size

    (output_root / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")

    for cand_idx in range(args.num_candidates):
        cand_dir = output_root / f"candidate_{cand_idx:04d}"
        img_dir = cand_dir / "images"
        cand_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

        hiddens_ls: list[torch.Tensor] = []
        prompt_indices: list[int] = []
        image_files: list[str] = []
        current_step = {"value": -1}
        current_batch = {"value": 0}
        hidden_buf: dict[str, torch.Tensor] = {}

        def _hook(_module, _inp, out):
            if current_step["value"] != args.target_step_idx:
                return
            h = out[0] if isinstance(out, (tuple, list)) else out
            if do_cfg:
                h = h[: current_batch["value"]]
            hidden_buf["h"] = h.mean(dim=1).detach().float().cpu()

        handle = target_block.register_forward_hook(_hook)
        try:
            pbar = tqdm(
                range(0, len(prompts), args.batch_size),
                desc=f"rollout cand{cand_idx:04d}",
                leave=False,
            )
            for st in pbar:
                ed = min(st + args.batch_size, len(prompts))
                batch_prompts = prompts[st:ed]
                bsz = len(batch_prompts)
                if bsz <= 0:
                    continue
                current_batch["value"] = bsz

                batch_seed = args.seed + cand_idx * 100_000 + st
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

                for step_i, t in enumerate(timesteps):
                    current_step["value"] = step_i
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

                if "h" not in hidden_buf:
                    raise RuntimeError(f"hook failed for candidate={cand_idx}, prompts[{st}:{ed}]")

                imgs = _decode_pixart_latents_to_pil(pipe, latents)
                for j, img in enumerate(imgs):
                    pidx = st + j
                    img_name = f"image_{pidx:05d}_{_safe_name(prompts[pidx], max_len=32)}.png"
                    img.save(img_dir / img_name)
                    image_files.append(img_name)
                    prompt_indices.append(pidx)
                hiddens_ls.append(hidden_buf["h"][:bsz].clone())
        finally:
            handle.remove()

        hiddens = torch.cat(hiddens_ls, dim=0).float()
        if hiddens.shape[0] != len(prompts):
            raise RuntimeError(f"candidate {cand_idx}: hidden rows {hiddens.shape[0]} != {len(prompts)}")

        payload = {
            "hiddens": hiddens,
            "prompt_indices": torch.tensor(prompt_indices, dtype=torch.long),
            "image_files": image_files,
            "meta": {
                "candidate_idx": int(cand_idx),
                "target_step_idx": int(args.target_step_idx),
                "target_layer_idx": int(args.target_layer_idx),
                "num_prompts": int(len(prompts)),
                "num_inference_steps": int(args.num_inference_steps),
                "guidance_scale": float(args.guidance_scale),
                "height": int(args.height),
                "width": int(args.width),
                "model_path": str(args.model_path),
            },
        }
        torch.save(payload, cand_dir / "repr.pt")
        (cand_dir / "summary.json").write_text(
            json.dumps(
                {
                    "candidate_idx": cand_idx,
                    "saved_repr": str(cand_dir / "repr.pt"),
                    "num_images": len(image_files),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    suite_meta = {
        "num_candidates": int(args.num_candidates),
        "num_prompts": int(len(prompts)),
        "target_step_idx": int(args.target_step_idx),
        "target_layer_idx": int(args.target_layer_idx),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "height": int(args.height),
        "width": int(args.width),
        "model_path": str(args.model_path),
    }
    (output_root / "suite_meta.json").write_text(json.dumps(suite_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] rollout suite saved to: {output_root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout PixArt candidates from prompt JSON and save hooked hiddens.")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/PixArt-XL-2-1024-MS")
    p.add_argument("--prompts_json", type=str, required=True, help="JSON list[str] or list[{prompt:...}]")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--negative_prompt", type=str, default="blurry, low quality, watermark, text, deformed")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--target_step_idx", type=int, default=10)
    p.add_argument("--target_layer_idx", type=int, default=14)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rollout_prompt_candidates(args)


if __name__ == "__main__":
    main()
