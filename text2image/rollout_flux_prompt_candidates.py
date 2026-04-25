#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
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


def _setup_flux_timesteps(pipe: FluxPipeline, latents: torch.Tensor, num_inference_steps: int, device: torch.device):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    return timesteps


def _decode_flux_latents_to_pil(pipe: FluxPipeline, latents_packed: torch.Tensor, height: int, width: int):
    latents = pipe._unpack_latents(latents_packed, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


@torch.inference_mode()
def rollout_prompt_candidates(args: argparse.Namespace) -> None:
    prompts = _load_prompts(Path(args.prompts_json).expanduser().resolve())
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _resolve_dtype(args.dtype, device)
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    if args.target_block_type == "dual":
        blocks = pipe.transformer.transformer_blocks
    else:
        blocks = pipe.transformer.single_transformer_blocks
    if not (0 <= args.target_layer_idx < len(blocks)):
        raise ValueError(
            f"target_layer_idx out of range for {args.target_block_type}: "
            f"[0, {len(blocks)-1}], got {args.target_layer_idx}"
        )
    target_block = blocks[args.target_layer_idx]
    if not (0 <= args.target_step_idx < args.num_inference_steps):
        raise ValueError(
            f"target_step_idx must be in [0, {args.num_inference_steps-1}], got {args.target_step_idx}"
        )

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
        hidden_buf: dict[str, torch.Tensor] = {}

        def _hook(_module, _inp, out):
            if current_step["value"] != args.target_step_idx:
                return
            h = out[0] if isinstance(out, (tuple, list)) else out
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

                batch_seed = args.seed + cand_idx * 100_000 + st
                generator = torch.Generator(device=device).manual_seed(batch_seed)

                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=batch_prompts,
                    prompt_2=None,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                )

                num_channels_latents = pipe.transformer.config.in_channels // 4
                latents, latent_image_ids = pipe.prepare_latents(
                    bsz,
                    num_channels_latents,
                    args.height,
                    args.width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents=None,
                )
                timesteps = _setup_flux_timesteps(pipe, latents, args.num_inference_steps, device=device)
                pipe.scheduler.set_begin_index(0)
                guidance = None
                if pipe.transformer.config.guidance_embeds:
                    guidance = torch.full([1], args.guidance_scale, device=device, dtype=torch.float32).expand(
                        latents.shape[0]
                    )

                hidden_buf.clear()
                for step_i, t in enumerate(timesteps):
                    current_step["value"] = step_i
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    with pipe.transformer.cache_context("cond"):
                        noise_pred = pipe.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]
                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if "h" not in hidden_buf:
                    raise RuntimeError(f"hook failed for candidate={cand_idx}, prompts[{st}:{ed}]")

                imgs = _decode_flux_latents_to_pil(pipe, latents, height=args.height, width=args.width)
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
                "target_block_type": str(args.target_block_type),
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
        gc_info = {"candidate_idx": cand_idx, "saved_repr": str(cand_dir / "repr.pt"), "num_images": len(image_files)}
        (cand_dir / "summary.json").write_text(json.dumps(gc_info, ensure_ascii=False, indent=2), encoding="utf-8")

    suite_meta = {
        "num_candidates": int(args.num_candidates),
        "num_prompts": int(len(prompts)),
        "target_step_idx": int(args.target_step_idx),
        "target_block_type": str(args.target_block_type),
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
    p = argparse.ArgumentParser(description="Rollout FLUX candidates from prompt JSON and save hooked hiddens.")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/FLUX.1-dev")
    p.add_argument("--prompts_json", type=str, required=True, help="JSON list[str] or list[{prompt:...}]")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--target_step_idx", type=int, default=14)
    p.add_argument("--target_block_type", type=str, default="dual", choices=("dual", "single"))
    p.add_argument("--target_layer_idx", type=int, default=9)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rollout_prompt_candidates(args)


if __name__ == "__main__":
    main()
