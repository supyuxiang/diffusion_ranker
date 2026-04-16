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
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_ROOT_ = Path(__file__).resolve().parent.parent
if str(_ROOT_) not in sys.path:
    sys.path.insert(0, str(_ROOT_))

from search.reward import load_imagereward
from utils import compute_fid_is_clipscore_for_uncondition, set_device, set_seed


def _dtype_to_torch(dtype: str) -> torch.dtype:
    d = dtype.lower().strip()
    if d == "fp16":
        return torch.float16
    if d == "fp32":
        return torch.float32
    if d == "bf16":
        # DDPMPipeline's PIL conversion path is not bf16-safe.
        return torch.float16
    raise ValueError(f"unknown dtype: {dtype}")


def _load_clip(device: torch.device, clip_model_dir: Path) -> tuple[CLIPModel, CLIPProcessor]:
    clip = CLIPModel.from_pretrained(str(clip_model_dir), local_files_only=True).to(device)
    clip.eval()
    processor = CLIPProcessor.from_pretrained(str(clip_model_dir), local_files_only=True)
    return clip, processor


@torch.inference_mode()
def _batch_clipscore_face(
    images: list[Image.Image],
    clip: CLIPModel,
    processor: CLIPProcessor,
) -> float:
    prompt = "a photo of a human face"
    inputs = processor(
        text=[prompt] * len(images),
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(clip.device) for k, v in inputs.items()}
    out = clip(**inputs)
    image_embeds = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    sim = (image_embeds * text_embeds).sum(dim=-1)
    return float(sim.mean().item())


def _mean_imagereward_for_dir(gen_dir: Path, device: torch.device) -> float:
    rm = load_imagereward(device)
    prompt = "a photo of a human face"
    vals: list[float] = []
    for p in tqdm(sorted(gen_dir.glob("*.png")), desc="ImageReward(final)"):
        img = Image.open(p).convert("RGB")
        vals.append(float(rm.score(prompt, img)))
    return float(sum(vals) / max(1, len(vals)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Online random search with CLIPScore reward (parallel size=10)")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/uncond_random_search_clipscore_ps2")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/google_ddpm-celebahq-256")
    p.add_argument("--real_dir", type=str, default="/root/autodl-tmp/img_align_celeba")
    p.add_argument("--clip_model_dir", type=str, default="/root/autodl-tmp/openai_clip-vit-large-patch14")
    p.add_argument("--clip_batch_size", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--parallel_size", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()

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

    clip, clip_processor = _load_clip(device, Path(args.clip_model_dir).expanduser().resolve())

    n_rounds = math.ceil(args.num_samples / args.batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    round_logs: list[dict[str, float]] = []
    for ridx in tqdm(range(n_rounds), desc="online best-of-2 (clipscore)"):
        batch_n = min(args.batch_size, args.num_samples - img_counter)
        if batch_n <= 0:
            break

        best_mean = float("-inf")
        best_images: list[Image.Image] | None = None

        for _k in range(args.parallel_size):
            cand_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            pipe.scheduler = copy.deepcopy(base_scheduler)
            out = pipe(
                batch_size=batch_n,
                generator=torch.Generator(device=device).manual_seed(cand_seed),
                num_inference_steps=args.num_inference_steps,
                output_type="pil",
            )
            cand_images = [im.copy() for im in out.images]
            mean_score = _batch_clipscore_face(cand_images, clip, clip_processor)
            if mean_score > best_mean:
                best_mean = mean_score
                best_images = cand_images
            else:
                del cand_images
            del out
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
    metrics["selection_reward"] = "clipscore"

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "round_log.json").write_text(json.dumps(round_logs, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
