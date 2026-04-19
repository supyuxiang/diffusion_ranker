#!/usr/bin/env python3
'''
python /root/diffusion_ranker/uncondition_celebahq/random_search_dino_reward.py \
  --output_dir /root/autodl-tmp/uncond_random_search_dino_ps8 \
  --parallel_size 8 \
  --num_samples 100 \
  --batch_size 16 \
  --num_inference_steps 25

'''
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
from transformers import AutoImageProcessor, AutoModel

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


def _load_dino(device: torch.device, dino_model_dir: Path) -> tuple[AutoModel, AutoImageProcessor]:
    model = AutoModel.from_pretrained(str(dino_model_dir), local_files_only=True).to(device)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(str(dino_model_dir), local_files_only=True)
    return model, processor


@torch.inference_mode()
def _dino_features(
    images: list[Image.Image],
    dino: AutoModel,
    processor: AutoImageProcessor,
) -> torch.Tensor:
    inputs = processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
    inputs = {k: v.to(dino.device) for k, v in inputs.items()}
    out = dino(**inputs)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output.float()
    else:
        feat = out.last_hidden_state[:, 0, :].float()
    return F.normalize(feat, dim=-1)


def _iter_image_paths(img_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


@torch.inference_mode()
def _build_dino_prototype(
    real_dir: Path,
    dino: AutoModel,
    processor: AutoImageProcessor,
    *,
    dino_batch_size: int,
    max_ref_images: int,
) -> torch.Tensor:
    paths = _iter_image_paths(real_dir)
    if max_ref_images > 0:
        paths = paths[:max_ref_images]
    if not paths:
        raise FileNotFoundError(f"no images found under {real_dir}")

    feats_sum = None
    n = 0
    for i in tqdm(range(0, len(paths), dino_batch_size), desc="build dino prototype"):
        chunk = paths[i : i + dino_batch_size]
        pil_ls: list[Image.Image] = []
        for p in chunk:
            try:
                pil_ls.append(Image.open(p).convert("RGB"))
            except Exception:
                continue
        if not pil_ls:
            continue
        feat = _dino_features(pil_ls, dino, processor)
        cur = feat.sum(dim=0)
        feats_sum = cur if feats_sum is None else (feats_sum + cur)
        n += feat.shape[0]
    if feats_sum is None or n == 0:
        raise RuntimeError(f"failed to build dino prototype from {real_dir}")
    proto = F.normalize(feats_sum / n, dim=-1)
    return proto


@torch.inference_mode()
def _batch_dino_reward_face(
    images: list[Image.Image],
    dino: AutoModel,
    processor: AutoImageProcessor,
    prototype: torch.Tensor,
) -> float:
    feat = _dino_features(images, dino, processor)
    sim = (feat * prototype.unsqueeze(0)).sum(dim=-1)
    return float(sim.mean().item())


def _mean_imagereward_for_dir(gen_dir: Path, device: torch.device) -> float:
    rm = load_imagereward(device)
    prompt = "A photo of a human face."
    vals: list[float] = []
    for p in tqdm(sorted(gen_dir.glob("*.png")), desc="ImageReward(final)"):
        img = Image.open(p).convert("RGB")
        vals.append(float(rm.score(prompt, img)))
    return float(sum(vals) / max(1, len(vals)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Online random search with DINO reward (parallel size=8)")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/uncond_random_search_dino_ps8")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/google-ddpm-ema-celebahq-256")
    p.add_argument("--real_dir", type=str, default="/root/autodl-tmp/celebahq256_png_30k_flat")
    p.add_argument("--clip_model_dir", type=str, default="/root/autodl-tmp/openai_clip-vit-large-patch14")
    p.add_argument("--clip_batch_size", type=int, default=64)
    p.add_argument("--dino_model_dir", type=str, default="/root/autodl-tmp/facebook_dinov2-base")
    p.add_argument("--dino_batch_size", type=int, default=64)
    p.add_argument("--max_ref_images", type=int, default=4096)
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--parallel_size", type=int, default=8)
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

    dino, dino_processor = _load_dino(device, Path(args.dino_model_dir).expanduser().resolve())
    prototype = _build_dino_prototype(
        Path(args.real_dir).expanduser().resolve(),
        dino,
        dino_processor,
        dino_batch_size=args.dino_batch_size,
        max_ref_images=args.max_ref_images,
    )

    n_rounds = math.ceil(args.num_samples / args.batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    round_logs: list[dict[str, float]] = []
    for ridx in tqdm(range(n_rounds), desc=f"online best-of-{args.parallel_size} (dino)"):
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
            mean_score = _batch_dino_reward_face(cand_images, dino, dino_processor, prototype)
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
    metrics["selection_reward"] = "dino_prototype_cosine"

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "round_log.json").write_text(json.dumps(round_logs, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
