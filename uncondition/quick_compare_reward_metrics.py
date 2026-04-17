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

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uncondition import random_search_clipscore_reward as rs_clip
from uncondition import random_search_imagereward_reward as rs_ir
from uncondition import random_search_ranker_reward as rs_ranker
from search.reward import load_imagereward
from utils import compute_clipscore, set_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Quick compare 4 methods by CLIPScore + ImageReward (no FID/IS)")
    p.add_argument("--ranker_ckpt", type=str, required=True)

    p.add_argument("--output_root", type=str, default="/root/autodl-tmp/uncond_quick_compare")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/google_ddpm-celebahq-256")
    p.add_argument("--clip_model_dir", type=str, default="/root/autodl-tmp/openai_clip-vit-large-patch14")

    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    p.add_argument("--clip_batch_size", type=int, default=64)

    # Ranker config
    p.add_argument("--parallel_size_ranker", type=int, default=2)
    p.add_argument("--layer_idx", type=int, default=19)
    p.add_argument("--step_idx", type=int, default=12)
    p.add_argument("--pool_kernel", type=int, default=2)
    p.add_argument("--pool_stride", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--input_channels", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Search-based reward methods (clip / imagereward)
    p.add_argument("--parallel_size_reward", type=int, default=2)

    p.add_argument("--skip_generation", action="store_true")
    return p.parse_args()


def _mean_imagereward_for_dir(gen_dir: Path, device: torch.device) -> float:
    rm = load_imagereward(device)
    prompt = "a photo of a human face"
    paths = sorted([p for p in gen_dir.glob("*.png") if p.is_file()])
    if not paths:
        raise FileNotFoundError(f"no generated png found under {gen_dir}")
    scores: list[float] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        scores.append(float(rm.score(prompt, img)))
    return float(sum(scores) / len(scores))


def _clear_pngs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink()


def _generate_no_reward_baseline(
    out_dir: Path,
    *,
    model_path: str,
    num_samples: int,
    batch_size: int,
    num_inference_steps: int,
    dtype: str,
    seed: int,
) -> None:
    device = set_device()
    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    if torch_dtype == torch.bfloat16:
        torch_dtype = torch.float16
    pipe = DDPMPipeline.from_pretrained(
        str(Path(model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=torch_dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    _clear_pngs(out_dir)

    n_rounds = math.ceil(num_samples / batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for ridx in range(n_rounds):
        bsz = min(batch_size, num_samples - img_counter)
        if bsz <= 0:
            break
        g_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        gen = torch.Generator(device=device).manual_seed(g_seed)
        out = pipe(
            batch_size=bsz,
            num_inference_steps=num_inference_steps,
            generator=gen,
            output_type="pil",
        )
        for im in out.images:
            if img_counter >= num_samples:
                break
            im.save(out_dir / f"{img_counter:05d}.png")
            img_counter += 1
        print(f"[baseline] round {ridx+1}/{n_rounds}, generated={img_counter}/{num_samples}")


@torch.inference_mode()
def _generate_ranker_random_search_fast(args: argparse.Namespace, out_dir: Path) -> None:
    device = set_device()
    _clear_pngs(out_dir)

    pipe = DDPMPipeline.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=rs_ranker._dtype_to_torch(args.dtype),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    base_scheduler = copy.deepcopy(pipe.scheduler)
    timesteps = base_scheduler.timesteps

    resnet_blocks = [(n, m) for n, m in pipe.unet.named_modules() if m.__class__.__name__ == "ResnetBlock2D"]
    layer_name, layer_module = resnet_blocks[args.layer_idx]
    print(f"[ranker] layer_idx={args.layer_idx} name={layer_name}, step_idx={args.step_idx}")

    ranker = rs_ranker._load_ranker(
        Path(args.ranker_ckpt).expanduser().resolve(),
        device,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        input_channels=args.input_channels,
        dropout=args.dropout,
    )

    n_rounds = math.ceil(args.num_samples / args.batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    for ridx in range(n_rounds):
        batch_n = min(args.batch_size, args.num_samples - img_counter)
        if batch_n <= 0:
            break

        cand_images_all: list[list[Image.Image]] = []
        cand_hidden_all: list[torch.Tensor] = []
        for _ in range(args.parallel_size_ranker):
            cand_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            cand_images, cand_hidden = rs_ranker._run_one_candidate(
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
            cand_images_all.append(cand_images)
            cand_hidden_all.append(cand_hidden)

        stacked_hidden = torch.stack(cand_hidden_all, dim=1).to(device, non_blocking=True)
        pred = ranker(stacked_hidden)
        best_idx = int(torch.argmax(pred.mean(dim=0)).item())
        best_images = cand_images_all[best_idx]

        for im in best_images:
            if img_counter >= args.num_samples:
                break
            im.save(out_dir / f"{img_counter:05d}.png")
            img_counter += 1

        del stacked_hidden, pred
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[ranker] round {ridx+1}/{n_rounds}, generated={img_counter}/{args.num_samples}")


@torch.inference_mode()
def _generate_clip_random_search_fast(args: argparse.Namespace, out_dir: Path) -> None:
    device = set_device()
    _clear_pngs(out_dir)

    pipe = DDPMPipeline.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=rs_clip._dtype_to_torch(args.dtype),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    base_scheduler = copy.deepcopy(pipe.scheduler)
    clip, clip_processor = rs_clip._load_clip(device, Path(args.clip_model_dir).expanduser().resolve())

    n_rounds = math.ceil(args.num_samples / args.batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    for ridx in range(n_rounds):
        batch_n = min(args.batch_size, args.num_samples - img_counter)
        if batch_n <= 0:
            break

        best_mean = float("-inf")
        best_images: list[Image.Image] | None = None
        for _ in range(args.parallel_size_reward):
            cand_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            pipe.scheduler = copy.deepcopy(base_scheduler)
            out = pipe(
                batch_size=batch_n,
                generator=torch.Generator(device=device).manual_seed(cand_seed),
                num_inference_steps=args.num_inference_steps,
                output_type="pil",
            )
            cand_images = [im.copy() for im in out.images]
            mean_score = rs_clip._batch_clipscore_face(cand_images, clip, clip_processor)
            if mean_score > best_mean:
                best_mean = mean_score
                best_images = cand_images
        assert best_images is not None
        for im in best_images:
            if img_counter >= args.num_samples:
                break
            im.save(out_dir / f"{img_counter:05d}.png")
            img_counter += 1
        print(f"[clip] round {ridx+1}/{n_rounds}, generated={img_counter}/{args.num_samples}")


@torch.inference_mode()
def _generate_imagereward_random_search_fast(args: argparse.Namespace, out_dir: Path) -> None:
    device = set_device()
    _clear_pngs(out_dir)

    pipe = DDPMPipeline.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=rs_ir._dtype_to_torch(args.dtype),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    base_scheduler = copy.deepcopy(pipe.scheduler)
    rm_model = load_imagereward(device)

    n_rounds = math.ceil(args.num_samples / args.batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    for ridx in range(n_rounds):
        batch_n = min(args.batch_size, args.num_samples - img_counter)
        if batch_n <= 0:
            break

        best_mean = float("-inf")
        best_images: list[Image.Image] | None = None
        for _ in range(args.parallel_size_reward):
            cand_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            pipe.scheduler = copy.deepcopy(base_scheduler)
            out = pipe(
                batch_size=batch_n,
                generator=torch.Generator(device=device).manual_seed(cand_seed),
                num_inference_steps=args.num_inference_steps,
                output_type="pil",
            )
            cand_images = [im.copy() for im in out.images]
            mean_score = rs_ir._batch_imagereward_face(cand_images, rm_model)
            if mean_score > best_mean:
                best_mean = mean_score
                best_images = cand_images
        assert best_images is not None
        for im in best_images:
            if img_counter >= args.num_samples:
                break
            im.save(out_dir / f"{img_counter:05d}.png")
            img_counter += 1
        print(f"[imagereward] round {ridx+1}/{n_rounds}, generated={img_counter}/{args.num_samples}")


def _load_images_from_dir_uint8(img_dir: Path) -> torch.Tensor:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not paths:
        raise FileNotFoundError(f"no generated image found under {img_dir}")
    tensors: list[torch.Tensor] = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    return torch.stack(tensors, dim=0).to(torch.uint8)


def _collect_clipscore_imagereward_only(
    out_dir: Path,
    *,
    clip_model_dir: str,
    clip_batch_size: int,
) -> dict[str, float]:
    device = set_device()
    gen_images = _load_images_from_dir_uint8(out_dir)
    prompts = ["a photo of a human face"] * int(gen_images.shape[0])
    clipscore = float(
        compute_clipscore(
            gen_images,
            prompts,
            clip_model_dir=clip_model_dir,
            device=device,
            clip_batch_size=clip_batch_size,
        )
    )
    imagereward = _mean_imagereward_for_dir(out_dir, device)
    return {
        "clipscore": clipscore,
        "imagereward": imagereward,
        "num_generated": float(gen_images.shape[0]),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = {
        "ranker_random_search": out_root / "ranker_random_search",
        "imagenet_random_search": out_root / "imagereward_random_search",  # alias per user wording
        "clip_random_search": out_root / "clip_random_search",
        "no_reward_baseline": out_root / "no_reward_baseline",
    }

    if not args.skip_generation:
        _generate_ranker_random_search_fast(args, case_dirs["ranker_random_search"])
        _generate_imagereward_random_search_fast(args, case_dirs["imagenet_random_search"])
        _generate_clip_random_search_fast(args, case_dirs["clip_random_search"])
        _generate_no_reward_baseline(
            case_dirs["no_reward_baseline"],
            model_path=args.model_path,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            dtype=args.dtype,
            seed=args.seed,
        )

    summary: dict[str, dict[str, float]] = {}
    for name, d in case_dirs.items():
        print(f"[eval] {name} from {d}")
        m = _collect_clipscore_imagereward_only(
            d,
            clip_model_dir=args.clip_model_dir,
            clip_batch_size=args.clip_batch_size,
        )
        summary[name] = {
            "clipscore": float(m["clipscore"]),
            "imagereward": float(m["imagereward"]),
            "num_generated": float(m["num_generated"]),
        }

    out_json = out_root / "summary_clipscore_imagereward.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
