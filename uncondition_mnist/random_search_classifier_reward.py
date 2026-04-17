#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMPipeline
from PIL import Image
from torchvision.datasets import MNIST
from tqdm import tqdm

_ROOT_ = Path(__file__).resolve().parent.parent
if str(_ROOT_) not in sys.path:
    sys.path.insert(0, str(_ROOT_))

from uncondition_mnist.mnist_classifier_reward import (
    load_mnist_classifier,
    reward_neg_entropy_mnist,
)
from utils import compute_fid, compute_is, set_device, set_seed


def _dtype_to_torch(dtype: str) -> torch.dtype:
    d = dtype.lower().strip()
    if d == "fp16":
        return torch.float16
    if d == "fp32":
        return torch.float32
    if d == "bf16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype: {dtype}")


def _load_mnist_test_images(data_root: str | Path) -> torch.Tensor:
    ds = MNIST(root=str(Path(data_root).expanduser().resolve()), train=False, download=True)
    arr = torch.from_numpy(ds.data.numpy()).unsqueeze(1).contiguous().repeat(1, 3, 1, 1)
    return arr.to(torch.uint8)


def _load_images_from_dir_uint8(img_dir: str | Path) -> torch.Tensor:
    d = Path(img_dir).expanduser().resolve()
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)
    imgs: list[torch.Tensor] = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        imgs.append(t)
    if not imgs:
        raise FileNotFoundError(f"no images found under {d}")
    return torch.stack(imgs, dim=0).to(torch.uint8)


@torch.inference_mode()
def _generate_candidate_images(
    pipe: DDPMPipeline,
    *,
    batch_n: int,
    num_inference_steps: int,
    seed: int,
    device: torch.device,
) -> list[Image.Image]:
    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        batch_size=batch_n,
        num_inference_steps=num_inference_steps,
        generator=gen,
        output_type="pil",
    )
    return out.images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MNIST random search with final-step classifier reward")
    p.add_argument(
        "--classifier_ckpt",
        type=str,
        default="/root/autodl-tmp/mnist_classifier/resnet18/classifier_best.pt",
    )
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/mnist_random_search_classifier_ps2")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/1aurent_ddpm-mnist")
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp")
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--parallel_size", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.parallel_size != 2:
        raise ValueError(f"this script is configured for parallel_size=2, got {args.parallel_size}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}")

    set_seed(args.seed)
    device = set_device()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink()

    real_images = _load_mnist_test_images(args.data_root)
    print(f"[info] loaded MNIST test images: {tuple(real_images.shape)}")

    classifier = load_mnist_classifier(args.classifier_ckpt, device)

    dtype = args.dtype
    if dtype == "bf16":
        print("[warn] bf16 is not supported in DDPMPipeline numpy conversion, fallback to fp16.")
        dtype = "fp16"
    pipe = DDPMPipeline.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=_dtype_to_torch(dtype),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    n_rounds = math.ceil(args.num_samples / args.batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    round_logs: list[dict[str, float]] = []

    for ridx in tqdm(range(n_rounds), desc="online final-step classifier best-of-2"):
        batch_n = min(args.batch_size, args.num_samples - img_counter)
        if batch_n <= 0:
            break

        seed0 = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        seed1 = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())

        cand0 = _generate_candidate_images(
            pipe,
            batch_n=batch_n,
            num_inference_steps=args.num_inference_steps,
            seed=seed0,
            device=device,
        )
        cand1 = _generate_candidate_images(
            pipe,
            batch_n=batch_n,
            num_inference_steps=args.num_inference_steps,
            seed=seed1,
            device=device,
        )

        r0 = reward_neg_entropy_mnist(classifier, cand0, device)
        r1 = reward_neg_entropy_mnist(classifier, cand1, device)
        choose_first = r0 >= r1
        chosen_rewards = torch.where(choose_first, r0, r1)

        for i in range(batch_n):
            best_im = cand0[i] if bool(choose_first[i].item()) else cand1[i]
            best_im.save(out_dir / f"{img_counter:05d}.png")
            img_counter += 1

        round_logs.append(
            {
                "round": float(ridx),
                "seed0": float(seed0),
                "seed1": float(seed1),
                "cand0_mean_reward": float(r0.mean().item()),
                "cand1_mean_reward": float(r1.mean().item()),
                "chosen_mean_reward": float(chosen_rewards.mean().item()),
                "choose_first_ratio": float(choose_first.float().mean().item()),
            }
        )

        del cand0, cand1, r0, r1, choose_first, chosen_rewards
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gen_images = _load_images_from_dir_uint8(out_dir)
    metrics = {"fid": float(compute_fid(real_images, gen_images, device=device))}
    is_mean, is_std = compute_is(gen_images, device=device)
    metrics["is"] = float(is_mean)
    metrics["is_std"] = float(is_std)
    metrics["num_generated"] = float(img_counter)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "round_log.json").write_text(json.dumps(round_logs, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
