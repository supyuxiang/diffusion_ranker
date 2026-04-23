#!/usr/bin/env python3
"""
DiT rollout-level 评测：不同总 inference steps 的 pipeline 各自独立跑一遍
（每个 step 数对应不同的 timestep 调度，非同一轨迹截断）

使用 torch_fidelity 计算 FID、IS，可选 KID、Precision/Recall。

CUDA_VISIBLE_DEVICES=5 python3 /data/home/yxfeng/project1/idm/scripts/eval/eval_dit_scaling_laws_rollout_level.py \
    --dit_path /data/yxfeng/models/dit_model \
    --output_dir /data/yxfeng/eval_dit_scaling_rollout \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --num_samples 50000 \
    --fid_real_max_samples None \
    --fid_real_subset_seed 0 \
    --batch_size 24 \
    --compile \
    --fp16 \
    --kid \
    --prc \
    --deno_steps 10



# 单独计算 step=10 的 baseline
cd /data/home/yxfeng/project1/idm && CUDA_VISIBLE_DEVICES=4 python3 scripts/eval/eval_dit_scaling_laws_rollout_level.py \
    --dit_path /data/yxfeng/models/dit_model \
    --output_dir /data/yxfeng/eval_dit_step10_baseline_50k_1.25cfg \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --num_samples 50000 \
    --deno_steps 10 \
    --batch_size 24 \
    --compile \
    --fid_real_max_samples None \
    --fid_real_subset_seed 0 \
    --kid \
    --prc \
    --guidance_scale 1.25
"""

import json
import os
import sys
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm import tqdm

from utils import compute_metrics, setup_seed, load_dit


@torch.inference_mode()
def rollout_dit_with_steps(
    pipe: DiTPipeline,
    save_dir: Path,
    num_steps: int,
    num_samples: int,
    batch_size: int,
    device: str,
    seed: int,
    guidance_scale: float = 1.0,
):
    """
    Run pipeline with specified steps independently, save to save_dir.
    Each time set_timesteps(num_steps), i.e. the complete timestep scheduling corresponding to the step number.
    """
    num_classes = 1000
    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size
    img_counter = 0

    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc=f"steps={num_steps}", leave=False):
        batch_num_samples = min(batch_size, num_samples - batch_idx * batch_size)
        labels = torch.randint(0, num_classes, (batch_num_samples,), device=device).tolist()
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)

        latents = randn_tensor(
            (batch_num_samples, latent_channels, latent_size, latent_size),
            generator=gen,
            device=device,
            dtype=pipe.transformer.dtype,
        )
        if guidance_scale > 1:
            latents = torch.cat([latents] * 2, dim=0)
        class_labels = torch.tensor(labels, device=device).reshape(-1)
        class_null = torch.tensor([1000] * batch_num_samples, device=device)
        class_labels_input = torch.cat([class_labels, class_null], dim=0) if guidance_scale > 1 else class_labels

        # critical: use independent timestep scheduling for each step number
        pipe.scheduler.set_timesteps(num_steps) # set_timesteps(num_steps) sets the timesteps for the scheduler, i.e. the complete timestep scheduling corresponding to the step number.

        # 已经有了torch.inference_mode(),不需要手动添加no_grad()
        for step_idx, t in enumerate(pipe.scheduler.timesteps):
            if guidance_scale > 1:
                half = latents[: len(latents) // 2]
                latents = torch.cat([half, half], dim=0)
            latent_input = pipe.scheduler.scale_model_input(latents, t)
            if not torch.is_tensor(t):
                timesteps = torch.tensor([t], dtype=torch.long, device=latent_input.device).expand(latent_input.shape[0])
            else:
                timesteps = (t.expand(latent_input.shape[0]) if t.dim() == 0 else t).to(latent_input.device)

            noise_pred = pipe.transformer(latent_input, timestep=timesteps, class_labels=class_labels_input).sample
            if guidance_scale > 1:
                # Match diffusers DiTPipeline: rest keeps cond+uncond batch; eps must be doubled after CFG.
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                noise_pred = torch.cat([eps, rest], dim=1)
            if pipe.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            latents = pipe.scheduler.step(model_output, t, latents).prev_sample

        # after denoising, decode and save
        to_decode = latents[:batch_num_samples] if guidance_scale > 1 else latents
        to_decode = to_decode / pipe.vae.config.scaling_factor
        samples = pipe.vae.decode(to_decode).sample
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples_np = samples.cpu().permute(0, 2, 3, 1).float().detach().numpy()
        images = pipe.numpy_to_pil(samples_np)

        for i, img in enumerate(images):
            if img_counter >= num_samples:
                break
            img = img.resize((256, 256), Image.LANCZOS)
            img.save(save_dir / f"{img_counter:05d}.png")
            img_counter += 1

    return img_counter


def _parse_fid_real_max_samples(s: str) -> int | None:
    t = str(s).strip().lower()
    if t in ("none", "null", ""):
        return None
    return int(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_path", default="/data/home/yxfeng/data/models/dit_model")
    parser.add_argument("--output_dir", default="/data/yxfeng/eval_dit_scaling_rollout")
    parser.add_argument("--imagenet_val", default="/data/yxfeng/datasets/imagenet/val",
                        help="ImageNet validation set path (for FID reference)")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=1.0,help="Guidance scale for DDPM.")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile to accelerate transformer inference.")
    parser.add_argument("--kid", action="store_true", help="Extra compute KID (Kernel Inception Distance).")
    parser.add_argument("--prc", action="store_true", help="Extra compute Precision/Recall (need VGG features, slower).")
    parser.add_argument("--deno_steps", type=int, nargs="+", default=None,
                        help="Specify the denoising steps to evaluate, e.g. --deno_steps 10 or --deno_steps 10 20 50")
    parser.add_argument("--fid_real_max_samples", type=_parse_fid_real_max_samples, default=50000,
                        help="Max real images for FID/KID/PRC reference; pass none/null to use all images in val. Default 50000.")
    parser.add_argument("--fid_real_subset_seed", type=int, default=0,
                        help="Reference subset RNG seed (only used when --fid_real_max_samples limits the set).")
    args = parser.parse_args()


    if args.deno_steps is not None:
        deno_steps = [s for s in args.deno_steps if 1 <= s <= 1000]
        deno_steps = sorted(set(deno_steps))
    else:
        deno_steps = list(range(10, 1000, 10))
        deno_steps = [s for s in deno_steps if 1 <= s <= 1000]
        deno_steps = sorted(set(deno_steps))


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = Path(output_dir) / 'metrics_results.jsonl'

    imagenet_val = Path(args.imagenet_val)

    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = load_dit(args.dit_path, device, fp16=args.fp16)
    if args.compile and device == "cuda":
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        print("torch.compile enabled (transformer)")

    print(f"denoising steps: {deno_steps}")

    results = []
    for steps in tqdm(deno_steps, desc="rollout"):
        gen_dir = Path(output_dir) / f'gen_steps_{steps}'
        gen_dir.mkdir(parents=True, exist_ok=True)

        n = rollout_dit_with_steps(
            pipe,
            gen_dir,
            steps,
            args.num_samples,
            args.batch_size,
            device,
            args.seed,
            guidance_scale=args.guidance_scale,
        )
        print(f"  steps={steps:4d}: generated {n} images to {gen_dir}")

        metrics = compute_metrics(
            gen_dir, imagenet_val, device,
            real_cache_name="imagenet_val",
            fid=True, isc=True, kid=args.kid, prc=args.prc,
            real_max_samples=args.fid_real_max_samples,
            real_subset_seed=args.fid_real_subset_seed,
        )
        row = {"steps": steps, **metrics}

        results.append(row)
        with open(results_jsonl, 'a') as f:
            f.write(json.dumps(row) + '\n')
            print(f'steps={steps:4d} metrics have been saved to {results_jsonl}')
        
        shutil.rmtree(gen_dir, ignore_errors=True)
        print(f"  steps={steps:4d}:  deleted image directory {gen_dir.name}")

    save_metrics_curves(results, output_dir)


def save_metrics_curves(results: list[dict], output_dir: Path):
    """Plot FID, IS and optional KID, Precision/Recall curves vs denoising steps.
    
    Args:
        results: list[metrics_step_idx1, metrics_step_idx2, ...]
        output_dir: path to the output directory
    """
    import matplotlib.pyplot as plt

    has_kid = 1 if 'KID' in results[0].keys() else 0
    steps = [r['steps'] for r in results]
    n_plots = 2 + int(has_kid)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), sharex=True)

    idx = 0
    # FID
    fid_vals = [r['FID'] for r in results]
    axes[idx].plot(steps, fid_vals, "o-", color="C0", linewidth=2, markersize=4)
    axes[idx].set_ylabel("FID")
    axes[idx].set_title("FID vs Steps")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # IS
    is_vals = [r['IS'] for r in results]
    axes[idx].plot(steps, is_vals, "o-", color="C2", linewidth=2, markersize=4)
    axes[idx].set_ylabel("IS")
    axes[idx].set_title("IS vs Steps")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    if has_kid:
        kid_vals = [r['KID'] for r in results]
        axes[idx].plot(steps, kid_vals, "o-", color="C1", linewidth=2, markersize=4)
        axes[idx].set_ylabel("KID")
        axes[idx].set_title("KID vs Steps")
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    plt.tight_layout()
    fig_path = output_dir / "metrics_curves.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nMetrics curves saved to: {fig_path}")


if __name__ == "__main__":
    main()
