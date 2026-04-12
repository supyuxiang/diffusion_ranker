#!/usr/bin/env python3
"""
DiT rollout-level 评测：不同总 inference steps 的 pipeline 各自独立跑一遍
（每个 step 数对应不同的 timestep 调度，非同一轨迹截断）

**FID / IS / KID** 由 ``utils.compute_metrics`` 使用 **torchmetrics**（Inception v3 via torch-fidelity）计算；
**Precision / Recall**（``--prc``）仍用 **torch_fidelity**（torchmetrics 无官方 P/R）。

真实图像目录由 ``--imagenet_val`` 指定（本机布局为 ``.../imagenet/val`` 下按 WordNet id 分子目录）。

流程：每个 step 数 **先保存生成图到目录**，再调用 ``compute_metrics`` 从磁盘读图算 FID/IS/KID（及可选 P/R）。
可用 ``--save_format jpeg``、``--save_workers``、``--png_compress_level`` 加速落盘。

**默认 ``--dtype bf16``**（见 ``utils.load_dit``）；也可用 ``--dtype fp16`` 或 ``--dtype fp32``。

CUDA_VISIBLE_DEVICES=0 python3 /root/diffusion_ranker/scripts/eval/eval_dit_scaling_laws_rollout_level.py \
    --model_name dit-xl-2-256 \
    --output_dir /hy-tmp/diffusion_ranker/eval_dit_scaling_rollout \
    --imagenet_val /hy-tmp/diffusion_ranker/datasets/imagenet/val \
    --num_samples 50000 \
    --fid_real_max_samples None \
    --fid_real_subset_seed 0 \
    --compile \
    --kid \
    --batch_size 512 \
    --seed 42 \
    --save_workers 4 \
    --jpeg_quality 92 \
    --png_compress_level 1 \
    --guidance_scale 1.0 \
    --deno_steps 10 20 30 40 50 60 70 80 90 100

"""

import json
import os
import sys
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from diffusers import DiTPipeline
from PIL import Image
from tqdm import tqdm
import gc

from utils import compute_metrics, set_seed, load_dit


__all__ = [
    'rollout_dit_with_steps',

]


def _prepare_pil_for_save(img: Image.Image, target_short: int) -> Image.Image:
    if img.size != (target_short, target_short):
        return img.resize((target_short, target_short), Image.Resampling.LANCZOS)
    return img


def _save_pil_task(path: Path, img: Image.Image, *, fmt: str, jpeg_quality: int, png_compress_level: int) -> None:
    if fmt == "jpeg":
        img.save(path, quality=jpeg_quality, subsampling=0)
    else:
        img.save(path, compress_level=png_compress_level)


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
    *,
    jpeg_quality: int = 92,
    png_compress_level: int = 1,
    save_workers: int = 1,
    image_size: int = 256,
):
    """
    For each batch: call ``DiTPipeline.__call__`` with ``num_inference_steps=num_steps`` so the
    scheduler uses an independent full trajectory (same semantics as the former manual loop).

    This is not inherently faster than a hand-written loop (both step over timesteps in Python);
    speed comes from ``torch.compile``, fp16, and batch size. Using the official API keeps parity
    with diffusers and avoids duplicating CFG / sigma-split logic.
    """
    num_classes = 1000
    img_counter = 0
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"steps={num_steps}", leave=False):
        batch_num_samples = min(batch_size, num_samples - batch_idx * batch_size)
        labels = torch.randint(0, num_classes, (batch_num_samples,), device=device).tolist()
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)

        out = pipe(
            class_labels=labels,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        )
        images = out.images

        to_save: list[tuple[Path, Image.Image]] = []
        for img in images:
            if img_counter >= num_samples:
                break
            img = _prepare_pil_for_save(img, image_size)
            to_save.append((save_dir / f"{img_counter:05d}.png", img.copy()))
            img_counter += 1

        if not to_save:
            continue

        if save_workers <= 1:
            for path, im in to_save:
                _save_pil_task(
                    path,
                    im,
                    fmt="png",
                    jpeg_quality=jpeg_quality,
                    png_compress_level=png_compress_level,
                )
        else:

            def _work(item: tuple[Path, Image.Image]) -> None:
                p, im = item
                _save_pil_task(
                    p,
                    im,
                    fmt="png",
                    jpeg_quality=jpeg_quality,
                    png_compress_level=png_compress_level,
                )

            with ThreadPoolExecutor(max_workers=save_workers) as ex:
                futures = [ex.submit(_work, it) for it in to_save]
                for fut in as_completed(futures):
                    fut.result()
    gc.collect()
    torch.cuda.empty_cache()

    return img_counter



def rollout_dit_with_all_steps(
    pipe:DiTPipeline,
    save_dir:Path | str,
    deno_steps_ls:list[int],
    save_format:str='png',
    jpeg_quality:int=92,
    png_compress_level:int=1,
    save_workers:int=4,
    image_size:int=256,
    batch_size:int=256,
    dtype:str='bf16',
    seed:int=42,
    cfg:int=1.0,
    num_samples:int=50000,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = save_dir / "metrics_results.jsonl"

    imagenet_val=Path('/hy-tmp/diffusion_ranker/datasets/imagenet/val')

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pbar = tqdm(deno_steps_ls, desc="rollout")
    for steps in pbar:
        rollout_dit_with_steps(
            pipe,
            save_dir,
            steps,
            num_samples,
            batch_size,
            device,
            seed,
            guidance_scale=cfg,
            save_format=save_format,
            jpeg_quality=jpeg_quality,
            png_compress_level=png_compress_level,
            save_workers=save_workers,
            image_size=image_size,
        )
        
    return


def main_my():
    dit = load_dit('dit-xl-2-256', 'cuda', dtype='bf16')
    pass








def _parse_fid_real_max_samples(s: str) -> int | None:
    t = str(s).strip().lower()
    if t in ("none", "null", ""):
        return None
    return int(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="dit-xl-2-256",
        help="Model name (see utils.model_name2path).",
    )
    parser.add_argument(
        "--output_dir",
        default="/hy-tmp/diffusion_ranker/eval_dit_scaling_rollout",
        help="Metrics jsonl + optional curves.",
    )
    parser.add_argument(
        "--imagenet_val",
        default="/hy-tmp/diffusion_ranker/datasets/imagenet/val",
        help="Directory of reference images (ILSVRC val: class subdirs under .../val).",
    )
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=1.0,help="Guidance scale for DDPM.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=("fp32", "fp16", "bf16"),
        help="Pipeline weight dtype (see utils.load_dit). Default bf16.",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile to accelerate transformer inference.")
    parser.add_argument("--kid", action="store_true", help="Extra compute KID (Kernel Inception Distance).")
    parser.add_argument("--prc", action="store_true", help="Extra compute Precision/Recall (need VGG features, slower).")
    parser.add_argument("--deno_steps", type=int, nargs="+", default=None,
                        help="Specify the denoising steps to evaluate, e.g. --deno_steps 10 or --deno_steps 10 20 50")
    parser.add_argument("--fid_real_max_samples", type=_parse_fid_real_max_samples, default=50000,
                        help="Max real images for FID/KID/PRC reference; pass none/null to use all images in val. Default 50000.")
    parser.add_argument("--fid_real_subset_seed", type=int, default=0,
                        help="Reference subset RNG seed (only used when --fid_real_max_samples limits the set).")
    parser.add_argument("--jpeg_quality", type=int, default=92, help="JPEG quality (1–95) when --save_format jpeg.")
    parser.add_argument(
        "--png_compress_level",
        type=int,
        default=1,
        help="PNG zlib level 0–9; lower is faster writes (default 1).",
    )
    parser.add_argument(
        "--save_workers",
        type=int,
        default=4,
        help="Parallel writer threads per batch when saving to disk (1 = sequential).",
    )
    args = parser.parse_args()

    imagenet_val = Path(args.imagenet_val).expanduser().resolve()
    if not imagenet_val.is_dir():
        raise FileNotFoundError(f"--imagenet_val is not a directory: {imagenet_val}")
    print(f"Using ImageNet real dir: {imagenet_val}")

    if args.deno_steps is not None:
        deno_steps = [s for s in args.deno_steps if 1 <= s <= 1000]
        deno_steps = sorted(set(deno_steps))
    else:
        deno_steps = list(range(10, 1000, 10))
        deno_steps = [s for s in deno_steps if 1 <= s <= 1000]
        deno_steps = sorted(set(deno_steps))


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = Path(output_dir) / "metrics_results.jsonl"

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = load_dit(args.model_name, device, dtype=args.dtype)
    # One tqdm per batch is enough; disable diffusers' per-step bar inside pipe.__call__.
    pipe.set_progress_bar_config(disable=True)
    if args.compile and device == "cuda":
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        print("torch.compile enabled (transformer)")

    print(f"denoising steps: {deno_steps}")

    results = []
    for steps in tqdm(deno_steps, desc="rollout"):
        gen_dir = Path(output_dir) / f"gen_steps_{steps}"
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
            save_format=args.save_format,
            jpeg_quality=args.jpeg_quality,
            png_compress_level=args.png_compress_level,
            save_workers=max(1, args.save_workers),
        )
        print(f"  steps={steps:4d}: generated {n} images to {gen_dir}")

        metrics = compute_metrics(
            gen_dir,
            imagenet_val,
            device,
            real_cache_name="imagenet_val",
            fid=True,
            isc=True,
            kid=args.kid,
            prc=args.prc,
            real_max_samples=args.fid_real_max_samples,
            real_subset_seed=args.fid_real_subset_seed,
        )

        shutil.rmtree(gen_dir, ignore_errors=True)
        print(f"  steps={steps:4d}:  deleted image directory {gen_dir.name}")
        gc.collect()
        torch.cuda.empty_cache()
        row = {"steps": steps, **metrics}

        results.append(row)
        with open(results_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
            print(f"steps={steps:4d} metrics have been saved to {results_jsonl}")

    if results:
        save_metrics_curves(results, output_dir)
    else:
        print("No rollout steps completed; skip metrics_curves.png")


def save_metrics_curves(results: list[dict], output_dir: Path) -> None:
    """Plot FID, IS and optional KID; optional Precision & Recall vs denoising steps.

    Writes ``metrics_curves.png`` (and ``metrics_curves.pdf``) under ``output_dir``.
    """
    import matplotlib.pyplot as plt

    if not results:
        return

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    has_kid = "KID" in results[0]
    has_prc = "Precision" in results[0] and "Recall" in results[0]
    steps = [r["steps"] for r in results]

    n_rows = 2 + int(has_kid) + int(has_prc)
    fig_h = max(9.0, 3.4 * n_rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(9.5, fig_h), sharex=True)
    if n_rows == 1:
        axes = [axes]
    fig.suptitle("DiT rollout-level metrics vs denoising steps", fontsize=13, fontweight="semibold", y=1.01)

    idx = 0
    fid_vals = [r["FID"] for r in results]
    axes[idx].plot(steps, fid_vals, "o-", color="#1f77b4", linewidth=2.2, markersize=5, markeredgecolor="white", markeredgewidth=0.6)
    axes[idx].set_ylabel("FID ↓", fontsize=11)
    axes[idx].set_title("Fréchet Inception Distance", fontsize=11)
    axes[idx].grid(True, alpha=0.35)
    idx += 1

    is_vals = [r["IS"] for r in results]
    axes[idx].plot(steps, is_vals, "o-", color="#2ca02c", linewidth=2.2, markersize=5, markeredgecolor="white", markeredgewidth=0.6)
    axes[idx].set_ylabel("IS ↑", fontsize=11)
    axes[idx].set_title("Inception Score (mean)", fontsize=11)
    axes[idx].grid(True, alpha=0.35)
    idx += 1

    if has_kid:
        kid_vals = [r["KID"] for r in results]
        axes[idx].plot(steps, kid_vals, "o-", color="#ff7f0e", linewidth=2.2, markersize=5, markeredgecolor="white", markeredgewidth=0.6)
        axes[idx].set_ylabel("KID ↓", fontsize=11)
        axes[idx].set_title("Kernel Inception Distance (mean)", fontsize=11)
        axes[idx].grid(True, alpha=0.35)
        idx += 1

    if has_prc:
        pr = [float(r["Precision"]) for r in results]
        rec = [float(r["Recall"]) for r in results]
        axes[idx].plot(steps, pr, "s-", color="#9467bd", linewidth=2.0, markersize=4, label="Precision", markeredgecolor="white", markeredgewidth=0.5)
        axes[idx].plot(steps, rec, "^-", color="#d62728", linewidth=2.0, markersize=4, label="Recall", markeredgecolor="white", markeredgewidth=0.5)
        axes[idx].set_ylabel("P / R", fontsize=11)
        axes[idx].set_title("Precision & Recall (torch_fidelity)", fontsize=11)
        axes[idx].legend(loc="best", framealpha=0.92)
        axes[idx].grid(True, alpha=0.35)
        idx += 1

    axes[-1].set_xlabel("Denoising steps", fontsize=11)
    fig.align_ylabels(axes)
    plt.tight_layout()

    for name in ("metrics_curves.png", "metrics_curves.pdf"):
        fig_path = output_dir / name
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Metrics curves saved to: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
