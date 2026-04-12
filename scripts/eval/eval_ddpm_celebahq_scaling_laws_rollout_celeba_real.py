#!/usr/bin/env python3
"""
DDPM (``google/ddpm-celebahq-256``) rollout-level scaling law 评测，结构与
``eval_dit_scaling_laws_rollout_level.py`` 对齐：每个 ``num_inference_steps`` 单独跑完整调度，
生成图落盘后调用 ``utils.compute_metrics`` 算 FID / IS（可选 KID、P/R），再删生成目录。

**真实图**：默认使用 autodl 公共盘上的 CelebA 对齐人脸。若
``/root/autodl-tmp/celeba/img_align_celeba`` 尚不存在或为空，则自动从
``/root/autodl-pub/CelebA/Img/img_align_celeba.zip`` 解压到 ``/root/autodl-tmp/celeba``。

说明：模型在 CelebA-HQ 上训练，参考集为 CelebA ``img_align_celeba`` 时 FID 数值与
「HQ 对 HQ」论文设定不同，仅便于在本地固定参考集上做 scaling 曲线。

示例：


cd /root/diffusion_ranker
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_ddpm_celebahq_scaling_laws_rollout_celeba_real.py \
  --model_id /root/autodl-tmp/google_ddpm-celebahq-256 \
  --local_files_only \
  --output_dir /root/autodl-tmp/eval_ddpm_celebahq_rollout_celeba \
  --fp16 \
  --compile \
  --num_samples 50000 --batch_size 156 \
  --deno_steps 10 20 30 40 50 60 70 80 90 100
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gc
import torch
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm

from utils import compute_metrics, set_seed


CELEBA_PUB_ROOT_DEFAULT = Path("/root/autodl-pub/CelebA")
CELEBA_ZIP_REL = Path("Img/img_align_celeba.zip")
CELEBA_EXTRACT_ROOT_DEFAULT = Path("/root/autodl-tmp/celeba")


def ensure_celeba_img_align_dir(
    pub_root: Path,
    extract_root: Path,
    *,
    min_files: int = 10,
) -> Path:
    """
    返回包含 ``*.jpg`` 的 ``img_align_celeba`` 目录；必要时从 ``pub_root/Img/img_align_celeba.zip`` 解压。
    """
    pub_root = pub_root.expanduser().resolve()
    extract_root = extract_root.expanduser().resolve()
    aligned = extract_root / "img_align_celeba"

    def _count_jpgs(d: Path) -> int:
        if not d.is_dir():
            return 0
        return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))

    if _count_jpgs(aligned) >= min_files:
        print(f"Using existing CelebA align dir ({_count_jpgs(aligned)} images): {aligned}")
        return aligned

    zpath = pub_root / CELEBA_ZIP_REL
    if not zpath.is_file():
        raise FileNotFoundError(
            f"CelebA zip not found: {zpath}. "
            f"Set --celeba_pub_root or place img_align_celeba.zip under <pub>/Img/."
        )

    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zpath} -> {extract_root} (one-time, large)...")
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(extract_root)

    n = _count_jpgs(aligned)
    if n < min_files:
        raise RuntimeError(
            f"After unzip, expected many images under {aligned}, found {n}. "
            "Check zip integrity or --celeba_extract_root."
        )
    print(f"CelebA align ready: {n} images under {aligned}")
    return aligned


def _parse_fid_real_max_samples(s: str) -> int | None:
    t = str(s).strip().lower()
    if t in ("none", "null", ""):
        return None
    return int(t)


def load_ddpm(
    model_id: str,
    device: str,
    *,
    fp16: bool,
    local_files_only: bool = False,
) -> DDPMPipeline:
    kwargs: dict = {"torch_dtype": torch.float16 if fp16 else torch.float32}
    if local_files_only:
        kwargs["local_files_only"] = True
    pipe = DDPMPipeline.from_pretrained(model_id, **kwargs)
    return pipe.to(device)


def _prepare_pil_for_save(img: Image.Image, target_short: int) -> Image.Image:
    if img.size != (target_short, target_short):
        return img.resize((target_short, target_short), Image.Resampling.LANCZOS)
    return img


def _save_pil_task(
    path: Path,
    img: Image.Image,
    *,
    fmt: str,
    jpeg_quality: int,
    png_compress_level: int,
) -> None:
    if fmt == "jpeg":
        path = path.with_suffix(".jpg")
        img.save(path, quality=jpeg_quality, subsampling=0)
    else:
        img.save(path, compress_level=png_compress_level)


@torch.inference_mode()
def rollout_ddpm_with_steps(
    pipe: DDPMPipeline,
    save_dir: Path,
    num_steps: int,
    num_samples: int,
    batch_size: int,
    device: str,
    seed: int,
    *,
    jpeg_quality: int = 92,
    png_compress_level: int = 1,
    save_workers: int = 1,
    save_format: str = "png",
    image_size: int = 256,
) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    img_counter = 0
    num_batches = (num_samples + batch_size - 1) // batch_size
    fmt = "jpeg" if save_format.lower() in ("jpeg", "jpg") else "png"
    ext = ".jpg" if fmt == "jpeg" else ".png"

    for batch_idx in tqdm(range(num_batches), desc=f"steps={num_steps}", leave=False):
        batch_n = min(batch_size, num_samples - batch_idx * batch_size)
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
        out = pipe(
            batch_size=batch_n,
            generator=gen,
            num_inference_steps=num_steps,
            output_type="pil",
        )

        to_save: list[tuple[Path, Image.Image]] = []
        for img in out.images:
            if img_counter >= num_samples:
                break
            img = _prepare_pil_for_save(img.convert("RGB"), image_size)
            to_save.append((save_dir / f"{img_counter:05d}{ext}", img.copy()))
            img_counter += 1

        if not to_save:
            continue

        if save_workers <= 1:
            for path, im in to_save:
                _save_pil_task(
                    path,
                    im,
                    fmt=fmt,
                    jpeg_quality=jpeg_quality,
                    png_compress_level=png_compress_level,
                )
        else:

            def _work(item: tuple[Path, Image.Image]) -> None:
                p, im = item
                _save_pil_task(
                    p,
                    im,
                    fmt=fmt,
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


def save_metrics_curves(results: list[dict], output_dir: Path) -> None:
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
    fig.suptitle(
        "DDPM (CelebA-HQ 256) rollout vs steps — real: CelebA img_align_celeba",
        fontsize=12,
        fontweight="semibold",
        y=1.01,
    )

    idx = 0
    axes[idx].plot(
        steps,
        [r["FID"] for r in results],
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.6,
    )
    axes[idx].set_ylabel("FID ↓", fontsize=11)
    axes[idx].set_title("Fréchet Inception Distance", fontsize=11)
    axes[idx].grid(True, alpha=0.35)
    idx += 1

    axes[idx].plot(
        steps,
        [r["IS"] for r in results],
        "o-",
        color="#2ca02c",
        linewidth=2.2,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.6,
    )
    axes[idx].set_ylabel("IS ↑", fontsize=11)
    axes[idx].set_title("Inception Score (mean)", fontsize=11)
    axes[idx].grid(True, alpha=0.35)
    idx += 1

    if has_kid:
        axes[idx].plot(
            steps,
            [r["KID"] for r in results],
            "o-",
            color="#ff7f0e",
            linewidth=2.2,
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        axes[idx].set_ylabel("KID ↓", fontsize=11)
        axes[idx].set_title("Kernel Inception Distance (mean)", fontsize=11)
        axes[idx].grid(True, alpha=0.35)
        idx += 1

    if has_prc:
        pr = [float(r["Precision"]) for r in results]
        rec = [float(r["Recall"]) for r in results]
        axes[idx].plot(
            steps,
            pr,
            "s-",
            color="#9467bd",
            linewidth=2.0,
            markersize=4,
            label="Precision",
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        axes[idx].plot(
            steps,
            rec,
            "^-",
            color="#d62728",
            linewidth=2.0,
            markersize=4,
            label="Recall",
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        axes[idx].set_ylabel("P / R", fontsize=11)
        axes[idx].set_title("Precision & Recall (torch_fidelity)", fontsize=11)
        axes[idx].legend(loc="best", framealpha=0.92)
        axes[idx].grid(True, alpha=0.35)

    axes[-1].set_xlabel("Denoising steps", fontsize=11)
    fig.align_ylabels(axes)
    plt.tight_layout()

    for name in ("metrics_curves.png", "metrics_curves.pdf"):
        fig_path = output_dir / name
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Metrics curves saved to: {fig_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/ddpm-celebahq-256",
        help="HF id 或本机已下载目录（如 /root/autodl-tmp/google_ddpm-celebahq-256）",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/eval_ddpm_celebahq_rollout_celeba",
    )
    parser.add_argument(
        "--celeba_pub_root",
        type=str,
        default=str(CELEBA_PUB_ROOT_DEFAULT),
        help="含 Img/img_align_celeba.zip 的 CelebA 根目录",
    )
    parser.add_argument(
        "--celeba_extract_root",
        type=str,
        default=str(CELEBA_EXTRACT_ROOT_DEFAULT),
        help="解压目标根目录（得到 .../img_align_celeba/*.jpg）",
    )
    parser.add_argument(
        "--real_images_dir",
        type=str,
        default=None,
        help="若已自备真实图目录（递归搜图），则跳过 zip 解压逻辑",
    )
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--kid", action="store_true")
    parser.add_argument("--prc", action="store_true")
    parser.add_argument(
        "--deno_steps",
        type=int,
        nargs="+",
        default=None,
        help="去噪步数列表；默认 10..100 步长 10 及 200 500 1000",
    )
    parser.add_argument(
        "--fid_real_max_samples",
        type=_parse_fid_real_max_samples,
        default=30000,
        help="参考集最多张数；none/null 表示用目录内全部（递归）",
    )
    parser.add_argument("--fid_real_subset_seed", type=int, default=0)
    parser.add_argument("--jpeg_quality", type=int, default=92)
    parser.add_argument("--png_compress_level", type=int, default=1)
    parser.add_argument(
        "--save_workers",
        type=int,
        default=4,
        help="每批并行写盘线程数（1 为顺序）",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="png",
        choices=("png", "jpeg"),
    )
    args = parser.parse_args()

    if args.real_images_dir:
        real_dir = Path(args.real_images_dir).expanduser().resolve()
        if not real_dir.is_dir():
            raise FileNotFoundError(f"--real_images_dir is not a directory: {real_dir}")
    else:
        real_dir = ensure_celeba_img_align_dir(
            Path(args.celeba_pub_root),
            Path(args.celeba_extract_root),
        )

    if args.deno_steps is not None:
        deno_steps = sorted({s for s in args.deno_steps if 1 <= s <= 10_000})
    else:
        deno_steps = sorted({*list(range(10, 110, 10)), 200, 500, 1000})

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "metrics_results.jsonl"
    results_jsonl.unlink(missing_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading DDPMPipeline: {args.model_id}")
    pipe = load_ddpm(args.model_id, device, fp16=args.fp16, local_files_only=args.local_files_only)
    pipe.set_progress_bar_config(disable=True)
    if args.compile and device == "cuda":
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
        print("torch.compile enabled (unet)")

    print(f"Real reference dir: {real_dir}")
    print(f"Denoising steps: {deno_steps}")

    results: list[dict] = []
    real_cache = "celeba_img_align_real"

    for steps in tqdm(deno_steps, desc="rollout"):
        gen_dir = output_dir / f"gen_steps_{steps}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        n = rollout_ddpm_with_steps(
            pipe,
            gen_dir,
            steps,
            args.num_samples,
            args.batch_size,
            device,
            args.seed,
            jpeg_quality=args.jpeg_quality,
            png_compress_level=args.png_compress_level,
            save_workers=max(1, args.save_workers),
            save_format=args.save_format,
        )
        print(f"  steps={steps:4d}: generated {n} images -> {gen_dir}")

        metrics = compute_metrics(
            gen_dir,
            real_dir,
            device,
            real_cache_name=real_cache,
            fid=True,
            isc=True,
            kid=args.kid,
            prc=args.prc,
            real_max_samples=args.fid_real_max_samples,
            real_subset_seed=args.fid_real_subset_seed,
        )
        row = {"steps": steps, **metrics}
        results.append(row)
        with open(results_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  steps={steps:4d}: metrics -> {results_jsonl}")

        shutil.rmtree(gen_dir, ignore_errors=True)
        print(f"  steps={steps:4d}: deleted {gen_dir.name}")
        gc.collect()
        torch.cuda.empty_cache()

    summary_path = output_dir / "metrics_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Full results: {summary_path}")

    if results:
        save_metrics_curves(results, output_dir)
    else:
        print("No results; skip curves")


if __name__ == "__main__":
    main()
