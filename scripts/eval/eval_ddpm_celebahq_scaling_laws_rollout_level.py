#!/usr/bin/env python3
"""
DDPM (google/ddpm-celebahq-256) rollout-level 评测：与 eval_dit_scaling_laws_rollout_level 结构类似，
每个去噪步数单独跑完整 scheduler 调度，生成 num_samples 张图后对目录算 FID / IS（可选 KID、P/R）。

生成数量与参考集子采样默认均为 30k。不传 --real_images_dir 时，自动从 Hugging Face 下载 CelebA-HQ 类数据（默认 nielsr/CelebA-HQ-resized-256，
若不可用会自动尝试其它 Hub id）并导出到 <output_dir>/real_images_hf_export；首次需联网且体积较大。
有本地真实图目录时可传 --real_images_dir；也可用 prepare_real_images_dir 从 save_to_disk 导出。

示例（无本地数据集）：

cd /data/home/yxfeng/diffusion_ranker && CUDA_VISIBLE_DEVICES=4 python3 scripts/eval/eval_ddpm_celebahq_scaling_laws_rollout_level.py \
  --model_id google/ddpm-celebahq-256 \
  --output_dir /data/home/yxfeng/diffusion_ranker/outputs/eval_ddpm_celebahq_rollout \
  --batch_size 156 \
  --fp16 \
  --compile \
  --num_samples 50000 \
  --fid_real_max_samples 30000 \
  --fid_real_subset_seed 0 
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm

from utils import compute_metrics, prepare_real_images_from_hf, setup_seed



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


@torch.inference_mode()
def rollout_ddpm_with_steps(
    pipe: DDPMPipeline,
    save_dir: Path,
    num_steps: int,
    num_samples: int,
    batch_size: int,
    device: str,
    seed: int,
) -> int:
    """
    使用指定 num_inference_steps 独立跑一遍采样；内部等价于每次 pipe.scheduler.set_timesteps(num_steps)。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    img_counter = 0
    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc=f"steps={num_steps}", leave=False):
        batch_n = min(batch_size, num_samples - batch_idx * batch_size)
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
        out = pipe(
            batch_size=batch_n,
            generator=gen,
            num_inference_steps=num_steps,
            output_type="pil",
        )
        for img in out.images:
            if img_counter >= num_samples:
                break
            if img.size != (256, 256):
                img = img.convert("RGB").resize((256, 256), Image.LANCZOS)
            else:
                img = img.convert("RGB")
            img.save(save_dir / f"{img_counter:05d}.png")
            img_counter += 1
    return img_counter



def save_metrics_curves(results: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not results:
        return
    has_kid = "KID" in results[0]
    steps = [r["steps"] for r in results]
    n_plots = 2 + int(has_kid)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    idx = 0
    axes[idx].plot(steps, [r["FID"] for r in results], "o-", color="C0", linewidth=2, markersize=4)
    axes[idx].set_ylabel("FID")
    axes[idx].set_title("FID vs denoising steps")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    axes[idx].plot(steps, [r["IS"] for r in results], "o-", color="C2", linewidth=2, markersize=4)
    axes[idx].set_ylabel("IS")
    axes[idx].set_title("IS vs denoising steps")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    if has_kid:
        axes[idx].plot(steps, [r["KID"] for r in results], "o-", color="C1", linewidth=2, markersize=4)
        axes[idx].set_ylabel("KID")
        axes[idx].set_title("KID vs denoising steps")
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("num_inference_steps")
    plt.tight_layout()
    fig_path = output_dir / "metrics_curves.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Metrics curves saved to: {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/ddpm-celebahq-256",
        help="与 demo.py 一致：HF 模型 id 或本地已缓存的同名目录",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="仅使用本地缓存加载模型（不发起网络请求）",
    )
    parser.add_argument("--output_dir", type=str, default="/data/yxfeng/eval_ddpm_celebahq_rollout")
    parser.add_argument(
        "--real_images_dir",
        type=str,
        default=None,
        help="真实图像根目录（递归 png/jpg/jpeg）。不传则从 Hugging Face 下载 CelebA-HQ 并导出到 output 子目录",
    )
    parser.add_argument(
        "--hf_real_dataset",
        type=str,
        default="nielsr/CelebA-HQ-resized-256",
        help="无 real_images_dir 时优先使用的 HF 数据集 id；不存在时会自动尝试其它 CelebA-HQ 镜像",
    )
    parser.add_argument(
        "--hf_real_split",
        type=str,
        default="train",
        help="HF 数据集 split 名",
    )
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="UNet 使用 float16")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(pipe.unet)，仅 CUDA 上建议尝试",
    )
    parser.add_argument("--kid", action="store_true")
    parser.add_argument("--prc", action="store_true")
    parser.add_argument(
        "--deno_steps",
        type=int,
        nargs="+",
        default=None,
        help="评测的去噪步数列表；默认 10..990 步长 10 及 1000",
    )
    parser.add_argument(
        "--fid_real_max_samples",
        type=int,
        default=30000,
        help="从真实目录子采样作参考集的张数；需 ≤ 目录内可发现图片总数（默认 30k）",
    )
    parser.add_argument("--fid_real_subset_seed", type=int, default=0)
    args = parser.parse_args()

    if args.deno_steps is not None:
        deno_steps = sorted({s for s in args.deno_steps if 1 <= s <= 10_000})
    else:
        deno_steps = list(range(10, 100, 10)) + [100]
        deno_steps = sorted({s for s in deno_steps if 1 <= s <= 10_000})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    need_real = max(args.fid_real_max_samples, 1)
    if args.real_images_dir:
        real_dir = Path(args.real_images_dir)
        assert real_dir.is_dir(), f"真实图像目录不存在: {real_dir}"
    else:
        export_root = output_dir / "real_images_hf_export"
        real_dir = prepare_real_images_from_hf(
            export_root,
            hf_dataset_id=args.hf_real_dataset,
            split=args.hf_real_split,
            num_samples=need_real,
            shuffle_seed=args.fid_real_subset_seed,
        )

    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading DDPMPipeline: {args.model_id}")
    pipe = load_ddpm(args.model_id, device, fp16=args.fp16, local_files_only=args.local_files_only)
    if args.compile and device == "cuda":
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
        print("torch.compile enabled (unet)")

    print(f"denoising steps: {deno_steps}")
    results: list[dict] = []
    results_jsonl = output_dir / "metrics_results.jsonl"
    results_jsonl.unlink(missing_ok=True)

    real_cache = "ddpm_celebahq256_real"

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
        print(f"  steps={steps:4d}: metrics appended to {results_jsonl}")

        log_parts = [
            f"FID={metrics.get('FID', float('nan')):.2f}",
            f"IS={metrics.get('IS', float('nan')):.4f}",
        ]
        if args.kid:
            log_parts.append(f"KID={metrics.get('KID', float('nan')):.4f}")
        if args.prc:
            log_parts.append(
                f"P={metrics.get('Precision', float('nan')):.3f} R={metrics.get('Recall', float('nan')):.3f}"
            )
        print(f"  steps={steps:4d}:  " + "  ".join(log_parts))

        shutil.rmtree(gen_dir, ignore_errors=True)
        print(f"  steps={steps:4d}: deleted image directory {gen_dir.name}")

    summary_path = output_dir / "metrics_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Full results: {summary_path}")

    save_metrics_curves(results, output_dir)


if __name__ == "__main__":
    main()
