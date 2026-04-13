#!/usr/bin/env python3
"""
按 ImageNet 类均衡生成训练 rollout：每个类固定 ``images_per_class`` 张（默认 128），共
``num_classes`` 类（默认 1000）。保存 ``image_{global_idx:05d}_{label:05d}.png``，并抓取表征：
**每一层 DiT block**（layer ratio 覆盖全部 block）、**每一个 hook 步索引 0..T-1**（step ratio
全覆盖）、pooling 为 **mean_dim1** 与 **last_token**（与 :class:`utils.DitRolloutReprCapture` 一致）。

默认：``deno_steps=10``，``dtype=bf16``，``guidance_scale=1.0``。与 ``torch.compile`` 不兼容。

示例::

    python3 /root/diffusion_ranker/ranker/rollout_trianing_data.py \
    --batch_size 256 --deno_steps 10 --guidance_scale 1.0 --dtype bf16
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from diffusers import DiTPipeline
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import (
    DitRolloutReprCapture,
    _prepare_pil_for_save,
    _ROLLOUT_SAVE_WORKERS,
    _save_pil_task,
    model_loader,
    refresh_memory,
    set_device,
)

BASE_SEED = 1314

NUM_IMAGENET_CLASSES = 1000
DEFAULT_IMAGES_PER_CLASS = 64


def _all_layer_ratios(n_blocks: int) -> tuple[float, ...]:
    """每个 transformer block 一个 ratio（0..1 均匀），与 ``DitRolloutReprCapture`` 去重逻辑一致。"""
    if n_blocks <= 1:
        return (1.0,)
    return tuple(i / (n_blocks - 1) for i in range(n_blocks))


def _all_step_ratio_ls(num_inference_steps: int) -> list[float]:
    """使 ``target_step_indices`` 覆盖 ``0 .. T-1`` 每一个去噪 hook 步。"""
    t = num_inference_steps
    if t <= 1:
        return [0.0]
    return [i / (t - 1) for i in range(t)]


def _output_dir(guidance_scale: float, deno_steps: int, images_per_class: int) -> Path:
    return Path(
        f"/root/autodl-tmp/dit_training_data_perclass{images_per_class}_cfg{guidance_scale}_deno{deno_steps}"
    )


@torch.inference_mode()
def rollout_training_data(
    pipe: DiTPipeline,
    training_data_dir: Path,
    num_classes: int,
    images_per_class: int,
    batch_size: int,
    num_steps: int,
    device: torch.device,
    guidance_scale: float,
    repr_capture: DitRolloutReprCapture,
    *,
    seed: int,
    jpeg_quality: int = 92,
    png_compress_level: int = 1,
    image_size: int = 256,
) -> int:
    """按类循环：每类 ``images_per_class`` 张，标签写入文件名；每 batch 记录表征。"""
    training_data_dir.mkdir(parents=True, exist_ok=True)
    img_counter = 0
    global_batch_idx = 0
    total_expected = num_classes * images_per_class

    for class_id in tqdm(range(num_classes), desc="per-class rollout"):
        remaining = images_per_class
        while remaining > 0:
            n_this = min(batch_size, remaining)
            batch_labels_ls = [class_id] * n_this
            gen = torch.Generator(device=device).manual_seed(seed + global_batch_idx)
            img_start = img_counter

            repr_capture.set_capturing(True)
            repr_capture.begin_batch(batch_labels_ls)

            out = pipe(
                class_labels=batch_labels_ls,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            )

            repr_capture.end_batch(
                global_batch_idx,
                img_start,
                generator_seed=seed + global_batch_idx,
            )

            batch_images = out.images
            to_save: list[tuple[Path, Image.Image]] = []
            for label, img in zip(batch_labels_ls, batch_images):
                im = _prepare_pil_for_save(img, image_size)
                save_path = training_data_dir / f"image_{img_counter:05d}_{label:05d}.png"
                to_save.append((save_path, im.copy()))
                img_counter += 1

            with ThreadPoolExecutor(max_workers=_ROLLOUT_SAVE_WORKERS) as ex:
                futures = [
                    ex.submit(
                        _save_pil_task,
                        path,
                        im,
                        fmt="png",
                        jpeg_quality=jpeg_quality,
                        png_compress_level=png_compress_level,
                    )
                    for path, im in to_save
                ]
                for fut in as_completed(futures):
                    fut.result()

            del out, gen, batch_images, to_save
            if device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            remaining -= n_this
            global_batch_idx += 1

    if img_counter != total_expected:
        print(
            f"warning: wrote {img_counter}, expected {total_expected}",
            file=sys.stderr,
        )
    return img_counter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="dit-xl-2-256")
    p.add_argument("--num_classes", type=int, default=NUM_IMAGENET_CLASSES)
    p.add_argument("--images_per_class", type=int, default=DEFAULT_IMAGES_PER_CLASS)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument(
        "--repr_pt",
        type=str,
        default="",
        help="表征 .pt 路径；空则默认为输出目录下 repr.pt",
    )
    p.add_argument(
        "--repr_store_dtype",
        default="bf16",
        choices=("fp16", "bf16"),
        help="表征落盘 dtype（默认 bf16 与推理 bf16 一致）",
    )
    args = p.parse_args()

    device = set_device()
    out_dir = _output_dir(args.guidance_scale, args.deno_steps, args.images_per_class)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = model_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)

    n_blocks = len(pipe.transformer.transformer_blocks)
    layer_ratios = _all_layer_ratios(n_blocks)
    step_ratio_ls = _all_step_ratio_ls(args.deno_steps)

    store_dtype = (
        torch.bfloat16 if args.repr_store_dtype == "bf16" else torch.float16
    )
    run_seed = BASE_SEED
    repr_capture = DitRolloutReprCapture(
        pipe,
        num_inference_steps=args.deno_steps,
        guidance_scale=args.guidance_scale,
        layer_ratios=layer_ratios,
        step_ratio_ls=step_ratio_ls,
        store_dtype=store_dtype,
        rollout_base_seed=run_seed,
    )
    repr_capture.install()

    repr_path = (
        Path(args.repr_pt).expanduser().resolve()
        if args.repr_pt.strip()
        else out_dir / "repr.pt"
    )

    n = rollout_training_data(
        pipe,
        out_dir,
        args.num_classes,
        args.images_per_class,
        args.batch_size,
        args.deno_steps,
        device,
        args.guidance_scale,
        repr_capture,
        seed=run_seed,
    )

    repr_capture.remove()
    refresh_memory()
    repr_capture.save_pt(repr_path)
    refresh_memory()
    print(f"representations saved -> {repr_path}")
    print(
        f"done -> {out_dir} ({n} images); "
        f"layers={len(layer_ratios)} steps={len(step_ratio_ls)} pooling=mean+last"
    )


if __name__ == "__main__":
    main()
