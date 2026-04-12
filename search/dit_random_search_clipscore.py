#!/usr/bin/env python3
"""
DiT Random Search with CLIP Score as Verifier (Inference-Time Scaling)

与 dit_random_search_imagereward.py 结构相同，但每轮从 K 个候选 batch 中选
「batch 平均 CLIPScore（图文余弦相似度）最高」的一批写入已选集；
每张图使用对应 ImageNet 类名填充的文本模板。

默认使用本机 Hugging Face 缓存中的 OpenAI CLIP ViT-L/14（transformers CLIPModel +
CLIPProcessor）；也可改用 open_clip（--clip_backend open_clip）。
图文相似度为 L2 归一化后的 image_embeds · text_embeds（余弦相似度）。

结束后对 output_dir 与 ImageNet val 计算 FID / IS。

用法:
  CUDA_VISIBLE_DEVICES=6 python3 dit_random_search_clipscore.py \
    --dit_path /data/yxfeng/models/dit_model \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --output_dir /data/yxfeng/eval_dit_random_search_clip \
    --num_samples 50000 \
    --batch_size 32 \
    --num_candidates 8 \
    --init_random_samples 32 \
    --deno_steps 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DiTPipeline
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dit_random_search_common import (
    CANDIDATE_SEED_HIGH,
    init_random_samples,
    load_dit,
    load_imagenet_simple_labels,
    prompt_for_class,
    rollout_dit_batch,
)
from utils import compute_fid_and_is, setup_seed

_DEFAULT_HF_CLIP_ROOT = Path("/data/yxfeng/models/clip/models--openai--clip-vit-large-patch14")


def resolve_hf_clip_snapshot_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    ref = path / "refs" / "main"
    if ref.is_file():
        snap_id = ref.read_text(encoding="utf-8").strip()
        cand = path / "snapshots" / snap_id
        if (cand / "config.json").is_file():
            return cand
    raise FileNotFoundError(
        f"无法在 {path} 解析 CLIP 权重目录（需要 config.json 或 refs/main + snapshots/<hash>）"
    )


def load_hf_clip(model_dir: Path, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True)
    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = model.to(device).float().eval()
    return model, processor


def load_open_clip(model_name: str, pretrained: str, device: str):
    import open_clip
    from torchvision import transforms

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).float().eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return model, tokenizer, transform


def mean_batch_clipscore_hf(
    model,
    processor,
    device: str,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
) -> float:
    prompts = [prompt_for_class(labels_1k, cid, prompt_template) for cid in class_ids]
    rgb = [img.convert("RGB") for img in images]
    inputs = processor(text=prompts, images=rgb, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_features = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (image_features * text_features).sum(dim=-1)
    return float(sim.mean().item()) if sim.numel() else float("-inf")


def mean_batch_clipscore_open_clip(
    model,
    tokenizer,
    image_transform,
    device: str,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
) -> float:
    prompts = [prompt_for_class(labels_1k, cid, prompt_template) for cid in class_ids]
    text_tokens = tokenizer(prompts).to(device)
    tensors = torch.stack(
        [image_transform(img.convert("RGB")) for img in images],
        dim=0,
    ).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sim = (image_features * text_features).sum(dim=-1)
    return float(sim.mean().item()) if sim.numel() else float("-inf")


def run_random_search_clipscore(
    pipe: DiTPipeline,
    clip_backend: str,
    clip_model,
    clip_aux,
    labels_1k: list[str],
    prompt_template: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    deno_steps: int,
    device: str,
    seed: int,
    guidance_scale: float = 1.0,
    n_init: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for _ in tqdm(range(num_rounds), desc="Random Search CLIPScore"):
        n_this_batch = min(batch_size, num_samples - img_counter)
        if n_this_batch <= 0:
            break

        candidate_seeds = rng.integers(0, CANDIDATE_SEED_HIGH, size=num_candidates).tolist()
        best_score = float("-inf")
        best_batch: list[Image.Image] | None = None

        for cand_seed in candidate_seeds:
            batch_images, batch_classes = rollout_dit_batch(
                pipe,
                deno_steps,
                n_this_batch,
                device,
                seed=int(cand_seed),
                guidance_scale=guidance_scale,
            )
            if clip_backend == "hf":
                score = mean_batch_clipscore_hf(
                    clip_model,
                    clip_aux,
                    device,
                    batch_images,
                    batch_classes,
                    labels_1k,
                    prompt_template,
                )
            else:
                tokenizer, image_transform = clip_aux
                score = mean_batch_clipscore_open_clip(
                    clip_model,
                    tokenizer,
                    image_transform,
                    device,
                    batch_images,
                    batch_classes,
                    labels_1k,
                    prompt_template,
                )
            if score > best_score:
                best_score = score
                best_batch = batch_images

        assert best_batch is not None
        for img in best_batch:
            if img_counter >= num_samples:
                break
            img.save(output_dir / f"{img_counter:06d}.png")
            img_counter += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dit_path", default="/data/home/yxfeng/data/models/dit_model")
    p.add_argument("--imagenet_val", default="/data/yxfeng/datasets/imagenet/val")
    p.add_argument("--output_dir", default="/data/yxfeng/eval_dit_random_search_clipscore")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--num_candidates",
        type=int,
        default=8,
        help="每轮随机采样的候选 batch 数（不同 seed）",
    )
    p.add_argument("--deno_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DiT 半精度（默认开）；可用 --no-fp16",
    )
    p.add_argument("--ref_cache_name", default="imagenet_val")
    p.add_argument("--init_random_samples", type=int, default=32)
    p.add_argument(
        "--prompt_template",
        type=str,
        default="a high-quality picture of a {name}",
        help="CLIP 文本；{name}、{id} 可用",
    )
    p.add_argument(
        "--clip_backend",
        type=str,
        choices=("hf", "open_clip"),
        default="hf",
        help="hf: 本地 transformers CLIP；open_clip: 在线/缓存 open_clip 权重",
    )
    p.add_argument(
        "--clip_model_path",
        type=str,
        default=str(_DEFAULT_HF_CLIP_ROOT),
        help="Hugging Face CLIP 缓存根目录或 snapshots/... 目录（--clip_backend hf）",
    )
    p.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="--clip_backend open_clip 时的模型名",
    )
    p.add_argument(
        "--clip_pretrained",
        type=str,
        default="openai",
        help="--clip_backend open_clip 时的预训练标签",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert args.deno_steps >= 1
    assert 0 <= args.init_random_samples <= args.num_samples

    output_dir = Path(args.output_dir)
    imagenet_val = Path(args.imagenet_val)
    assert imagenet_val.exists(), f"ImageNet val 未找到: {imagenet_val}"

    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_1k = load_imagenet_simple_labels()

    clip_backend = args.clip_backend
    clip_dir_resolved: Path | None = None
    if clip_backend == "hf":
        clip_dir_resolved = resolve_hf_clip_snapshot_dir(Path(args.clip_model_path))
        print(f"1. 加载 HuggingFace CLIP（本地）: {clip_dir_resolved}")
        clip_model, clip_aux = load_hf_clip(clip_dir_resolved, device)
    else:
        print(f"1. 加载 OpenCLIP {args.clip_model}/{args.clip_pretrained} ...")
        clip_model, clip_tokenizer, clip_transform = load_open_clip(
            args.clip_model, args.clip_pretrained, device
        )
        clip_aux = (clip_tokenizer, clip_transform)

    print("2. 加载 DiT...")
    pipe = load_dit(Path(args.dit_path), device, fp16=args.fp16)

    n_init = 0
    if args.init_random_samples > 0:
        print("3a. 随机初始化...")
        n_init = init_random_samples(
            pipe,
            output_dir,
            args.init_random_samples,
            args.batch_size,
            args.deno_steps,
            device,
            args.seed,
            guidance_scale=args.guidance_scale,
        )
        print(f"   已写入 {n_init} 张")

    print("3b. Random Search (CLIPScore verifier)...")
    print(f"   deno_steps={args.deno_steps}, num_candidates={args.num_candidates}")
    run_random_search_clipscore(
        pipe,
        clip_backend,
        clip_model,
        clip_aux,
        labels_1k,
        args.prompt_template,
        output_dir,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        args.deno_steps,
        device,
        args.seed,
        guidance_scale=args.guidance_scale,
        n_init=n_init,
    )

    log_path = output_dir / "config.json"
    cfg = {
        "verifier": "clipscore",
        "clip_backend": clip_backend,
        "init_random_samples": args.init_random_samples,
        "deno_steps": args.deno_steps,
        "num_candidates": args.num_candidates,
        "prompt_template": args.prompt_template,
        "fp16": args.fp16,
    }
    if clip_backend == "hf" and clip_dir_resolved is not None:
        cfg["clip_model_path"] = str(Path(args.clip_model_path))
        cfg["clip_model_path_resolved"] = str(clip_dir_resolved)
    else:
        cfg["clip_model"] = args.clip_model
        cfg["clip_pretrained"] = args.clip_pretrained
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"\n完成。图像保存至 {output_dir}，配置: {log_path}")

    print("\n4. 计算最终 FID / IS...")
    fid_val, is_val = compute_fid_and_is(
        output_dir, imagenet_val, device, real_cache_name=args.ref_cache_name
    )
    print(f"   最终 FID: {fid_val:.2f}, IS: {is_val:.4f}")
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"FID": fid_val, "IS": is_val}, f, indent=2)


if __name__ == "__main__":
    main()
