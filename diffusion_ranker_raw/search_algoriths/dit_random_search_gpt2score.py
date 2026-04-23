#!/usr/bin/env python3
"""
DiT Random Search with VQAScore (Qwen2.5-VL) as Verifier (Inference-Time Scaling)

与 dit_random_search_imagereward.py 结构相同，但每轮从 K 个候选 batch 中选
「batch 平均 VQAScore 最高」的一批写入已选集。

VQAScore 定义（ECCV 2024, Lin et al.）:
    对 (image, text) 对，构造 VQA 问题:
        "Does this figure show '{text}'? Please answer yes or no."
    Score = P("Yes" | image, question)
         = exp( -CrossEntropy(answer_tokens) )

本脚本使用本地 Qwen2.5-VL-7B-Instruct 作为 VQA 模型。
每张图使用对应 ImageNet 类名填充文本模板。

用法:
  CUDA_VISIBLE_DEVICES=6 python3 dit_random_search_gpt2score.py \
    --dit_path /data/yxfeng/models/dit_model \
    --vlm_path /data/yxfeng/models/qwen/Qwen2.5-VL-7B-Instruct \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --output_dir /data/yxfeng/eval_dit_random_search_vqascore \
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

IGNORE_INDEX = -100
DEFAULT_QUESTION_TEMPLATE = 'Does this figure show "{}"? Please answer yes or no.'
DEFAULT_ANSWER = "Yes"


def load_vlm(vlm_path: str, device: str, dtype: torch.dtype):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        vlm_path,
        local_files_only=True,
        min_pixels=224 * 224,
        max_pixels=512 * 512,
    )
    return model, processor


def _vqascore_single(
    model,
    processor,
    device: str,
    pil_image: Image.Image,
    text: str,
    question_template: str,
    answer: str,
) -> float:
    question = question_template.format(text)

    messages_with_answer = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]

    messages_question_only = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]

    rgb = pil_image.convert("RGB")

    full_inputs = processor.apply_chat_template(
        messages_with_answer,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images=[rgb],
    )

    question_inputs = processor.apply_chat_template(
        messages_question_only,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images=[rgb],
    )

    full_ids = full_inputs["input_ids"]
    q_len = question_inputs["input_ids"].shape[1]

    labels = full_ids.clone()
    labels[:, :q_len] = IGNORE_INDEX

    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(**full_inputs, labels=labels)
        loss = outputs.loss

    return float(torch.exp(-loss).item())


def mean_batch_vqascore(
    model,
    processor,
    device: str,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
    question_template: str,
    answer: str,
) -> float:
    scores = []
    for img, cid in zip(images, class_ids):
        text = prompt_for_class(labels_1k, cid, prompt_template)
        s = _vqascore_single(model, processor, device, img, text, question_template, answer)
        scores.append(s)
    return float(np.mean(scores)) if scores else float("-inf")


def run_random_search_vqascore(
    pipe: DiTPipeline,
    vlm_model,
    vlm_processor,
    labels_1k: list[str],
    prompt_template: str,
    question_template: str,
    answer: str,
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

    for _ in tqdm(range(num_rounds), desc="Random Search VQAScore"):
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
            score = mean_batch_vqascore(
                vlm_model,
                vlm_processor,
                device,
                batch_images,
                batch_classes,
                labels_1k,
                prompt_template,
                question_template,
                answer,
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
    p.add_argument("--output_dir", default="/data/yxfeng/eval_dit_random_search_vqascore")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--num_candidates",
        type=int,
        default=8,
        help="每轮随机采样的候选 batch 数（不同 seed）",
    )
    p.add_argument("--deno_steps", type=int, default=10)
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
        help="填入 VQA 问题中的文本描述模板；{name}、{id} 可用",
    )
    p.add_argument(
        "--question_template",
        type=str,
        default=DEFAULT_QUESTION_TEMPLATE,
        help="VQAScore 问题模板，{} 占位文本描述",
    )
    p.add_argument(
        "--answer",
        type=str,
        default=DEFAULT_ANSWER,
        help="VQAScore 期望答案（默认 Yes）",
    )
    p.add_argument(
        "--vlm_path",
        type=str,
        default="/data/yxfeng/models/qwen/Qwen2.5-VL-7B-Instruct",
        help="本地 Qwen2.5-VL 模型目录",
    )
    p.add_argument(
        "--vlm_dtype",
        type=str,
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="VLM 精度",
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

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    vlm_dtype = dtype_map[args.vlm_dtype]

    print(f"1. 加载 VLM (Qwen2.5-VL): {args.vlm_path}")
    vlm_model, vlm_processor = load_vlm(args.vlm_path, device, vlm_dtype)

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

    print("3b. Random Search (VQAScore verifier)...")
    print(
        f"   deno_steps={args.deno_steps}, num_candidates={args.num_candidates}, "
        f"score=P(\"{args.answer}\" | image, question)"
    )
    run_random_search_vqascore(
        pipe,
        vlm_model,
        vlm_processor,
        labels_1k,
        args.prompt_template,
        args.question_template,
        args.answer,
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
        "verifier": "vqascore",
        "vlm_path": args.vlm_path,
        "vlm_dtype": args.vlm_dtype,
        "question_template": args.question_template,
        "answer": args.answer,
        "score_type": "P(answer | image, question)",
        "init_random_samples": args.init_random_samples,
        "deno_steps": args.deno_steps,
        "num_candidates": args.num_candidates,
        "prompt_template": args.prompt_template,
        "fp16": args.fp16,
    }
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
