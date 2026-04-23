#!/usr/bin/env python3
"""
DiT Random Search with Qwen2.5-VL (Judge Score) as Verifier (Inference-Time Scaling)

与 dit_random_search_imagereward.py 结构相同，但每轮从 K 个候选 batch 中选
「batch 平均 Qwen Judge Score 最高」的一批写入已选集。

评分方式（Qwen-as-Judge）:
    给定 (image, text) 对，向 Qwen2.5-VL 发送如下消息：
        "Rate the quality and relevance of this image as a depiction of
         '{text}' on a scale from 1 to 10. Only output a single integer."
    Score = 解析模型输出中的数字（1-10）。

与 VQAScore (gpt2score) 的区别：
  - VQAScore 计算 P("Yes" | image, question) 的对数概率
  - 本脚本让 VLM 直接生成评分数字，利用其推理能力做"裁判"

用法:
  CUDA_VISIBLE_DEVICES=6 python3 dit_random_search_qwenscore.py \
    --dit_path /data/yxfeng/models/dit_model \
    --vlm_path /data/yxfeng/models/qwen/Qwen2.5-VL-7B-Instruct \
    --imagenet_val /data/yxfeng/datasets/imagenet/val \
    --output_dir /data/yxfeng/eval_dit_random_search_qwenscore \
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
import re
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

DEFAULT_JUDGE_PROMPT = (
    "Rate the quality and relevance of this image as a depiction of "
    "'{text}' on a scale from 1 to 10, where 1 means very poor quality or "
    "completely irrelevant, and 10 means excellent quality and highly relevant. "
    "Only output a single integer."
)


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


_NUM_RE = re.compile(r"\b(10|[1-9])\b")


def _parse_score(text: str, fallback: float = 1.0) -> float:
    m = _NUM_RE.search(text.strip())
    if m:
        return float(m.group(1))
    return fallback


def _qwen_judge_single(
    model,
    processor,
    device: str,
    pil_image: Image.Image,
    text_description: str,
    judge_prompt: str,
) -> float:
    prompt_text = judge_prompt.replace("{text}", text_description)

    rgb = pil_image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=1.0,
        )

    gen_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return _parse_score(response)


def mean_batch_qwenscore(
    model,
    processor,
    device: str,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
    judge_prompt: str,
) -> float:
    scores = []
    for img, cid in zip(images, class_ids):
        text = prompt_for_class(labels_1k, cid, prompt_template)
        s = _qwen_judge_single(model, processor, device, img, text, judge_prompt)
        scores.append(s)
    return float(np.mean(scores)) if scores else float("-inf")


def run_random_search_qwenscore(
    pipe: DiTPipeline,
    vlm_model,
    vlm_processor,
    labels_1k: list[str],
    prompt_template: str,
    judge_prompt: str,
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

    for _ in tqdm(range(num_rounds), desc="Random Search QwenScore"):
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
            score = mean_batch_qwenscore(
                vlm_model,
                vlm_processor,
                device,
                batch_images,
                batch_classes,
                labels_1k,
                prompt_template,
                judge_prompt,
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
    p.add_argument("--output_dir", default="/data/yxfeng/eval_dit_random_search_qwenscore")
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
        help="填入 judge prompt 中的文本描述模板；{name}、{id} 可用",
    )
    p.add_argument(
        "--judge_prompt",
        type=str,
        default=DEFAULT_JUDGE_PROMPT,
        help="Qwen Judge 评分提示词，{text} 会被替换为文本描述",
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

    print("3b. Random Search (Qwen Judge Score verifier)...")
    print(f"   deno_steps={args.deno_steps}, num_candidates={args.num_candidates}")
    run_random_search_qwenscore(
        pipe,
        vlm_model,
        vlm_processor,
        labels_1k,
        args.prompt_template,
        args.judge_prompt,
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
        "verifier": "qwenscore",
        "vlm_path": args.vlm_path,
        "vlm_dtype": args.vlm_dtype,
        "judge_prompt": args.judge_prompt,
        "score_type": "qwen_judge_1_to_10",
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
