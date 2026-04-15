#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from text2image.common import dump_json, load_prompts_jsonl


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser("Lightweight GenEval runner (CLIP proxy)")
    p.add_argument("--prompts_jsonl", type=str, required=True)
    p.add_argument("--selected_dir", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--clip_model_dir", type=str, default="/root/autodl-tmp/openai_clip-vit-large-patch14")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    prompts = load_prompts_jsonl(args.prompts_jsonl)
    selected_dir = Path(args.selected_dir).expanduser().resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clip = CLIPModel.from_pretrained(args.clip_model_dir, local_files_only=True).to(device).float().eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model_dir, local_files_only=True)

    rows: list[dict] = []
    for i in tqdm(range(0, len(prompts), int(args.batch_size)), desc="geneval"):
        chunk = prompts[i : i + int(args.batch_size)]
        paths = [selected_dir / f"{x.prompt_id}.png" for x in chunk]
        valid = [(x, pth) for x, pth in zip(chunk, paths) if pth.is_file()]
        if not valid:
            continue
        texts = [x.prompt_text for x, _ in valid]
        images = [Image.open(pth).convert("RGB").copy() for _, pth in valid]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = clip(**inputs)
        im = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        tx = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (im * tx).sum(dim=-1).float().cpu()
        for (item, _), s in zip(valid, sim):
            rows.append(
                {
                    "prompt_id": item.prompt_id,
                    "benchmark": item.benchmark,
                    "clipscore": float(s.item()),
                }
            )

    score = float(torch.tensor([r["clipscore"] for r in rows], dtype=torch.float32).mean().item()) if rows else float("nan")
    dump_json(
        {
            "metric_name": "geneval_clip_proxy",
            "num_evaluated": len(rows),
            "score": score,
            "rows": rows,
        },
        args.output_json,
    )


if __name__ == "__main__":
    main()
