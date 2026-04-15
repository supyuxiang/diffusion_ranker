#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from text2image.common import dump_json, ensure_dir, list_prompt_dirs


def _load_clip(model_dir: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    m = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(device).float().eval()
    p = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
    return m, p


@torch.inference_mode()
def _score_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Image.Image],
    prompts: list[str],
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    image_embeds = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    sim = (image_embeds * text_embeds).sum(dim=-1)
    return sim.float().cpu()


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser("Score candidate pool with CLIPScore")
    p.add_argument("--pool_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--benchmark", type=str, default="")
    p.add_argument("--clip_model_dir", type=str, default="/root/autodl-tmp/openai_clip-vit-large-patch14")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    out_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clip, processor = _load_clip(args.clip_model_dir, device)
    pdirs = list_prompt_dirs(args.pool_root, args.benchmark or None)
    if not pdirs:
        raise FileNotFoundError(f"no prompt folders under {args.pool_root}")

    all_means: list[float] = []
    for pdir in tqdm(pdirs, desc="clipscore prompts"):
        meta = (pdir / "meta.json")
        if not meta.is_file():
            continue
        obj = __import__("json").loads(meta.read_text(encoding="utf-8"))
        prompt = str(obj["prompt_text"])
        img_paths = sorted((pdir / "images").glob("image_*.png"))
        scores: list[torch.Tensor] = []
        for i in range(0, len(img_paths), int(args.batch_size)):
            batch_paths = img_paths[i : i + int(args.batch_size)]
            batch_images = [Image.open(x).convert("RGB").copy() for x in batch_paths]
            batch_prompts = [prompt] * len(batch_images)
            scores.append(_score_batch(clip, processor, batch_images, batch_prompts, device))
        s = torch.cat(scores, dim=0) if scores else torch.empty(0, dtype=torch.float32)
        all_means.append(float(s.mean().item()) if s.numel() > 0 else float("nan"))
        prompt_out_dir = out_dir / obj.get("benchmark", "custom")
        prompt_out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt_id": obj["prompt_id"],
                "benchmark": obj.get("benchmark", "custom"),
                "scores": s,
                "best_idx": int(torch.argmax(s).item()) if s.numel() > 0 else -1,
                "mean_score": float(s.mean().item()) if s.numel() > 0 else float("nan"),
            },
            prompt_out_dir / f"{obj['prompt_id']}.pt",
        )

    dump_json(
        {
            "reward": "clipscore",
            "num_prompts": len(pdirs),
            "global_mean_of_prompt_means": float(torch.tensor(all_means).nanmean().item()) if all_means else float("nan"),
        },
        out_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
