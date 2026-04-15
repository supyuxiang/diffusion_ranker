#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from text2image.common import dump_json, ensure_dir, list_prompt_dirs


def _score_with_pickscore(
    model,
    processor,
    prompt: str,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        img_in = processor(images=images, return_tensors="pt")
        txt_in = processor(text=[prompt] * len(images), return_tensors="pt", padding=True)
        img_in = {k: v.to(device) for k, v in img_in.items()}
        txt_in = {k: v.to(device) for k, v in txt_in.items()}
        image_emb = model.get_image_features(**img_in)
        text_emb = model.get_text_features(**txt_in)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return (image_emb * text_emb).sum(dim=-1).float().cpu()


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser("Score candidate pool with PickScore/HPSv2")
    p.add_argument("--pool_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--benchmark", type=str, default="")
    p.add_argument("--mode", type=str, choices=("pickscore", "hpsv2"), default="pickscore")
    p.add_argument("--model_name_or_path", type=str, default="/root/autodl-tmp/PickScore_v1")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.mode == "hpsv2":
        raise NotImplementedError(
            "hpsv2 direct scorer varies by installed package version. "
            "Use --mode pickscore now, or plug your local hpsv2 API in this file."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    model = AutoModel.from_pretrained(args.model_name_or_path, local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, local_files_only=True)
    pdirs = list_prompt_dirs(args.pool_root, args.benchmark or None)
    if not pdirs:
        raise FileNotFoundError(f"no prompt folders under {args.pool_root}")

    all_means: list[float] = []
    for pdir in tqdm(pdirs, desc=f"{args.mode} prompts"):
        meta_path = pdir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prompt = str(meta["prompt_text"])
        img_paths = sorted((pdir / "images").glob("image_*.png"))
        chunks: list[torch.Tensor] = []
        for i in range(0, len(img_paths), int(args.batch_size)):
            batch_paths = img_paths[i : i + int(args.batch_size)]
            batch_images = [Image.open(x).convert("RGB").copy() for x in batch_paths]
            chunks.append(_score_with_pickscore(model, processor, prompt, batch_images, device))
        scores = torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.float32)
        all_means.append(float(scores.mean().item()) if scores.numel() > 0 else float("nan"))
        prompt_out_dir = out_dir / meta.get("benchmark", "custom")
        prompt_out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt_id": meta["prompt_id"],
                "benchmark": meta.get("benchmark", "custom"),
                "scores": scores,
                "best_idx": int(torch.argmax(scores).item()) if scores.numel() > 0 else -1,
                "mean_score": float(scores.mean().item()) if scores.numel() > 0 else float("nan"),
            },
            prompt_out_dir / f"{meta['prompt_id']}.pt",
        )

    dump_json(
        {
            "reward": str(args.mode),
            "num_prompts": len(pdirs),
            "global_mean_of_prompt_means": float(torch.tensor(all_means).nanmean().item()) if all_means else float("nan"),
        },
        out_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
