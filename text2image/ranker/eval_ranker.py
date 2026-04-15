#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from search.dit_random_search_ranker import load_diffusion_ranker
from text2image.common import dump_json, ensure_dir, load_prompts_jsonl, safe_image_name


@torch.inference_mode()
def _embed_images(
    clip: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    for i in range(0, len(image_paths), batch_size):
        paths = image_paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB").copy() for p in paths]
        inp = processor(images=imgs, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        feat = clip.get_image_features(**inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        outs.append(feat.float().cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty(0, dtype=torch.float32)


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser("Evaluate trained ranker as reward model")
    p.add_argument("--prompts_jsonl", type=str, required=True)
    p.add_argument("--pool_root", type=str, required=True)
    p.add_argument("--ranker_ckpt", type=str, required=True)
    p.add_argument("--prompt_table_pt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--clip_model_dir", type=str, default="/root/autodl-tmp/openai_clip-vit-large-patch14")
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    prompts = load_prompts_jsonl(args.prompts_jsonl)
    pool_root = Path(args.pool_root).expanduser().resolve()
    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ranker, _keys, _listwise, _meta = load_diffusion_ranker(
        Path(args.ranker_ckpt).expanduser().resolve(),
        device,
        num_attention_heads=int(args.num_attention_heads),
        dropout=float(args.dropout),
    )
    pobj = torch.load(Path(args.prompt_table_pt).expanduser().resolve(), map_location="cpu", weights_only=False)
    prompt_table = pobj["tensor"] if isinstance(pobj, dict) and "tensor" in pobj else pobj
    if not isinstance(prompt_table, torch.Tensor) or prompt_table.dim() != 2:
        raise ValueError("prompt_table_pt must contain a 2D tensor")

    clip = CLIPModel.from_pretrained(args.clip_model_dir, local_files_only=True).to(device).float().eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model_dir, local_files_only=True)

    rows: list[dict] = []
    for prompt_idx, item in enumerate(tqdm(prompts, desc="ranker eval")):
        img_dir = pool_root / item.benchmark / item.prompt_id / "images"
        paths = sorted(img_dir.glob("image_*.png"))
        if not paths:
            continue
        img_emb = _embed_images(clip, processor, paths, device, int(args.batch_size))
        if img_emb.numel() == 0:
            continue
        cand = img_emb.to(device).unsqueeze(1).to(next(ranker.parameters()).dtype)
        p = prompt_table[prompt_idx].to(device=device, dtype=next(ranker.parameters()).dtype)
        p = p.unsqueeze(0).expand(cand.shape[0], -1)
        logits = ranker(cand, p).float().view(-1).cpu()
        best_idx = int(torch.argmax(logits).item())
        src = img_dir / safe_image_name(best_idx)
        dst = output_dir / f"{item.prompt_id}.png"
        shutil.copy2(src, dst)
        rows.append(
            {
                "prompt_id": item.prompt_id,
                "benchmark": item.benchmark,
                "chosen_idx": best_idx,
                "chosen_score": float(logits[best_idx].item()),
            }
        )

    (output_dir / "selections.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    dump_json(
        {
            "num_selected": len(rows),
            "mean_chosen_score": (
                float(torch.tensor([r["chosen_score"] for r in rows], dtype=torch.float32).mean().item())
                if rows
                else float("nan")
            ),
        },
        output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
