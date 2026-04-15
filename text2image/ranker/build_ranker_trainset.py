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

from text2image.common import ensure_dir, list_prompt_dirs, safe_image_name


@torch.inference_mode()
def _clip_embed_text(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompts: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        tin = processor(text=chunk, return_tensors="pt", padding=True)
        tin = {k: v.to(device) for k, v in tin.items()}
        feat = model.get_text_features(**tin)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        outs.append(feat.float().cpu())
    return torch.cat(outs, dim=0)


@torch.inference_mode()
def _clip_embed_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB").copy() for p in batch_paths]
        iin = processor(images=images, return_tensors="pt")
        iin = {k: v.to(device) for k, v in iin.items()}
        feat = model.get_image_features(**iin)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        outs.append(feat.float().cpu())
    return torch.cat(outs, dim=0)


def main() -> None:
    p = argparse.ArgumentParser("Build pointwise ranker trainset from pre-scored pool")
    p.add_argument("--pool_root", type=str, required=True)
    p.add_argument("--teacher_score_dir", type=str, required=True)
    p.add_argument("--output_pt", type=str, required=True)
    p.add_argument("--clip_model_dir", type=str, default="/root/autodl-tmp/openai_clip-vit-large-patch14")
    p.add_argument("--benchmark", type=str, default="")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    pool_root = Path(args.pool_root).expanduser().resolve()
    score_dir = Path(args.teacher_score_dir).expanduser().resolve()
    out_pt = Path(args.output_pt).expanduser().resolve()
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clip = CLIPModel.from_pretrained(args.clip_model_dir, local_files_only=True).to(device).float().eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model_dir, local_files_only=True)

    pdirs = list_prompt_dirs(pool_root, args.benchmark or None)
    prompt_items: list[tuple[str, str, str]] = []
    for pdir in pdirs:
        mp = pdir / "meta.json"
        if not mp.is_file():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        prompt_items.append((meta["benchmark"], meta["prompt_id"], meta["prompt_text"]))
    prompt_items = sorted(prompt_items)
    if not prompt_items:
        raise RuntimeError("no prompt meta found")

    prompt_texts = [x[2] for x in prompt_items]
    prompt_emb = _clip_embed_text(clip, processor, prompt_texts, device, int(args.batch_size))
    prompt_id_to_index = {(b, pid): i for i, (b, pid, _) in enumerate(prompt_items)}

    cand_rows: list[torch.Tensor] = []
    prompt_idx_rows: list[int] = []
    reward_rows: list[float] = []
    image_rel_paths: list[str] = []
    for bmk, pid, _txt in tqdm(prompt_items, desc="build ranker trainset"):
        spt = score_dir / bmk / f"{pid}.pt"
        if not spt.is_file():
            continue
        rec = torch.load(spt, map_location="cpu", weights_only=False)
        scores = rec["scores"].float()
        n = int(scores.numel())
        if n == 0:
            continue
        img_paths = [pool_root / bmk / pid / "images" / safe_image_name(i) for i in range(n)]
        img_emb = _clip_embed_images(clip, processor, img_paths, device, int(args.batch_size))
        if img_emb.shape[0] != n:
            raise RuntimeError(f"embedding row mismatch for {(bmk, pid)}")

        idx = prompt_id_to_index[(bmk, pid)]
        for i in range(n):
            cand_rows.append(img_emb[i])
            prompt_idx_rows.append(idx)
            reward_rows.append(float(scores[i].item()))
            image_rel_paths.append(str(Path(bmk) / pid / "images" / safe_image_name(i)))

    if not cand_rows:
        raise RuntimeError("empty trainset after scan")

    candidate_tensor = torch.stack(cand_rows, dim=0)
    prompt_index = torch.tensor(prompt_idx_rows, dtype=torch.long)
    reward_tensor = torch.tensor(reward_rows, dtype=torch.float32)
    torch.save(
        {
            "candidate_tensor": candidate_tensor,
            "prompt_table": prompt_emb,
            "prompt_index": prompt_index,
            "reward": reward_tensor,
            "prompt_items": prompt_items,
            "image_rel_paths": image_rel_paths,
            "repr_key": "clip_image",
        },
        out_pt,
    )
    ensure_dir(out_pt.parent)


if __name__ == "__main__":
    main()
