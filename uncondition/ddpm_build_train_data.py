#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path("/root/diffusion_ranker")
SEARCH_DIR = REPO_ROOT / "search"
for p in (REPO_ROOT, SEARCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from reward import load_imagereward
from utils import CLIP_HF_LOCAL_DIR, set_seed


def _sorted_rollout_pngs(d: Path) -> list[Path]:
    paths = [p for p in d.glob("*.png") if p.is_file()]

    def _key(p: Path) -> int:
        s = p.stem
        if s.isdigit(): # 如果 s 是数字，则返回 int(s)
            return int(s)
        for tok in s.split("_"): # 如果 s 不是数字，则分割 s 并返回第一个数字
            if tok.isdigit():
                return int(tok)
        raise ValueError(f"cannot parse index from {p.name}") # 如果 s 不是数字，则抛出错误

    return sorted(paths, key=_key)


@torch.inference_mode()
def _clip_image_embedding(
    clip: CLIPModel,
    processor: CLIPProcessor,
    paths: list[Path],
    device: torch.device,
) -> torch.Tensor:
    imgs = [Image.open(p).convert("RGB").copy() for p in paths]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feat = clip.get_image_features(**inputs)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.float().cpu()


@torch.inference_mode()
def _clip_teacher_score(
    clip: CLIPModel,
    processor: CLIPProcessor,
    prompt: str,
    paths: list[Path],
    device: torch.device,
) -> torch.Tensor:
    imgs = [Image.open(p).convert("RGB").copy() for p in paths]
    inputs = processor(
        text=[prompt] * len(imgs),
        images=imgs,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = clip(**inputs)
    image_embeds = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return (image_embeds * text_embeds).sum(dim=-1).float().cpu()


@torch.inference_mode()
def _imagereward_teacher_score(
    rm_model,
    prompt: str,
    paths: list[Path],
) -> torch.Tensor:
    vals: list[float] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        vals.append(float(rm_model.score(prompt, img)))
    return torch.tensor(vals, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build DDPM ranker train data")
    p.add_argument(
        "--candidates_root",
        type=Path,
        default=Path("/root/autodl-tmp/ddpm_candidates_50k_deno10"),
    )
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument(
        "--max_per_candidate",
        type=int,
        default=5000,
        help="每个 candidate 最多采样多少图用于训练（避免 80 万样本过大）",
    )
    p.add_argument(
        "--teacher",
        type=str,
        choices=("imagereward", "clipscore"),
        default="imagereward",
    )
    p.add_argument(
        "--teacher_prompt",
        type=str,
        default="a photo of a human face",
    )
    p.add_argument(
        "--clip_model_dir",
        type=Path,
        default=CLIP_HF_LOCAL_DIR,
        help="既用于 image embedding，也可用于 clipscore teacher",
    )
    p.add_argument(
        "--repr_root",
        type=Path,
        default=None,
        help="可选：DDPM rollout 表征根目录（含 candidate_xxxx/<repr_key>.pt）",
    )
    p.add_argument(
        "--repr_key",
        type=str,
        default="s9_r0.6",
        help="当 --repr_root 提供时使用的表征键名（例如 s9_r0.6）",
    )
    p.add_argument("--embed_batch_size", type=int, default=128)
    p.add_argument("--score_batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_pt",
        type=Path,
        default=Path("/root/autodl-tmp/ddpm_ranker_train_data_deno10.pt"),
    )
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_pt = args.output_pt.expanduser().resolve()
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    use_repr = args.repr_root is not None
    repr_root = args.repr_root.expanduser().resolve() if use_repr else None
    need_clip = (not use_repr) or (args.teacher == "clipscore")
    clip = None
    processor = None
    clip_dir = args.clip_model_dir.expanduser().resolve()
    if need_clip:
        clip = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True).to(device).float().eval()
        processor = CLIPProcessor.from_pretrained(str(clip_dir), local_files_only=True)

    rm_model = None
    if args.teacher == "imagereward":
        rm_model = load_imagereward(device)

    emb_chunks: list[torch.Tensor] = []
    reward_chunks: list[torch.Tensor] = []
    cand_idx_chunks: list[torch.Tensor] = []
    img_idx_chunks: list[torch.Tensor] = []
    source_path_ls: list[str] = []

    for cand_idx in tqdm(range(args.num_candidates), desc="build ddpm ranker data"):
        cdir = args.candidates_root.expanduser().resolve() / f"candidate_{cand_idx:04d}"
        if not cdir.is_dir():
            raise FileNotFoundError(f"missing candidate directory: {cdir}")
        all_paths = _sorted_rollout_pngs(cdir)
        if len(all_paths) == 0:
            raise RuntimeError(f"no png files in {cdir}")
        if args.max_per_candidate > 0 and len(all_paths) > args.max_per_candidate:
            pick_ids = sorted(rng.sample(range(len(all_paths)), args.max_per_candidate))
            sel_paths = [all_paths[i] for i in pick_ids]
            sel_img_idx = pick_ids
        else:
            sel_paths = all_paths
            sel_img_idx = list(range(len(sel_paths)))

        local_emb_chunks: list[torch.Tensor] = []
        local_reward_chunks: list[torch.Tensor] = []
        if use_repr:
            repr_pt = repr_root / f"candidate_{cand_idx:04d}" / f"{args.repr_key}.pt"
            if not repr_pt.is_file():
                raise FileNotFoundError(f"missing repr file: {repr_pt}")
            robj = torch.load(repr_pt, map_location="cpu", weights_only=False)
            rten = robj["tensor"] if isinstance(robj, dict) and "tensor" in robj else robj
            if not isinstance(rten, torch.Tensor) or rten.dim() != 2:
                raise ValueError(f"{repr_pt}: expected 2D tensor repr, got {type(rten)} {getattr(rten, 'shape', None)}")
            if rten.shape[0] < len(all_paths):
                raise ValueError(f"{repr_pt}: repr rows {rten.shape[0]} < image count {len(all_paths)}")
            local_emb_chunks.append(rten[sel_img_idx].float().cpu())
        else:
            for st in range(0, len(sel_paths), int(args.embed_batch_size)):
                batch_paths = sel_paths[st : st + int(args.embed_batch_size)]
                local_emb_chunks.append(_clip_image_embedding(clip, processor, batch_paths, device))
        for st in range(0, len(sel_paths), int(args.score_batch_size)):
            batch_paths = sel_paths[st : st + int(args.score_batch_size)]
            if args.teacher == "clipscore":
                local_reward_chunks.append(
                    _clip_teacher_score(
                        clip,
                        processor,
                        str(args.teacher_prompt),
                        batch_paths,
                        device,
                    )
                )
            else:
                local_reward_chunks.append(
                    _imagereward_teacher_score(
                        rm_model,
                        str(args.teacher_prompt),
                        batch_paths,
                    )
                )
        emb = torch.cat(local_emb_chunks, dim=0)
        rew = torch.cat(local_reward_chunks, dim=0)
        if emb.shape[0] != rew.shape[0]:
            raise RuntimeError(f"embedding/reward size mismatch for candidate {cand_idx}")

        emb_chunks.append(emb)
        reward_chunks.append(rew)
        cand_idx_chunks.append(torch.full((emb.shape[0],), cand_idx, dtype=torch.long))
        img_idx_chunks.append(torch.tensor(sel_img_idx, dtype=torch.long))
        source_path_ls.extend([str(p) for p in sel_paths])

    candidate_tensor = torch.cat(emb_chunks, dim=0)
    reward_tensor = torch.cat(reward_chunks, dim=0)
    candidate_idx = torch.cat(cand_idx_chunks, dim=0)
    image_idx = torch.cat(img_idx_chunks, dim=0)

    payload = {
        "candidate_tensor": candidate_tensor,
        "reward": reward_tensor,
        "candidate_idx": candidate_idx,
        "image_idx": image_idx,
        "source_paths": source_path_ls,
        "repr_key": str(args.repr_key) if use_repr else "ddpm_clip_image",
        "meta": {
            "candidates_root": str(args.candidates_root.expanduser().resolve()),
            "repr_root": str(repr_root) if use_repr else None,
            "repr_key": str(args.repr_key) if use_repr else None,
            "num_candidates": int(args.num_candidates),
            "max_per_candidate": int(args.max_per_candidate),
            "teacher": str(args.teacher),
            "teacher_prompt": str(args.teacher_prompt),
            "clip_model_dir": str(clip_dir),
            "embed_batch_size": int(args.embed_batch_size),
            "score_batch_size": int(args.score_batch_size),
            "seed": int(args.seed),
            "n_samples": int(candidate_tensor.shape[0]),
            "hidden_dim": int(candidate_tensor.shape[1]),
        },
    }
    torch.save(payload, out_pt)
    print(
        json.dumps(
            {
                "saved": str(out_pt),
                "n_samples": int(candidate_tensor.shape[0]),
                "hidden_dim": int(candidate_tensor.shape[1]),
                "teacher": str(args.teacher),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
