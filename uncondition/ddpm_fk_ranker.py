#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path("/root/diffusion_ranker")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.dit_random_search_ranker import load_diffusion_ranker
from utils import CLIP_HF_LOCAL_DIR, compute_clipscore, compute_fid_is_from_dirs, set_seed, _load_images_from_dir


def _sorted_rollout_pngs(d: Path) -> list[Path]:
    paths = [p for p in d.glob("*.png") if p.is_file()]

    def _key(p: Path) -> int:
        s = p.stem
        if s.isdigit():
            return int(s)
        for tok in s.split("_"):
            if tok.isdigit():
                return int(tok)
        raise ValueError(f"cannot parse index from {p.name}")

    return sorted(paths, key=_key)


@torch.inference_mode()
def _embed_paths(
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
    return feat


@torch.inference_mode()
def _mean_ranker_score(ranker, cand_feat: torch.Tensor, device: torch.device) -> float:
    w = next(ranker.parameters())
    cand = cand_feat.to(device=device, dtype=w.dtype).unsqueeze(1)
    logits = ranker(cand, None)
    return float(logits.float().mean().item())


def _update_fk_state(
    log_belief: torch.Tensor,
    scores: torch.Tensor,
    *,
    tau: float,
    potential: str,
    prev_scores: torch.Tensor | None,
    running_max: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if potential == "difference":
        inc = scores if prev_scores is None else (scores - prev_scores)
        log_belief = log_belief + float(tau) * inc
        prev_scores = scores.clone()
    else:
        running_max = scores.clone() if running_max is None else torch.maximum(running_max, scores)
        log_belief = float(tau) * running_max
    w = torch.softmax(log_belief, dim=0)
    return log_belief, w, prev_scores, running_max


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DDPM FK steering with trained ranker")
    p.add_argument("--ranker_ckpt", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--candidates_root",
        type=Path,
        default=Path("/root/autodl-tmp/ddpm_candidates_50k_deno10"),
    )
    p.add_argument("--num_candidates", type=int, default=10)
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip_model_dir", type=Path, default=CLIP_HF_LOCAL_DIR)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--potential", type=str, choices=("difference", "max"), default="difference")
    p.add_argument("--pick", type=str, choices=("sample", "argmax"), default="argmax")
    p.add_argument("--real_dir", type=Path, default=Path("/root/autodl-tmp/img_align_celeba_fid_ref_50k"))
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument("--compute_clipscore", action="store_true")
    p.add_argument("--clip_prompt", type=str, default="a photo of a human face")
    p.add_argument("--clip_batch_size", type=int, default=64)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best_images"
    best_dir.mkdir(parents=True, exist_ok=True)

    ranker, _keys, _listwise, _meta = load_diffusion_ranker(
        args.ranker_ckpt.expanduser().resolve(),
        device,
        num_attention_heads=int(args.num_attention_heads),
        dropout=float(args.dropout),
    )
    clip_dir = args.clip_model_dir.expanduser().resolve()
    clip = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True).to(device).float().eval()
    processor = CLIPProcessor.from_pretrained(str(clip_dir), local_files_only=True)

    candidate_paths: dict[int, list[Path]] = {}
    for i in range(int(args.num_candidates)):
        cdir = args.candidates_root.expanduser().resolve() / f"candidate_{i:04d}"
        if not cdir.is_dir():
            raise FileNotFoundError(f"missing candidate directory: {cdir}")
        paths = _sorted_rollout_pngs(cdir)
        if len(paths) < int(args.num_samples):
            raise ValueError(f"{cdir} only has {len(paths)} images < num_samples {args.num_samples}")
        candidate_paths[i] = paths

    loops = (int(args.num_samples) + int(args.batch_size) - 1) // int(args.batch_size)
    log_belief = torch.zeros(int(args.num_candidates), dtype=torch.float64)
    prev_scores: torch.Tensor | None = None
    running_max: torch.Tensor | None = None
    img_counter = 0
    trace: list[dict] = []
    for step in tqdm(range(loops), desc="ddpm fk ranker"):
        n_this = min(int(args.batch_size), int(args.num_samples) - img_counter)
        if n_this <= 0:
            break
        scores = torch.empty(int(args.num_candidates), dtype=torch.float64)
        for i in range(int(args.num_candidates)):
            batch_paths = candidate_paths[i][img_counter : img_counter + n_this]
            emb = _embed_paths(clip, processor, batch_paths, device)
            scores[i] = _mean_ranker_score(ranker, emb, device)

        log_belief, w, prev_scores, running_max = _update_fk_state(
            log_belief,
            scores,
            tau=float(args.tau),
            potential=str(args.potential),
            prev_scores=prev_scores,
            running_max=running_max,
        )
        if args.pick == "sample":
            pick_idx = int(torch.multinomial(w.float(), 1).item())
        else:
            pick_idx = int(torch.argmax(w).item())
        picked_paths = candidate_paths[pick_idx][img_counter : img_counter + n_this]
        for p in picked_paths:
            shutil.copy2(p, best_dir / f"image_{img_counter:05d}.png")
            img_counter += 1
        trace.append(
            {
                "step": int(step),
                "scores": [float(x) for x in scores.tolist()],
                "weights": [float(x) for x in w.tolist()],
                "picked_candidate": int(pick_idx),
            }
        )

    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "ranker_ckpt": str(args.ranker_ckpt.expanduser().resolve()),
                "candidates_root": str(args.candidates_root.expanduser().resolve()),
                "num_candidates": int(args.num_candidates),
                "num_samples": int(args.num_samples),
                "batch_size": int(args.batch_size),
                "tau": float(args.tau),
                "potential": str(args.potential),
                "pick": str(args.pick),
                "seed": int(args.seed),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    real_dir = args.real_dir.expanduser().resolve()
    if real_dir.is_dir():
        dev_str = "cuda" if device.type == "cuda" else "cpu"
        metrics: dict[str, float] = {
            k: float(v)
            for k, v in compute_fid_is_from_dirs(
                best_dir,
                real_dir,
                dev_str,
                batch_size=int(args.metrics_batch_size),
            ).items()
        }
        if args.compute_clipscore:
            gen_images = _load_images_from_dir(best_dir)
            prompts = [str(args.clip_prompt)] * int(gen_images.shape[0])
            metrics["CLIPScore"] = float(
                compute_clipscore(
                    gen_images,
                    prompts,
                    device=device,
                    clip_model_dir=clip_dir,
                    clip_batch_size=int(args.clip_batch_size),
                )
            )
            del gen_images, prompts
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        (best_dir / "final_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(metrics)


if __name__ == "__main__":
    main()
