#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import FluxPipeline
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text2image.ranker import DiffusionRanker_CrossAttention, DiffusionRanker_MLP, DiffusionRanker_Transformer
from utils import set_seed


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_ranker_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    *,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if "output_weights" in state:
        wc = state["linear_candidate.weight"]
        wp = state["linear_prompt.weight"]
        hidden_dim, input_dim_candidate = int(wc.shape[0]), int(wc.shape[1])
        input_dim_prompt = int(wp.shape[1])
        layer_ids = [
            int(k.split(".")[1])
            for k in state
            if k.startswith("layers.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
        ]
        num_layers = max(layer_ids) + 1 if layer_ids else 2
        model = DiffusionRanker_Transformer(
            input_dim_candidate=input_dim_candidate,
            input_dim_prompt=input_dim_prompt,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            num_layers=num_layers,
            half=False,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return model

    if "score_head.weight" in state:
        wc = state["process_candidate.0.weight"]
        wp = state["process_prompt.0.weight"]
        hidden_dim, input_dim_candidate = int(wc.shape[0]), int(wc.shape[1])
        input_dim_prompt = int(wp.shape[1])
        layer_ids = [
            int(k.split(".")[1])
            for k in state
            if k.startswith("attn_layers.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
        ]
        num_layers = max(layer_ids) + 1 if layer_ids else 2
        model = DiffusionRanker_CrossAttention(
            candidate_hidden_dim=input_dim_candidate,
            prompt_hidden_dim=input_dim_prompt,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            num_layers=num_layers,
            dropout=dropout,
            half=False,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return model

    if "projection.weight" in state and "process_prompt.0.weight" in state:
        wc = state["linear_candidate.weight"]
        wp = state["process_prompt.0.weight"]
        hidden_dim, input_dim_candidate = int(wc.shape[0]), int(wc.shape[1])
        input_dim_prompt = int(wp.shape[1])
        layer_ids = [
            int(k.split(".")[1])
            for k in state
            if k.startswith("layers.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
        ]
        num_layers = max(layer_ids) + 1 if layer_ids else 2
        model = DiffusionRanker_MLP(
            candidate_hidden_dim=input_dim_candidate,
            prompt_hidden_dim=input_dim_prompt,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            half=False,
            criterion=None,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return model

    raise ValueError(f"Unsupported ranker checkpoint format: {ckpt_path}")


def _ranker_dims(ranker: nn.Module) -> tuple[int, int]:
    if hasattr(ranker, "linear_candidate") and hasattr(ranker.linear_candidate, "in_features"):
        d_c = int(ranker.linear_candidate.in_features)
    else:
        raise ValueError("Cannot infer candidate dim from ranker.")
    if hasattr(ranker, "linear_prompt") and hasattr(ranker.linear_prompt, "in_features"):
        d_p = int(ranker.linear_prompt.in_features)
    elif hasattr(ranker, "process_prompt"):
        d_p = int(ranker.process_prompt[0].in_features)
    else:
        raise ValueError("Cannot infer prompt dim from ranker.")
    return d_c, d_p


def _prepare_prompt_vecs(pipe: FluxPipeline, prompts: list[str], expected_dim: int, device: torch.device) -> torch.Tensor:
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    mean_embeds = prompt_embeds.mean(dim=1)
    if mean_embeds.shape[1] == expected_dim:
        return mean_embeds.float().cpu()
    if pooled_prompt_embeds.shape[1] == expected_dim:
        return pooled_prompt_embeds.float().cpu()
    raise ValueError(
        f"Prompt dim mismatch: ranker expects {expected_dim}, "
        f"but flux mean={mean_embeds.shape[1]}, pooled={pooled_prompt_embeds.shape[1]}"
    )


def _load_candidate_repr(cand_dir: Path) -> tuple[torch.Tensor, list[str]]:
    obj = torch.load(cand_dir / "repr.pt", map_location="cpu", weights_only=False)
    hs = obj["hiddens"].float()
    files = list(obj["image_files"])
    if hs.shape[0] != len(files):
        raise ValueError(f"{cand_dir}: hidden rows != image files")
    return hs, files


@torch.inference_mode()
def run_rs(args: argparse.Namespace) -> None:
    rollout_root = Path(args.rollout_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_img_dir = out_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    prompts = json.loads((rollout_root / "prompts.json").read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError("rollout_root/prompts.json is empty or invalid.")

    cand_dirs = sorted([p for p in rollout_root.glob("candidate_*") if p.is_dir()])
    if len(cand_dirs) == 0:
        raise FileNotFoundError(f"No candidate dirs under {rollout_root}")
    if args.num_candidates > len(cand_dirs):
        raise ValueError(f"--num_candidates={args.num_candidates} > available={len(cand_dirs)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranker = load_ranker_from_ckpt(
        Path(args.ranker_ckpt).expanduser().resolve(),
        device,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
    )
    d_c, d_p = _ranker_dims(ranker)

    model_dtype = _resolve_dtype(args.dtype, device)
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=model_dtype, local_files_only=True).to(device)
    prompt_vecs = _prepare_prompt_vecs(pipe, [str(x) for x in prompts], d_p, device=device)

    cand_hs: list[torch.Tensor] = []
    cand_files: list[list[str]] = []
    for cd in cand_dirs:
        hs, files = _load_candidate_repr(cd)
        if hs.shape[1] != d_c:
            raise ValueError(f"{cd}: candidate dim {hs.shape[1]} != ranker expected {d_c}")
        if hs.shape[0] != len(prompts):
            raise ValueError(f"{cd}: prompts mismatch {hs.shape[0]} vs {len(prompts)}")
        cand_hs.append(hs)
        cand_files.append(files)

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    for pidx in tqdm(range(len(prompts)), desc="RS over prompts"):
        chosen = rng.choice(len(cand_dirs), size=args.num_candidates, replace=False).tolist()
        cand_tok = torch.stack([cand_hs[j][pidx] for j in chosen], dim=0).unsqueeze(0).to(device=device)
        prompt_vec = prompt_vecs[pidx].to(device=device)

        if isinstance(ranker, DiffusionRanker_Transformer):
            logits = ranker(cand_tok, prompt_vec.unsqueeze(0).unsqueeze(1))
        else:
            logits = ranker(cand_tok, prompt_vec.unsqueeze(0))
        logits = logits.view(-1).float().cpu()
        best_local = int(torch.argmax(logits).item())
        best_cand = chosen[best_local]
        best_score = float(logits[best_local].item())

        src_img = cand_dirs[best_cand] / "images" / cand_files[best_cand][pidx]
        dst_img = out_img_dir / f"image_{pidx:05d}_cand{best_cand:04d}.png"
        shutil.copy2(src_img, dst_img)

        records.append(
            {
                "prompt_idx": int(pidx),
                "prompt": str(prompts[pidx]),
                "selected_candidate": int(best_cand),
                "selected_score": float(best_score),
                "selected_image": str(dst_img),
                "candidate_indices": [int(x) for x in chosen],
                "candidate_scores": [float(x) for x in logits.tolist()],
            }
        )

    (out_dir / "rs_selected.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "rollout_root": str(rollout_root),
                "ranker_ckpt": str(args.ranker_ckpt),
                "num_candidates": int(args.num_candidates),
                "seed": int(args.seed),
                "num_prompts": int(len(prompts)),
                "model_path": str(args.model_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] RS outputs saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random search with trained ranker on pre-rolled FLUX prompt candidates.")
    p.add_argument("--rollout_root", type=str, required=True)
    p.add_argument("--ranker_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_candidates", type=int, default=8, help="Randomly sampled candidates per prompt.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/FLUX.1-dev")
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_rs(args)


if __name__ == "__main__":
    main()
