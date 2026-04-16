#!/usr/bin/env python3

"""
Single-rollout grid builder for ranker mid-step data.

Compared with the previous version (25 independent rollouts), this script:
1) runs ONE DiT rollout with multi-(layer_ratio, step_ratio) capture
2) computes ImageReward once on generated images
3) splits captured representations into per-case files
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from search.reward import labels_1k, load_imagereward
from utils import DitRolloutReprCapture, dit_loader, set_seed, rollout_dit_with_steps


LAYER_RATIO_LS = [0.1, 0.3, 0.6, 0.9, 1.0]
STEP_RATIO_LS = [0.0, 0.25, 0.5, 0.75, 1.0]


def step_ratio_to_idx(step_ratio: float, num_inference_steps: int) -> int:
    t = int(num_inference_steps)
    if t <= 1:
        return 0
    return min(t - 1, max(0, round(float(step_ratio) * (t - 1))))


def safe_tag(x: float) -> str:
    return f"{float(x):g}".replace(".", "p")


def build_case_tag(layer_ratio: float, step_ratio: float, num_inference_steps: int) -> str:
    s_idx = step_ratio_to_idx(step_ratio, num_inference_steps)
    return f"l{safe_tag(layer_ratio)}_s{s_idx}"


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _sorted_generated_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.glob("image_*_*.png"), key=lambda p: int(p.stem.split("_")[1]))


def _build_case_map(layer_ratios: list[float], step_ratios: list[float], num_inference_steps: int) -> list[dict]:
    out: list[dict] = []
    for lr in layer_ratios:
        for sr in step_ratios:
            s_idx = step_ratio_to_idx(sr, num_inference_steps)
            out.append(
                {
                    "layer_ratio": float(lr),
                    "step_ratio": float(sr),
                    "step_idx": int(s_idx),
                    "repr_key": f"s{s_idx}_r{float(lr):g}",
                    "case_tag": build_case_tag(lr, sr, num_inference_steps),
                }
            )
    return out


@torch.inference_mode()
def _run_shared_rollout(
    *,
    model_name: str,
    num_samples: int,
    num_inference_steps: int,
    guidance_scale: float,
    batch_size: int,
    seed: int,
    dtype: str,
    layer_ratios: list[float],
    step_ratios: list[float],
    shared_image_dir: Path,
    shared_repr_pt: Path,
    overwrite: bool,
) -> None:
    if shared_repr_pt.is_file() and shared_image_dir.is_dir() and (not overwrite):
        print(f"[shared rollout] reuse existing: {shared_repr_pt}")
        return

    shared_image_dir.mkdir(parents=True, exist_ok=True)
    shared_repr_pt.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(model_name, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    capture = DitRolloutReprCapture(
        pipe,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        layer_ratios=tuple(float(x) for x in layer_ratios),
        step_ratio_ls=[float(x) for x in step_ratios],
        store_dtype=torch.float16,
        rollout_base_seed=int(seed),
    )
    capture.install()
    try:
        rollout_dit_with_steps(
            pipe=pipe,
            save_dir=shared_image_dir,
            num_steps=int(num_inference_steps),
            num_samples=int(num_samples),
            batch_size=int(batch_size),
            seed=int(seed),
            guidance_scale=float(guidance_scale),
            repr_capture=capture,
            repr_max_batches=None,
        )
        capture.save_pt(shared_repr_pt)
    finally:
        capture.remove()


@torch.inference_mode()
def _compute_imagereward_tensor(image_dir: Path, reward_pt: Path, overwrite: bool) -> torch.Tensor:
    if reward_pt.is_file() and (not overwrite):
        obj = torch.load(reward_pt, map_location="cpu", weights_only=False)
        r = obj["reward"] if isinstance(obj, dict) and "reward" in obj else obj
        if not isinstance(r, torch.Tensor) or r.dim() != 1:
            raise RuntimeError(f"{reward_pt}: invalid cached reward tensor")
        print(f"[imagereward] reuse existing: {reward_pt}")
        return r.float().cpu()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rm = load_imagereward(device)

    img_paths = _sorted_generated_images(image_dir)
    if not img_paths:
        raise RuntimeError(f"no generated images in {image_dir}")

    vals: list[float] = []
    for p in tqdm(img_paths, desc="ImageReward scoring"):
        lab = int(p.stem.split("_")[2])
        prompt = f"a photo of a {labels_1k[lab]}"
        img = Image.open(p).convert("RGB")
        vals.append(float(rm.score(prompt, img)))
    reward_t = torch.tensor(vals, dtype=torch.float32)
    torch.save({"reward": reward_t}, reward_pt)
    return reward_t


def _assemble_repr_tensor(shared_repr_pt: Path, repr_key: str, pooling_mode: str) -> torch.Tensor:
    obj = torch.load(shared_repr_pt, map_location="cpu", weights_only=False)
    batches = obj.get("batches")
    if not isinstance(batches, list) or not batches:
        raise RuntimeError(f"{shared_repr_pt}: missing batches")
    chunks: list[torch.Tensor] = []
    for rec in batches:
        repr_d = rec.get("repr", {})
        if repr_key not in repr_d:
            raise KeyError(f"{shared_repr_pt}: repr_key {repr_key!r} not found; keys={list(repr_d.keys())[:8]}")
        t = repr_d[repr_key][pooling_mode]
        chunks.append(t.float().cpu())
    return torch.cat(chunks, dim=0)


def _ensure_case_image_link(case_image_dir: Path, shared_image_dir: Path) -> None:
    if case_image_dir.exists():
        return
    try:
        case_image_dir.symlink_to(shared_image_dir, target_is_directory=True)
    except Exception:
        case_image_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Single-rollout ranker data builder for (layer_ratio, step_ratio) grid.")
    p.add_argument("--layer_ratios", type=str, default="0.1,0.3,0.6,0.9,1.0")
    p.add_argument("--step_ratios", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--base_output_dir", type=str, default="/root/autodl-tmp/ranker_mid_training_data_mc")
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--num_samples", type=int, default=64_000)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--pooling_mode", type=str, default="mean", choices=("mean", "last"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    layer_ratios = parse_float_list(args.layer_ratios) or LAYER_RATIO_LS
    step_ratios = parse_float_list(args.step_ratios) or STEP_RATIO_LS
    set_seed(args.seed)

    base_output_dir = Path(args.base_output_dir).expanduser().resolve()
    shared_dir = base_output_dir / "shared"
    shared_image_dir = shared_dir / "images"
    shared_repr_pt = shared_dir / "repr_batches.pt"
    shared_reward_pt = shared_dir / "imagereward.pt"
    shared_dir.mkdir(parents=True, exist_ok=True)

    _run_shared_rollout(
        model_name=args.model_name,
        num_samples=int(args.num_samples),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        dtype=str(args.dtype),
        layer_ratios=layer_ratios,
        step_ratios=step_ratios,
        shared_image_dir=shared_image_dir,
        shared_repr_pt=shared_repr_pt,
        overwrite=bool(args.overwrite),
    )
    reward_t = _compute_imagereward_tensor(shared_image_dir, shared_reward_pt, overwrite=bool(args.overwrite))

    case_map = _build_case_map(layer_ratios, step_ratios, args.num_inference_steps)
    summary: list[dict] = []
    for i, case in enumerate(case_map, start=1):
        case_tag = case["case_tag"]
        repr_key = case["repr_key"]
        case_dir = base_output_dir / case_tag
        case_dir.mkdir(parents=True, exist_ok=True)
        repr_pt = case_dir / "repr_reward.pt"
        reward_json = case_dir / "reward_ls.json"
        case_image_dir = case_dir / "images"
        _ensure_case_image_link(case_image_dir, shared_image_dir)

        if repr_pt.is_file() and reward_json.is_file() and (not args.overwrite):
            print(f"[{i}/{len(case_map)}] [skip] {case_tag}")
            payload = torch.load(repr_pt, map_location="cpu", weights_only=False)
            summary.append(
                {
                    "case_tag": case_tag,
                    "layer_ratio": case["layer_ratio"],
                    "step_ratio": case["step_ratio"],
                    "repr_key": payload.get("repr_key"),
                    "dir": str(case_dir),
                    "repr_pt": str(repr_pt),
                    "reward_json": str(reward_json),
                }
            )
            continue

        print(f"[{i}/{len(case_map)}] [build] {case_tag} repr_key={repr_key}")
        tensor = _assemble_repr_tensor(shared_repr_pt, repr_key=repr_key, pooling_mode=args.pooling_mode)
        if tensor.shape[0] != reward_t.shape[0]:
            raise RuntimeError(
                f"{case_tag}: tensor rows {tensor.shape[0]} != reward rows {reward_t.shape[0]}"
            )
        torch.save(
            {
                "tensor": tensor,
                "reward": reward_t,
                "repr_key": repr_key,
                "layer_ratio": float(case["layer_ratio"]),
                "step_ratio": float(case["step_ratio"]),
                "pooling_mode": str(args.pooling_mode),
                "shared_image_dir": str(shared_image_dir),
                "shared_repr_pt": str(shared_repr_pt),
            },
            repr_pt,
        )
        reward_json.write_text(json.dumps([float(x) for x in reward_t.tolist()], ensure_ascii=False), encoding="utf-8")
        case_meta = {
            "case_tag": case_tag,
            "layer_ratio": float(case["layer_ratio"]),
            "step_ratio": float(case["step_ratio"]),
            "repr_key": repr_key,
            "dir": str(case_dir),
            "image_dir": str(case_image_dir),
            "repr_pt": str(repr_pt),
            "reward_json": str(reward_json),
            "tensor_shape": list(tensor.shape),
            "reward_shape": list(reward_t.shape),
            "shared_image_dir": str(shared_image_dir),
            "shared_repr_pt": str(shared_repr_pt),
        }
        (case_dir / "meta.json").write_text(json.dumps(case_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        summary.append(case_meta)

    (base_output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. summary -> {base_output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

