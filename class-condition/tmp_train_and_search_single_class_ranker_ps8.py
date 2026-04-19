#!/usr/bin/env python3
"""
Temporary helper script:
1. Train a ranker from a single-class rollout file
2. Run fixed-class online random search with ps=8
3. Compare ImageReward mean against the rollout baseline

Intended for one-off experiments and safe to delete afterwards.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
_SEARCH = _ROOT / "search"
for p in (_ROOT, _SEARCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from search.dit_random_search_ranker import load_ranker_from_ckpt
from search.reward import labels_1k, load_imagereward, ranker_mean_logit_score
from utils import DitRolloutReprCapture, dit_loader, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Temporary single-class train + ps8 ranker random search + ImageReward comparison."
    )
    p.add_argument(
        "--rollout_pt",
        type=str,
        default="/root/autodl-tmp/dit_imagenet_singleclass123_hiddens_and_rewards_mc_s1_6_9_l16_per128.pth",
    )
    p.add_argument("--step_idx", type=int, default=6)
    p.add_argument("--layer_idx", type=int, default=16)
    p.add_argument("--pooling_mode", type=str, default="mean", choices=("mean", "last"))
    p.add_argument("--ranker_type", type=str, default="listwise", choices=("pointwise", "listwise", "cross_attention"))
    p.add_argument(
        "--prompt_repr_pt",
        type=str,
        default="/root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt",
    )
    p.add_argument("--train_num_epochs", type=int, default=500)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--num_cand", type=int, default=8)
    p.add_argument("--num_candidate_groups", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_min_ratio", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--search_num_samples", type=int, default=-1)
    p.add_argument("--search_batch_size", type=int, default=32)
    p.add_argument("--parallel_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--dit_path", type=str, default="/root/autodl-tmp/DiT-XL-2-256")
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--cuda_visible_devices", type=str, default="0")
    p.add_argument("--extra_train_args", type=str, default="")
    return p.parse_args()


def load_prompt_table(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    t = obj["tensor"] if isinstance(obj, dict) and "tensor" in obj else obj
    if not isinstance(t, torch.Tensor) or t.dim() != 2:
        raise ValueError(
            f"prompt table must be a 2D Tensor, got {type(t)} shape={getattr(t, 'shape', None)}"
        )
    return t.cpu()


def _rollout_candidate_with_fixed_class(
    pipe,
    repr_capture: DitRolloutReprCapture,
    *,
    class_id: int,
    num_steps: int,
    n_this: int,
    device: torch.device,
    generator_seed: int,
    guidance_scale: float,
) -> tuple[list, list[int], dict]:
    labels = [int(class_id)] * int(n_this)
    gen = torch.Generator(device=device).manual_seed(int(generator_seed))
    repr_capture.set_capturing(True)
    repr_capture.begin_batch(labels)
    out = pipe(
        class_labels=labels,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        generator=gen,
    )
    repr_capture.end_batch(0, 0, generator_seed=int(generator_seed))
    rec = repr_capture.records.pop()
    repr_dict = rec["repr"]
    images = [im.copy() for im in out.images]
    del out, gen
    return images, labels, repr_dict


def _evaluate_imagereward_for_dir(image_dir: Path, class_id: int, device: torch.device) -> dict[str, float]:
    rm = load_imagereward(device)
    prompt = f"A photo of a {labels_1k[int(class_id)]}."
    paths = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem.split("_")[1]))
    vals: list[float] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        vals.append(float(rm.score(prompt, img)))
    t = torch.tensor(vals, dtype=torch.float32)
    return {
        "num_images": float(len(paths)),
        "imagereward_mean": float(t.mean().item()),
        "imagereward_std": float(t.std(unbiased=False).item()),
    }


@torch.inference_mode()
def run_ps8_search(
    *,
    ranker,
    prompt_table: torch.Tensor,
    repr_key: str,
    class_id: int,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    num_steps: int,
    device: torch.device,
    seed: int,
    guidance_scale: float,
    dit_path: Path,
    dtype: str,
    layer_ratio: float,
    step_ratio: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_counter = 0
    num_rounds = (num_samples + batch_size - 1) // batch_size

    pipe = dit_loader(dit_path, device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    repr_capture = DitRolloutReprCapture(
        pipe,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        layer_ratios=(float(layer_ratio),),
        step_ratio_ls=[float(step_ratio)],
        store_dtype=torch.float16,
        rollout_base_seed=int(seed),
    )
    repr_capture.install()

    try:
        for _ in range(num_rounds):
            n_this = min(batch_size, num_samples - img_counter)
            if n_this <= 0:
                break

            best_score = float("-inf")
            best_images = None
            best_labels = None

            for _k in range(int(num_candidates)):
                cand_seed = int(rng.integers(0, 2**31 - 1))
                images, labels, repr_dict = _rollout_candidate_with_fixed_class(
                    pipe,
                    repr_capture,
                    class_id=class_id,
                    num_steps=num_steps,
                    n_this=n_this,
                    device=device,
                    generator_seed=cand_seed,
                    guidance_scale=guidance_scale,
                )
                score = ranker_mean_logit_score(
                    ranker=ranker,
                    repr_dict=repr_dict,
                    repr_key=repr_key,
                    labels=labels,
                    prompt_table=prompt_table,
                    device=device,
                )
                if score > best_score:
                    best_score = score
                    best_images = images
                    best_labels = labels

            assert best_images is not None and best_labels is not None
            for img, lab in zip(best_images, best_labels):
                if img_counter >= num_samples:
                    break
                img.save(output_dir / f"image_{img_counter:05d}_{int(lab):05d}.png")
                img_counter += 1
    finally:
        repr_capture.remove()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    rollout_pt = Path(args.rollout_pt).expanduser().resolve()
    if not rollout_pt.is_file():
        raise SystemExit(f"missing rollout pt: {rollout_pt}")

    obj = torch.load(rollout_pt, map_location="cpu", weights_only=False)
    meta = obj.get("meta", {})
    reward_store = obj.get("reward_store")
    label_store = obj.get("label_store")
    if not isinstance(reward_store, dict) or not isinstance(label_store, torch.Tensor):
        raise SystemExit(f"{rollout_pt}: invalid rollout file")

    class_id = int(meta.get("class_id", int(label_store[0].item())))
    if not torch.all(label_store == class_id):
        raise SystemExit("this helper expects a single-class rollout file")

    step_key = f"s{int(args.step_idx)}"
    if step_key not in reward_store:
        raise SystemExit(f"{rollout_pt}: missing reward key {step_key}")
    baseline_reward = reward_store[step_key].float().view(-1)
    baseline_mean = float(baseline_reward.mean().item())
    baseline_std = float(baseline_reward.std(unbiased=False).item())

    num_search_samples = int(args.search_num_samples)
    if num_search_samples <= 0:
        num_search_samples = int(meta.get("num_samples", int(label_store.numel())))

    layer_ratio = float(meta.get("layer_index_ratio", 0.6))
    deno_steps = int(meta.get("num_inference_steps", 10))
    guidance_scale = float(meta.get("guidance_scale", 1.0))
    step_ratio = float(int(args.step_idx)) / float(max(1, deno_steps - 1))
    repr_key = f"s{int(args.step_idx)}_r{layer_ratio:g}"

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else Path(f"/root/autodl-tmp/tmp_singleclass{class_id}_ranker_ps{int(args.parallel_size)}_s{int(args.step_idx)}")
    )
    train_out_dir = output_root / "train"
    search_out_dir = output_root / "search_ps8"
    output_root.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "/root/diffusion_ranker/class-condition/train_from_rollout_mc_imagenet.py",
        "--rollout_pt",
        str(rollout_pt),
        "--step_idx",
        str(args.step_idx),
        "--layer_idx",
        str(args.layer_idx),
        "--pooling_mode",
        str(args.pooling_mode),
        "--ranker_type",
        str(args.ranker_type),
        "--prompt_repr_pt",
        str(Path(args.prompt_repr_pt).expanduser().resolve()),
        "--output_dir",
        str(train_out_dir),
        "--num_epochs",
        str(args.train_num_epochs),
        "--batch_size",
        str(args.train_batch_size),
        "--num_cand",
        str(args.num_cand),
        "--num_candidate_groups",
        str(args.num_candidate_groups),
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_attention_heads",
        str(args.num_attention_heads),
        "--num_layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--optimizer",
        str(args.optimizer),
        "--lr_scheduler",
        str(args.lr_scheduler),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--lr_min_ratio",
        str(args.lr_min_ratio),
        "--alpha",
        str(args.alpha),
        "--tau",
        str(args.tau),
        "--seed",
        str(args.seed),
        "--save_freq",
        "1",
        "--cuda_visible_devices",
        str(args.cuda_visible_devices),
    ]
    if args.extra_train_args.strip():
        train_cmd.extend(shlex.split(args.extra_train_args.strip()))

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    print("Training command:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True, env=env)

    ckpt_path = train_out_dir / f"ranker_epoch{int(args.train_num_epochs)}.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"missing trained checkpoint: {ckpt_path}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranker = load_ranker_from_ckpt(
        ckpt_path,
        device,
        num_attention_heads=int(args.num_attention_heads),
        dropout=float(args.dropout),
    )
    prompt_table = load_prompt_table(Path(args.prompt_repr_pt).expanduser().resolve())

    run_ps8_search(
        ranker=ranker,
        prompt_table=prompt_table,
        repr_key=repr_key,
        class_id=class_id,
        output_dir=search_out_dir,
        num_samples=num_search_samples,
        batch_size=int(args.search_batch_size),
        num_candidates=int(args.parallel_size),
        num_steps=deno_steps,
        device=device,
        seed=int(args.seed),
        guidance_scale=guidance_scale,
        dit_path=Path(args.dit_path).expanduser().resolve(),
        dtype=str(args.dtype),
        layer_ratio=layer_ratio,
        step_ratio=step_ratio,
    )

    search_metrics = _evaluate_imagereward_for_dir(search_out_dir, class_id, device)
    summary = {
        "rollout_pt": str(rollout_pt),
        "class_id": int(class_id),
        "class_name": str(labels_1k[int(class_id)]),
        "step_idx": int(args.step_idx),
        "layer_idx": int(args.layer_idx),
        "repr_key": repr_key,
        "ranker_ckpt": str(ckpt_path),
        "search_output_dir": str(search_out_dir),
        "baseline_rollout_imagereward_mean": float(baseline_mean),
        "baseline_rollout_imagereward_std": float(baseline_std),
        "search_imagereward_mean": float(search_metrics["imagereward_mean"]),
        "search_imagereward_std": float(search_metrics["imagereward_std"]),
        "imagereward_delta": float(search_metrics["imagereward_mean"] - baseline_mean),
        "num_search_images": int(search_metrics["num_images"]),
        "parallel_size": int(args.parallel_size),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
