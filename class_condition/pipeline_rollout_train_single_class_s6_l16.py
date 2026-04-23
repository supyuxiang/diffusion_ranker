#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from class_condition.reward import (
    _build_inception_logits_extractor,
    labels_1k,
    load_clip,
    load_imagereward,
)
from utils import compute_fid_is_from_dirs, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Rollout 3k/class at s6_l16, train rankers, RS-generate 50/class, then evaluate 50k metrics"
    )
    p.add_argument("--start_class", type=int, default=0)
    p.add_argument("--end_class", type=int, default=999)

    # rollout
    p.add_argument("--num_samples", type=int, default=3000)
    p.add_argument("--rollout_batch_size", type=int, default=256)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--mc", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # fixed per request
    p.add_argument("--step_idx", type=int, default=6)
    p.add_argument("--layer_idx", type=int, default=16)

    # train overrides (kept same defaults as existing training config)
    p.add_argument("--train_num_epochs", type=int, default=100)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--train_num_workers", type=int, default=4)
    p.add_argument("--train_lr", type=float, default=1e-3)
    p.add_argument("--train_alpha", type=float, default=0.1)
    p.add_argument("--train_parallel_size", type=int, default=8)
    p.add_argument("--train_num_bootstrap", type=int, default=8)

    # generation via trained ranker
    p.add_argument("--gen_per_class_samples", type=int, default=50)
    p.add_argument("--gen_parallel_size", type=int, default=8)
    p.add_argument("--gen_batch_size", type=int, default=2)

    p.add_argument(
        "--rollout_save_root",
        type=str,
        default="/root/autodl-tmp/class_condition/single_class",
        help="Where rollout hidden/reward .pt files are written",
    )
    p.add_argument(
        "--ranker_root",
        type=str,
        default="/root/autodl-tmp/ranker_ckpts/single_class_s6_l16",
        help="Where class-wise ranker ckpts are written",
    )
    p.add_argument(
        "--images_root",
        type=str,
        default="/root/autodl-tmp/class_condition/single_class_s6_l16_rs50_images",
        help="Where class-wise generated images are written",
    )
    p.add_argument(
        "--merged_output_dir",
        type=str,
        default="/root/autodl-tmp/class_condition/single_class_s6_l16_rs50_merged",
        help="Where merged 50k images are written",
    )
    p.add_argument(
        "--metrics_output_json",
        type=str,
        default="/root/autodl-tmp/class_condition/single_class_s6_l16_rs50_metrics.json",
        help="Where final metrics json is written",
    )
    p.add_argument(
        "--real_dir",
        type=str,
        default="/root/autodl-tmp/imagenet/val",
        help="ImageNet val directory for FID/IS",
    )

    p.add_argument("--force_rerollout", action="store_true")
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--force_regen", action="store_true")
    p.add_argument("--force_remerge", action="store_true")
    p.add_argument("--force_reeval", action="store_true")
    p.add_argument("--skip_rollout", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    return p.parse_args()


def _run_cmd(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _count_pngs(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.glob("*.png"))


def _rollout_target_path(rollout_save_root: Path, class_idx: int, step_idx: int, layer_idx: int) -> Path:
    return rollout_save_root / (
        f"dit_imagenet_hiddens_and_is_rewards_s1_6_9_l16_class{class_idx}_s{step_idx}_l{layer_idx}.pt"
    )


def rollout_one_class(args: argparse.Namespace, class_idx: int, rollout_save_root: Path) -> Path:
    target_pt = _rollout_target_path(rollout_save_root, class_idx, args.step_idx, args.layer_idx)
    if target_pt.is_file() and not args.force_rerollout:
        print(f"[rollout] class={class_idx} skip (exists)")
        return target_pt

    save_path_prefix = rollout_save_root / f"dit_imagenet_hiddens_and_is_rewards_s1_6_9_l16_class{class_idx}.pt"
    rollout_script = ROOT / "class_condition" / "dit_rollout_mc_is_single.py"
    cmd = [
        sys.executable,
        str(rollout_script),
        "--save_path",
        str(save_path_prefix),
        "--class_idx",
        str(class_idx),
        "--num_samples",
        str(int(args.num_samples)),
        "--batch_size",
        str(int(args.rollout_batch_size)),
        "--num_inference_steps",
        str(int(args.num_inference_steps)),
        "--guidance_scale",
        str(float(args.guidance_scale)),
        "--seed",
        str(int(args.seed) + int(class_idx)),
        "--dtype",
        str(args.dtype),
        "--target_step_indices",
        str(int(args.step_idx)),
        "--target_layer_idx",
        str(int(args.layer_idx)),
        "--mc",
        str(int(args.mc)),
        "--hidden_pool",
        "mean",
    ]
    _run_cmd(cmd, cwd=ROOT / "class_condition")
    if not target_pt.is_file():
        raise FileNotFoundError(f"rollout finished but missing target file: {target_pt}")
    return target_pt


def train_one_class(args: argparse.Namespace, class_idx: int, ranker_root: Path) -> Path:
    class_dir = ranker_root / f"class_{class_idx:04d}"
    best_ckpt = class_dir / "single_class_ranker_best.pth"
    if best_ckpt.is_file() and not args.force_retrain:
        print(f"[train] class={class_idx} skip (exists)")
        return best_ckpt

    class_dir.mkdir(parents=True, exist_ok=True)
    train_script = ROOT / "class_condition" / "train_ranker_single_class.py"
    cmd = [
        sys.executable,
        str(train_script),
        f"class_idx={class_idx}",
        f"step_idx={int(args.step_idx)}",
        f"layer_idx={int(args.layer_idx)}",
        f"batch_size={int(args.train_batch_size)}",
        f"num_workers={int(args.train_num_workers)}",
        f"lr={float(args.train_lr)}",
        f"alpha={float(args.train_alpha)}",
        f"parallel_size={int(args.train_parallel_size)}",
        f"num_bootstrap={int(args.train_num_bootstrap)}",
        f"num_epochs={int(args.train_num_epochs)}",
        "save_interval=1",
        "eval_interval=1",
        f"seed={int(args.seed)}",
        f"output_dir={str(class_dir)}",
        "run_name=single_class_ranker",
    ]
    env = dict(os.environ)
    env["SINGLE_CLASS_DATA_ROOT"] = str(Path(args.rollout_save_root).expanduser().resolve())
    _run_cmd(cmd, cwd=ROOT / "class_condition", env=env)
    if not best_ckpt.is_file():
        raise FileNotFoundError(f"train finished but missing best ckpt: {best_ckpt}")
    return best_ckpt


def _class_image_dir(images_root: Path, class_idx: int) -> Path:
    return images_root / f"class_{class_idx:04d}"


def generate_one_class(
    args: argparse.Namespace,
    class_idx: int,
    ranker_ckpt: Path,
    images_root: Path,
) -> Path:
    class_out_dir = _class_image_dir(images_root, class_idx)
    best_dir = class_out_dir / "best_images"
    if _count_pngs(best_dir) >= int(args.gen_per_class_samples) and not args.force_regen:
        print(f"[generate] class={class_idx} skip (exists)")
        return best_dir

    rs_script = ROOT / "class_condition" / "dit_random_search_ranker_single_class.py"
    class_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(rs_script),
        "--ranker_ckpt",
        str(ranker_ckpt),
        "--output_dir",
        str(class_out_dir),
        "--model_name",
        "dit-xl-2-256",
        "--dtype",
        str(args.dtype),
        "--num_samples",
        str(int(args.gen_per_class_samples)),
        "--batch_size",
        str(int(args.gen_batch_size)),
        "--parallel_size",
        str(int(args.gen_parallel_size)),
        "--num_inference_steps",
        str(int(args.num_inference_steps)),
        "--guidance_scale",
        str(float(args.guidance_scale)),
        "--seed",
        str(int(args.seed) + int(class_idx)),
        "--class_idx",
        str(int(class_idx)),
        "--target_step_idx",
        str(int(args.step_idx)),
        "--target_layer_idx",
        str(int(args.layer_idx)),
        "--hidden_pool",
        "mean",
    ]
    _run_cmd(cmd, cwd=ROOT / "class_condition")
    got = _count_pngs(best_dir)
    if got < int(args.gen_per_class_samples):
        raise RuntimeError(
            f"class={class_idx} generated {got} images, expected {int(args.gen_per_class_samples)}"
        )
    return best_dir


def merge_all_images(args: argparse.Namespace, class_list: list[int], images_root: Path, merged_dir: Path) -> Path:
    expected = len(class_list) * int(args.gen_per_class_samples)
    if merged_dir.is_dir() and _count_pngs(merged_dir) == expected and not args.force_remerge:
        print(f"[merge] skip (already {expected} images)")
        return merged_dir

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    global_idx = 0
    for cid in tqdm(class_list, desc="merge generated images"):
        best_dir = _class_image_dir(images_root, cid) / "best_images"
        paths = sorted(best_dir.glob("*.png"))
        if len(paths) < int(args.gen_per_class_samples):
            raise RuntimeError(
                f"class={cid} has {len(paths)} images, expected >= {int(args.gen_per_class_samples)}"
            )
        for src in paths[: int(args.gen_per_class_samples)]:
            dst = merged_dir / f"image_{global_idx:05d}_{cid:05d}.png"
            shutil.copy2(src, dst)
            global_idx += 1

    if global_idx != expected:
        raise RuntimeError(f"merged {global_idx} images, expected {expected}")
    return merged_dir


@torch.inference_mode()
def _compute_clipscore_from_dir(image_dir: Path, device: torch.device, batch_size: int = 64) -> dict[str, float]:
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"no png images found in {image_dir}")

    clip_model, clip_processor = load_clip(device)
    scores: list[float] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="eval clipscore"):
        chunk = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB").copy() for p in chunk]
        labels = [int(p.stem.split("_")[-1]) for p in chunk]
        prompts = [f"A photo of a {labels_1k[lab]}." for lab in labels]
        inputs = clip_processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = clip_model(**inputs)
        img_f = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_f = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (img_f * txt_f).sum(dim=-1)
        scores.extend(sim.detach().float().cpu().tolist())

    arr = np.asarray(scores, dtype=np.float64)
    return {"clipscore_mean": float(arr.mean()), "clipscore_std": float(arr.std(ddof=0))}


@torch.inference_mode()
def _compute_imagereward_from_dir(image_dir: Path, device: torch.device) -> dict[str, float]:
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"no png images found in {image_dir}")

    rm_model = load_imagereward(device)
    scores: list[float] = []
    for p in tqdm(paths, desc="eval imagereward"):
        lab = int(p.stem.split("_")[-1])
        prompt = f"A photo of a {labels_1k[lab]}."
        img = Image.open(p).convert("RGB").copy()
        scores.append(float(rm_model.score(prompt, img)))

    arr = np.asarray(scores, dtype=np.float64)
    return {"imagereward_mean": float(arr.mean()), "imagereward_std": float(arr.std(ddof=0))}


@torch.inference_mode()
def _compute_inception_confidence_from_dir(
    image_dir: Path,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    from torch_fidelity.datasets import TransformPILtoRGBTensor
    from torchvision.transforms import CenterCrop, Compose, Resize

    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"no png images found in {image_dir}")
    feat_extractor = _build_inception_logits_extractor(device)
    transform = Compose([Resize(256), CenterCrop(256), TransformPILtoRGBTensor()])

    scores: list[float] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="eval inception confidence"):
        chunk = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB").copy() for p in chunk]
        tensors = torch.stack([transform(img) for img in images], dim=0).to(
            device, non_blocking=(device.type == "cuda")
        )
        logits = feat_extractor(tensors)[0]
        probs = F.softmax(logits.float(), dim=-1)
        vals = probs.max(dim=-1).values
        scores.extend(vals.detach().cpu().tolist())
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "inception_confidence_mean": float(arr.mean()),
        "inception_confidence_std": float(arr.std(ddof=0)),
    }


def evaluate_and_save(
    args: argparse.Namespace,
    merged_dir: Path,
    metrics_output_json: Path,
) -> dict:
    if metrics_output_json.is_file() and not args.force_reeval:
        print(f"[eval] skip (exists): {metrics_output_json}")
        return json.loads(metrics_output_json.read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_dir = Path(args.real_dir).expanduser().resolve()
    fid_is = compute_fid_is_from_dirs(
        merged_dir,
        real_dir,
        device=device,
        batch_size=64,
        samples_resize_and_crop=256,
        samples_find_deep=True,
    )
    clip_metrics = _compute_clipscore_from_dir(merged_dir, device=device, batch_size=64)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ir_metrics = _compute_imagereward_from_dir(merged_dir, device=device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    inc_metrics = _compute_inception_confidence_from_dir(merged_dir, device=device, batch_size=64)

    out = {
        "num_images": _count_pngs(merged_dir),
        "FID": float(fid_is["FID"]),
        "IS": float(fid_is["IS"]),
        "IS_STD": float(fid_is["IS_STD"]),
        **clip_metrics,
        **ir_metrics,
        **inc_metrics,
        "merged_dir": str(merged_dir),
        "real_dir": str(real_dir),
    }
    metrics_output_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    if args.start_class < 0 or args.end_class > 999 or args.start_class > args.end_class:
        raise ValueError("Require 0 <= start_class <= end_class <= 999")

    set_seed(int(args.seed))
    class_list = list(range(int(args.start_class), int(args.end_class) + 1))
    rollout_save_root = Path(args.rollout_save_root).expanduser().resolve()
    ranker_root = Path(args.ranker_root).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve()
    merged_output_dir = Path(args.merged_output_dir).expanduser().resolve()
    metrics_output_json = Path(args.metrics_output_json).expanduser().resolve()

    rollout_save_root.mkdir(parents=True, exist_ok=True)
    ranker_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[pipeline] classes={args.start_class}-{args.end_class}, "
        f"rollout num_samples/class={args.num_samples}, step={args.step_idx}, layer={args.layer_idx}, "
        f"gen_per_class={args.gen_per_class_samples}"
    )

    for class_idx in tqdm(class_list, desc="rollout+train single-class"):
        ranker_ckpt = ranker_root / f"class_{class_idx:04d}" / "single_class_ranker_best.pth"
        if not args.skip_rollout:
            rollout_one_class(args, class_idx, rollout_save_root)
        if not args.skip_train:
            ranker_ckpt = train_one_class(args, class_idx, ranker_root)
        if not ranker_ckpt.is_file():
            raise FileNotFoundError(f"missing ranker ckpt for class={class_idx}: {ranker_ckpt}")
        if not args.skip_generate:
            generate_one_class(args, class_idx, ranker_ckpt, images_root)

    merged_dir = merge_all_images(args, class_list, images_root, merged_output_dir)
    if args.skip_eval:
        print(f"[done] merged images at: {merged_dir}")
        print(f"[done] rollout root: {rollout_save_root}")
        print(f"[done] ranker root: {ranker_root}")
        print(f"[done] images root: {images_root}")
        return

    metrics = evaluate_and_save(args, merged_dir, metrics_output_json)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("[done] full pipeline finished")
    print(f"[done] rollout root: {rollout_save_root}")
    print(f"[done] ranker root: {ranker_root}")
    print(f"[done] images root: {images_root}")
    print(f"[done] merged dir: {merged_dir}")
    print(f"[done] metrics json: {metrics_output_json}")


if __name__ == "__main__":
    main()
