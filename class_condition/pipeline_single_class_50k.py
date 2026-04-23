#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from class_condition.reward import labels_1k, load_clip, load_imagereward
from utils import compute_fid_is_from_dirs, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train 1000 single-class rankers, sample 50k, then evaluate metrics")
    p.add_argument("--run_root", type=str, default="/root/autodl-tmp/class_condition/single_class_50k_pipeline")
    p.add_argument("--real_dir", type=str, default="/root/autodl-tmp/imagenet/val")
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--start_class", type=int, default=0)
    p.add_argument("--end_class", type=int, default=999)
    p.add_argument("--per_class_samples", type=int, default=50)
    p.add_argument("--parallel_size", type=int, default=8)
    p.add_argument("--search_batch_size", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--step_idx", type=int, default=9)
    p.add_argument("--layer_idx", type=int, default=16)
    p.add_argument("--hidden_pool", type=str, default="mean", choices=("mean", "last"))

    p.add_argument("--train_num_epochs", type=int, default=100)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--train_num_workers", type=int, default=4)
    p.add_argument("--train_lr", type=float, default=1e-3)
    p.add_argument("--train_alpha", type=float, default=0.1)
    p.add_argument("--train_parallel_size", type=int, default=8)
    p.add_argument("--train_num_bootstrap", type=int, default=8)

    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--force_regen", action="store_true")
    p.add_argument("--force_remerge", action="store_true")
    p.add_argument("--force_reeval", action="store_true")
    return p.parse_args()


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _count_pngs(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.glob("*.png"))


def _class_dirs(run_root: Path, class_idx: int) -> tuple[Path, Path]:
    ranker_dir = run_root / "rankers" / f"class_{class_idx:04d}"
    sample_dir = run_root / "images_by_class" / f"class_{class_idx:04d}"
    return ranker_dir, sample_dir


def _train_single_class(args: argparse.Namespace, run_root: Path, class_idx: int) -> Path:
    ranker_dir, _ = _class_dirs(run_root, class_idx)
    ckpt = ranker_dir / "single_class_ranker_best.pth"
    if ckpt.is_file() and not args.force_retrain:
        print(f"[train] class={class_idx} skip (ckpt exists)")
        return ckpt

    ranker_dir.mkdir(parents=True, exist_ok=True)
    train_script = ROOT / "class_condition" / "train_ranker_single_class.py"
    train_cmd = [
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
        f"output_dir={str(ranker_dir)}",
        "run_name=single_class_ranker",
    ]
    _run_cmd(train_cmd, cwd=ROOT / "class_condition")
    if not ckpt.is_file():
        raise FileNotFoundError(f"train finished but ckpt missing: {ckpt}")
    return ckpt


def _generate_single_class(
    args: argparse.Namespace,
    run_root: Path,
    class_idx: int,
    ranker_ckpt: Path,
) -> Path:
    _, sample_dir = _class_dirs(run_root, class_idx)
    best_dir = sample_dir / "best_images"
    if _count_pngs(best_dir) >= int(args.per_class_samples) and not args.force_regen:
        print(f"[sample] class={class_idx} skip (images exist)")
        return best_dir

    rs_script = ROOT / "class_condition" / "dit_random_search_ranker_single_class.py"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_cmd = [
        sys.executable,
        str(rs_script),
        "--ranker_ckpt",
        str(ranker_ckpt),
        "--output_dir",
        str(sample_dir),
        "--model_name",
        str(args.model_name),
        "--dtype",
        str(args.dtype),
        "--num_samples",
        str(int(args.per_class_samples)),
        "--batch_size",
        str(int(args.search_batch_size)),
        "--parallel_size",
        str(int(args.parallel_size)),
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
        str(args.hidden_pool),
    ]
    _run_cmd(sample_cmd, cwd=ROOT / "class_condition")
    got = _count_pngs(best_dir)
    if got < int(args.per_class_samples):
        raise RuntimeError(
            f"class={class_idx} generated {got} images, expected {int(args.per_class_samples)}"
        )
    return best_dir


def _merge_all_images(args: argparse.Namespace, run_root: Path, class_list: list[int]) -> Path:
    merged_dir = run_root / "images_50k_all"
    expected = len(class_list) * int(args.per_class_samples)
    if merged_dir.is_dir() and _count_pngs(merged_dir) == expected and not args.force_remerge:
        print(f"[merge] skip (already {expected} images)")
        return merged_dir

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    global_idx = 0
    for cid in tqdm(class_list, desc="merge class images"):
        _, sample_dir = _class_dirs(run_root, cid)
        best_dir = sample_dir / "best_images"
        paths = sorted(best_dir.glob("*.png"))
        if len(paths) < int(args.per_class_samples):
            raise RuntimeError(
                f"class={cid} has {len(paths)} images, expected >= {int(args.per_class_samples)}"
            )
        for src in paths[: int(args.per_class_samples)]:
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
    for i in tqdm(range(0, len(paths), batch_size), desc="clipscore"):
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
        del inputs, out, img_f, txt_f, sim, images

    arr = np.asarray(scores, dtype=np.float64)
    return {
        "clipscore_mean": float(arr.mean()),
        "clipscore_std": float(arr.std(ddof=0)),
    }


@torch.inference_mode()
def _compute_imagereward_from_dir(image_dir: Path, device: torch.device) -> dict[str, float]:
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"no png images found in {image_dir}")

    rm_model = load_imagereward(device)
    scores: list[float] = []
    for p in tqdm(paths, desc="imagereward"):
        lab = int(p.stem.split("_")[-1])
        prompt = f"A photo of a {labels_1k[lab]}."
        img = Image.open(p).convert("RGB").copy()
        scores.append(float(rm_model.score(prompt, img)))

    arr = np.asarray(scores, dtype=np.float64)
    return {
        "imagereward_mean": float(arr.mean()),
        "imagereward_std": float(arr.std(ddof=0)),
    }


def _evaluate_all(args: argparse.Namespace, run_root: Path, merged_dir: Path) -> dict:
    metrics_path = run_root / "metrics" / "final_metrics.json"
    if metrics_path.is_file() and not args.force_reeval:
        print(f"[eval] skip (exists): {metrics_path}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
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

    out = {
        "num_images": _count_pngs(merged_dir),
        "FID": float(fid_is["FID"]),
        "IS": float(fid_is["IS"]),
        "IS_STD": float(fid_is["IS_STD"]),
        **clip_metrics,
        **ir_metrics,
        "merged_dir": str(merged_dir),
        "real_dir": str(real_dir),
    }
    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    if args.start_class < 0 or args.end_class > 999 or args.start_class > args.end_class:
        raise ValueError("class range must satisfy 0 <= start_class <= end_class <= 999")
    class_list = list(range(int(args.start_class), int(args.end_class) + 1))

    set_seed(int(args.seed))
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[pipeline] classes={args.start_class}-{args.end_class} ({len(class_list)}) "
        f"per_class={args.per_class_samples} total={len(class_list) * int(args.per_class_samples)}"
    )

    for cid in tqdm(class_list, desc="class pipeline"):
        ckpt = _class_dirs(run_root, cid)[0] / "single_class_ranker_best.pth"
        if not args.skip_train:
            ckpt = _train_single_class(args, run_root, cid)
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing ckpt for class={cid}: {ckpt}")
        if not args.skip_generate:
            _generate_single_class(args, run_root, cid, ckpt)

    merged_dir = _merge_all_images(args, run_root, class_list)
    if args.skip_eval:
        print(f"[done] merged images at: {merged_dir}")
        return

    metrics = _evaluate_all(args, run_root, merged_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
