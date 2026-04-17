#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uncondition_mnist.mnist_classifier_reward import load_mnist_classifier

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_RESIZE_HW = 224
_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate MNIST generated images recognizability")
    p.add_argument("--gen_dir", type=str, required=True, help="Directory containing generated images")
    p.add_argument(
        "--classifier_ckpt",
        type=str,
        default="/root/autodl-tmp/mnist_classifier/resnet18/classifier_best.pt",
    )
    p.add_argument("--threshold", type=float, default=0.9, help="High-confidence threshold on max softmax")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional output json path. Defaults to <gen_dir>/recognizability_metrics.json",
    )
    p.add_argument(
        "--save_per_image_jsonl",
        action="store_true",
        help="If set, save per-image prediction/confidence to <gen_dir>/recognizability_per_image.jsonl",
    )
    return p.parse_args()


def _collect_image_paths(gen_dir: Path) -> list[Path]:
    return sorted([p for p in gen_dir.iterdir() if p.is_file() and p.suffix.lower() in _EXTS])


def _build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((_RESIZE_HW, _RESIZE_HW), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def main() -> None:
    args = parse_args()
    gen_dir = Path(args.gen_dir).expanduser().resolve()
    if not gen_dir.is_dir():
        raise NotADirectoryError(f"gen_dir does not exist: {gen_dir}")
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"threshold must be in [0,1], got {args.threshold}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}")

    img_paths = _collect_image_paths(gen_dir)
    if not img_paths:
        raise FileNotFoundError(f"no images found under {gen_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mnist_classifier(args.classifier_ckpt, device)
    preprocess = _build_preprocess()

    pred_list: list[torch.Tensor] = []
    conf_list: list[torch.Tensor] = []

    for i in tqdm(range(0, len(img_paths), args.batch_size), desc="classifier inference"):
        chunk = img_paths[i : i + args.batch_size]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk], dim=0).to(device)
        logits = model(batch).float()
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        pred_list.append(pred.detach().cpu())
        conf_list.append(conf.detach().cpu())

    pred_all = torch.cat(pred_list, dim=0)
    conf_all = torch.cat(conf_list, dim=0)
    high_mask = conf_all >= float(args.threshold)

    n = int(pred_all.numel())
    n_high = int(high_mask.sum().item())
    high_ratio = float(n_high / max(1, n))
    mean_conf = float(conf_all.mean().item())
    mean_conf_high = float(conf_all[high_mask].mean().item()) if n_high > 0 else 0.0

    hist_all = torch.bincount(pred_all, minlength=10).tolist()
    hist_high = torch.bincount(pred_all[high_mask], minlength=10).tolist() if n_high > 0 else [0] * 10

    metrics = {
        "gen_dir": str(gen_dir),
        "classifier_ckpt": str(Path(args.classifier_ckpt).expanduser().resolve()),
        "threshold": float(args.threshold),
        "num_images": n,
        "num_high_conf": n_high,
        "high_conf_ratio": high_ratio,
        "mean_confidence": mean_conf,
        "mean_confidence_high_conf": mean_conf_high,
        "pred_hist_all": hist_all,
        "pred_hist_high_conf": hist_high,
    }

    out_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (gen_dir / "recognizability_metrics.json")
    )
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.save_per_image_jsonl:
        out_jsonl = gen_dir / "recognizability_per_image.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for p, pred, conf in zip(img_paths, pred_all.tolist(), conf_all.tolist()):
                row = {"path": str(p), "pred": int(pred), "confidence": float(conf)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
