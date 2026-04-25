#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from class_condition.saturation import SaturationTrainer, collect_training_data
from utils import set_seed


STEP_IDXS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
LAYER_IDXS = (2, 5, 7, 16, 19, 21, 24)
LAYER_RATIO_LABELS = {
    2: "0.1",
    5: "0.2",
    7: "0.3",
    16: "0.6",
    19: "0.7",
    21: "0.8",
    24: "0.9",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Grid rollout/train for saturation predictor")
    p.add_argument("--output_root", type=str, default="/root/autodl-tmp/saturation_grid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mc", type=int, default=5)
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--collect_batch_size", type=int, default=32)
    p.add_argument("--num_samples_per_setting", type=int, default=5000)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    p.add_argument("--force_recollect", action="store_true")
    p.add_argument("--force_retrain", action="store_true")
    return p.parse_args()


def _expected_data_path(data_dir: Path, step_idx: int, layer_idx: int, n: int, mc: int) -> Path:
    return data_dir / f"saturation_hs_reward_s{step_idx}_l{layer_idx}_n{n}_mc{mc}.pt"


def _load_existing_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_metrics(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    fieldnames = [
        "layer_idx",
        "layer_ratio",
        "step_idx",
        "best_val_pearson",
        "best_val_loss",
        "best_ckpt",
        "train_data",
        "val_data",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "layer_idx": r["layer_idx"],
                    "layer_ratio": r["layer_ratio"],
                    "step_idx": r["step_idx"],
                    "best_val_pearson": r["best_val_pearson"],
                    "best_val_loss": r["best_val_loss"],
                    "best_ckpt": r["best_ckpt"],
                    "train_data": r["train_data"],
                    "val_data": r["val_data"],
                }
            )


def _plot_grid(rows: list[dict], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for layer_idx in LAYER_IDXS:
        ratio = LAYER_RATIO_LABELS[layer_idx]
        sub = [r for r in rows if int(r["layer_idx"]) == int(layer_idx)]
        sub = sorted(sub, key=lambda x: int(x["step_idx"]))
        if not sub:
            continue
        xs = [int(r["step_idx"]) for r in sub]
        ys = [float(r["best_val_pearson"]) for r in sub]
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=5, label=f"layer ratio: {ratio}")

    ax.set_title("Saturation Predictor Pearson vs Step Index", fontsize=14, pad=12)
    ax.set_xlabel("step idx", fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_xticks(list(STEP_IDXS))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    output_root = Path(args.output_root).expanduser().resolve()
    data_root = output_root / "data"
    train_root = output_root / "train"
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)

    metrics_path = output_root / "grid_metrics.json"
    rows = _load_existing_metrics(metrics_path)

    for layer_idx in LAYER_IDXS:
        layer_data_dir = data_root / f"layer_{layer_idx}"
        layer_data_dir.mkdir(parents=True, exist_ok=True)

        expected = [
            _expected_data_path(
                layer_data_dir,
                step_idx=s,
                layer_idx=layer_idx,
                n=int(args.num_samples_per_setting),
                mc=int(args.mc),
            )
            for s in STEP_IDXS
        ]

        need_collect = args.force_recollect or (not all(p.exists() for p in expected))
        if need_collect:
            print(f"[collect] layer={layer_idx} steps={STEP_IDXS}")
            collect_training_data(
                output_dir=layer_data_dir,
                num_samples_per_setting=int(args.num_samples_per_setting),
                target_step_indices=STEP_IDXS,
                target_layer_idx=int(layer_idx),
                mc=int(args.mc),
                model_name=str(args.model_name),
                dtype=str(args.dtype),
                num_inference_steps=int(args.num_inference_steps),
                batch_size=int(args.collect_batch_size),
                guidance_scale=float(args.guidance_scale),
                seed=int(args.seed) + int(layer_idx) * 10000,
            )

        for step_idx in STEP_IDXS:
            key = (int(layer_idx), int(step_idx))
            run_dir = train_root / f"l{layer_idx}_s{step_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            final_metrics_path = run_dir / "final_metrics.json"
            data_path = _expected_data_path(
                layer_data_dir,
                step_idx=step_idx,
                layer_idx=layer_idx,
                n=int(args.num_samples_per_setting),
                mc=int(args.mc),
            )

            if (not args.force_retrain) and final_metrics_path.exists():
                m = json.loads(final_metrics_path.read_text(encoding="utf-8"))
                row = {
                    "layer_idx": int(layer_idx),
                    "layer_ratio": LAYER_RATIO_LABELS[layer_idx],
                    "step_idx": int(step_idx),
                    "best_val_pearson": float(m["val_pearson"]),
                    "best_val_loss": float(m["best_val_loss_during_training"]),
                    "best_ckpt": str(m["best_ckpt"]),
                    "train_data": str(m["train_data"]),
                    "val_data": str(m["val_data"]),
                }
                rows = [r for r in rows if (int(r["layer_idx"]), int(r["step_idx"])) != key]
                rows.append(row)
                _save_metrics(metrics_path, sorted(rows, key=lambda x: (int(x["layer_idx"]), int(x["step_idx"]))))
                print(f"[skip-train] layer={layer_idx} step={step_idx} (existing metrics)")
                continue

            print(f"[train] layer={layer_idx} step={step_idx}")
            trainer = SaturationTrainer(
                train_path=data_path,
                val_path=data_path,
                output_dir=run_dir,
                seed=int(args.seed),
                batch_size=int(args.train_batch_size),
                num_epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                hidden_dim=int(args.hidden_dim),
                num_layers=int(args.num_layers),
                dropout_prob=float(args.dropout_prob),
            )
            m = trainer.train_and_test()
            row = {
                "layer_idx": int(layer_idx),
                "layer_ratio": LAYER_RATIO_LABELS[layer_idx],
                "step_idx": int(step_idx),
                "best_val_pearson": float(m["val_pearson"]),
                "best_val_loss": float(m["best_val_loss_during_training"]),
                "best_ckpt": str(m["best_ckpt"]),
                "train_data": str(m["train_data"]),
                "val_data": str(m["val_data"]),
            }
            rows = [r for r in rows if (int(r["layer_idx"]), int(r["step_idx"])) != key]
            rows.append(row)
            _save_metrics(metrics_path, sorted(rows, key=lambda x: (int(x["layer_idx"]), int(x["step_idx"]))))

    rows = sorted(rows, key=lambda x: (int(x["layer_idx"]), int(x["step_idx"])))
    _save_metrics(metrics_path, rows)
    plot_path = output_root / "pearson_by_step_and_layer_ratio.png"
    _plot_grid(rows, plot_path)
    print(json.dumps({"metrics_json": str(metrics_path), "plot_png": str(plot_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

