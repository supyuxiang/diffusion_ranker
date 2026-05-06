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

from class_condition.clipscore import ClipScoreTrainer
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
    p = argparse.ArgumentParser("Grid rollout/train for clipscore predictor")
    p.add_argument("--output_root", type=str, default="/root/autodl-tmp/clipscore_grid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mc", type=int, default=5)
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--collect_batch_size", type=int, default=32)
    p.add_argument("--per_class", type=int, default=100)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    p.add_argument("--force_recollect", action="store_true")
    p.add_argument("--force_retrain", action="store_true")
    return p.parse_args()




@torch.inference_mode()
def collect_training_data_multi_layer(
    *,
    output_root: Path,
    per_class: int = 10,
    target_step_indices: tuple[int, ...] = (1, 6, 9),
    target_layer_indices: tuple[int, ...] = (16,),
    mc: int = 5,
    model_name: str = "dit-xl-2-256",
    dtype: str = "bf16",
    num_inference_steps: int = 10,
    batch_size: int = 32,
    guidance_scale: float = 1.0,
    seed: int = 42,
) -> dict[int, dict[int, Path]]:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(model_name, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    clip = CLIPModel.from_pretrained(str(_DEFAULT_CLIP_DIR), local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(_DEFAULT_CLIP_DIR), local_files_only=True)
    class_names = _load_class_names()

    labels_all = torch.tensor([c for c in range(1000) for _ in range(int(per_class))], dtype=torch.long)
    num_samples = int(labels_all.numel())
    
    target_steps = tuple(sorted(set(int(s) for s in target_step_indices)))
    target_layers = tuple(sorted(set(int(l) for l in target_layer_indices)))

    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels
    n_batches = (num_samples + int(batch_size) - 1) // int(batch_size)

    hidden_store: dict[int, dict[int, list[torch.Tensor]]] = {
        layer_idx: {s: [] for s in target_steps} for layer_idx in target_layers
    }
    reward_store: dict[int, list[torch.Tensor]] = {s: [] for s in target_steps}

    current_step_idx = {"value": -1}
    current_batch_size = {"value": 0}
    current_store_ref: dict[str, dict[int, dict[int, list[torch.Tensor]]] | None] = {"store": None}

    def _make_hook(layer_idx: int):
        def _hook(_module, _inp, out) -> None:
            s = current_step_idx["value"]
            store = current_store_ref["store"]
            if store is None or s not in target_steps:
                return
            h = out[0] if isinstance(out, (tuple, list)) else out
            if guidance_scale > 1:
                h = h[: current_batch_size["value"]]
            h = h.mean(dim=1).detach().to(torch.float16).cpu()
            store[layer_idx][s].append(h)

        return _hook

    handles = [
        pipe.transformer.transformer_blocks[layer_idx].register_forward_hook(_make_hook(layer_idx))
        for layer_idx in target_layers
    ]

    try:
        pbar = tqdm(range(n_batches), desc="collect clipscore hidden/reward (all layers)")
        for batch_i in pbar:
            st = batch_i * int(batch_size)
            ed = min(st + int(batch_size), num_samples)
            batch_n = ed - st
            if batch_n <= 0:
                break
            current_batch_size["value"] = batch_n
            batch_labels = labels_all[st:ed].clone()

            gen = torch.Generator(device=device).manual_seed(int(seed) + batch_i)
            latents = torch.randn(
                (batch_n, latent_channels, latent_size, latent_size),
                generator=gen,
                device=device,
                dtype=pipe.transformer.dtype,
            )
            x = torch.cat([latents] * 2, dim=0) if guidance_scale > 1 else latents

            cls = batch_labels.to(device=device)
            cls_null = torch.full((batch_n,), 1000, device=device, dtype=torch.long)
            class_labels_input = torch.cat([cls, cls_null], dim=0) if guidance_scale > 1 else cls

            pipe.scheduler.set_timesteps(int(num_inference_steps))
            timesteps = pipe.scheduler.timesteps

            batch_hidden_store: dict[int, dict[int, list[torch.Tensor]]] = {
                layer_idx: {s: [] for s in target_steps} for layer_idx in target_layers
            }
            current_store_ref["store"] = batch_hidden_store
            batch_snapshot_x: dict[int, torch.Tensor] = {}
            batch_snapshot_scheduler: dict[int, object] = {}

            for st_i, t in enumerate(timesteps):
                current_step_idx["value"] = st_i
                if guidance_scale > 1:
                    half = x[: len(x) // 2]
                    x = torch.cat([half, half], dim=0)
                x_in = pipe.scheduler.scale_model_input(x, t)
                timesteps_b = _broadcast_timesteps(t, x_in)
                noise_pred = pipe.transformer(x_in, timestep=timesteps_b, class_labels=class_labels_input).sample

                if guidance_scale > 1:
                    eps, tail = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    eps = torch.cat([half_eps, half_eps], dim=0)
                    noise_pred = torch.cat([eps, tail], dim=1)

                if pipe.transformer.config.out_channels // 2 == latent_channels:
                    model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
                else:
                    model_output = noise_pred
                x = pipe.scheduler.step(model_output, t, x, generator=gen).prev_sample

                if st_i in target_steps:
                    batch_snapshot_x[st_i] = x.detach().clone()
                    batch_snapshot_scheduler[st_i] = copy.deepcopy(pipe.scheduler)

            current_store_ref["store"] = None

            batch_reward_by_step: dict[int, torch.Tensor] = {}
            for st_i in target_steps:
                reward_t = _mc_clipscore_reward_from_snapshot(
                    pipe=pipe,
                    x_start=batch_snapshot_x[st_i],
                    labels=batch_labels,
                    start_step_idx=st_i + 1,
                    timesteps=timesteps,
                    scheduler_snapshot=batch_snapshot_scheduler[st_i],
                    class_labels_input=class_labels_input,
                    latent_channels=latent_channels,
                    guidance_scale=float(guidance_scale),
                    seed_base=int(seed) + batch_i * 1000 + st_i * 100,
                    mc=int(mc),
                    clip=clip,
                    processor=processor,
                    class_names=class_names,
                )
                batch_reward_by_step[st_i] = reward_t
                reward_store[st_i].append(reward_t)

            for layer_idx in target_layers:
                for st_i in target_steps:
                    h_t = torch.cat(batch_hidden_store[layer_idx][st_i], dim=0)
                    r_t = batch_reward_by_step[st_i]
                    if h_t.shape[0] != r_t.shape[0]:
                        raise RuntimeError(
                            f"rows mismatch for layer {layer_idx}, step {st_i}: "
                            f"hidden={h_t.shape[0]} reward={r_t.shape[0]}"
                        )
                    hidden_store[layer_idx][st_i].append(h_t)

            del latents, x, cls, cls_null, class_labels_input
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    out_paths: dict[int, dict[int, Path]] = {}
    for layer_idx in target_layers:
        layer_dir = output_root / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        out_paths[layer_idx] = {}
        for st_i in target_steps:
            h = torch.cat(hidden_store[layer_idx][st_i], dim=0).float()
            r = torch.cat(reward_store[st_i], dim=0).float()
            if h.shape[0] != r.shape[0]:
                raise RuntimeError(
                    f"final rows mismatch for layer {layer_idx}, step {st_i}: "
                    f"hidden={h.shape[0]} reward={r.shape[0]}"
                )
            out = layer_dir / f"clipscore_hs_reward_s{st_i}_l{layer_idx}_per{per_class}_mc{mc}.pt"
            torch.save(
                {
                    "hiddens": h,
                    "reward": r,
                    "labels": labels_all,
                    "meta": {
                        "target_step_idx": int(st_i),
                        "target_layer_idx": int(layer_idx),
                        "target_layer_indices": [int(x) for x in target_layers],
                        "per_class": int(per_class),
                        "num_samples": int(num_samples),
                        "mc": int(mc),
                        "reward_name": "clipscore",
                        "prompt_template": "A photo of a {class_name}.",
                    },
                },
                out,
            )
            out_paths[layer_idx][st_i] = out

    return out_paths


def _expected_data_path(data_dir: Path, step_idx: int, layer_idx: int, per_class: int, mc: int) -> Path:
    return data_dir / f"clipscore_hs_reward_s{step_idx}_l{layer_idx}_per{per_class}_mc{mc}.pt"


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

    ax.set_title("ClipScore Predictor Pearson vs Step Index", fontsize=14, pad=12)
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

    expected_paths: list[Path] = []
    for layer_idx in LAYER_IDXS:
        layer_data_dir = data_root / f"layer_{layer_idx}"
        layer_data_dir.mkdir(parents=True, exist_ok=True)
        for step_idx in STEP_IDXS:
            expected_paths.append(
                _expected_data_path(
                    layer_data_dir,
                    step_idx=step_idx,
                    layer_idx=layer_idx,
                    per_class=int(args.per_class),
                    mc=int(args.mc),
                )
            )

    need_collect_once = args.force_recollect or (not all(p.exists() for p in expected_paths))
    if need_collect_once:
        print(f"[collect-once] layers={LAYER_IDXS} steps={STEP_IDXS}")
        collect_training_data_multi_layer(
            output_root=data_root,
            per_class=int(args.per_class),
            target_step_indices=STEP_IDXS,
            target_layer_indices=LAYER_IDXS,
            mc=int(args.mc),
            model_name=str(args.model_name),
            dtype=str(args.dtype),
            num_inference_steps=int(args.num_inference_steps),
            batch_size=int(args.collect_batch_size),
            guidance_scale=float(args.guidance_scale),
            seed=int(args.seed),
        )
    else:
        print("[skip-collect] all layer-step rollout files already exist")

    for layer_idx in LAYER_IDXS:
        layer_data_dir = data_root / f"layer_{layer_idx}"

        for step_idx in STEP_IDXS:
            key = (int(layer_idx), int(step_idx))
            run_dir = train_root / f"l{layer_idx}_s{step_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            final_metrics_path = run_dir / "final_metrics.json"
            data_path = _expected_data_path(
                layer_data_dir,
                step_idx=step_idx,
                layer_idx=layer_idx,
                per_class=int(args.per_class),
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
            trainer = ClipScoreTrainer(
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

