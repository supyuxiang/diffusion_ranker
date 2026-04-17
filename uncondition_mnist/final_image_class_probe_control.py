#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMPipeline
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from uncondition_mnist.midstep_layer9_class_probe import HiddenProbe, parse_args
from uncondition_mnist.mnist_classifier_reward import load_mnist_classifier


@torch.inference_mode()
def _classify_final_images(classifier: nn.Module, x_final: torch.Tensor) -> torch.Tensor:
    x = (x_final / 2 + 0.5).clamp(0, 1)
    x_rgb = x.repeat(1, 3, 1, 1)  # [B,3,28,28]
    x_cls = F.interpolate(x_rgb, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=x.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=x.device).view(1, 3, 1, 1)
    x_cls = (x_cls - mean) / std
    logits = classifier(x_cls).float()
    return logits.argmax(dim=-1)


@torch.inference_mode()
def collect_control_features_and_labels(
    *,
    model_path: str,
    classifier_ckpt: str,
    num_samples: int,
    batch_size: int,
    num_inference_steps: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    pipe = DDPMPipeline.from_pretrained(
        str(Path(model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    classifier = load_mnist_classifier(classifier_ckpt, device)

    n_batches = math.ceil(num_samples / batch_size)
    gen = torch.Generator(device=device).manual_seed(seed)
    all_feat: list[torch.Tensor] = []
    all_label: list[torch.Tensor] = []

    for bi in tqdm(range(n_batches), desc="collect control features+labels"):
        bsz = min(batch_size, num_samples - bi * batch_size)
        if bsz <= 0:
            break
        x = torch.randn(
            (
                bsz,
                pipe.unet.config.in_channels,
                pipe.unet.config.sample_size,
                pipe.unet.config.sample_size,
            ),
            generator=gen,
            device=device,
            dtype=next(pipe.unet.parameters()).dtype,
        )
        for t in timesteps:
            eps = pipe.unet(x, t).sample
            x = pipe.scheduler.step(eps, t, x, generator=gen).prev_sample

        labels = _classify_final_images(classifier, x).cpu()
        # Keep probe structure consistent with the mid-step experiment:
        # convert final image [B,1,28,28] -> [B,64,14,14].
        feat = (x / 2 + 0.5).clamp(0, 1).float()
        feat = F.interpolate(feat, size=(14, 14), mode="bilinear", align_corners=False)
        feat = feat.repeat(1, 64, 1, 1).cpu()
        all_feat.append(feat)
        all_label.append(labels)

    X = torch.cat(all_feat, dim=0)
    y = torch.cat(all_label, dim=0)
    return X, y


def train_and_eval_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    n = X.size(0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(0.8 * n)
    tr_idx, va_idx = perm[:n_train], perm[n_train:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    probe = HiddenProbe(input_channels=64, hidden_dim=hidden_dim, num_classes=10).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_acc = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    last_val_acc = -1.0

    for ep in range(epochs):
        probe.train()
        loss_sum = 0.0
        cnt = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.size(0)
            cnt += xb.size(0)

        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = probe(xb).argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        val_acc = correct / max(total, 1)
        last_val_acc = val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep + 1
            best_state = copy.deepcopy(probe.state_dict())

        print(
            f"[control-probe] epoch {ep+1}/{epochs}, train_ce={loss_sum / max(cnt, 1):.4f}, "
            f"val_acc={val_acc:.4f}, best_val_acc={best_val_acc:.4f} (epoch={best_epoch})"
        )

    if best_state is None:
        raise RuntimeError("best_state is None; training did not run correctly")

    probe.load_state_dict(best_state, strict=True)
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = probe(xb).argmax(dim=-1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    best_ckpt_val_acc = correct / max(total, 1)

    binc = torch.bincount(y_va, minlength=10)
    majority = int(binc.argmax().item())
    majority_acc = float((y_va == majority).float().mean().item())

    return {
        "val_acc": float(best_ckpt_val_acc),
        "best_epoch": float(best_epoch),
        "best_val_acc": float(best_val_acc),
        "last_val_acc": float(last_val_acc),
        "majority_val_acc": float(majority_acc),
        "val_size": float(total),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = collect_control_features_and_labels(
        model_path=args.model_path,
        classifier_ckpt=args.classifier_ckpt,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        device=device,
    )
    print(f"[info] control X shape={tuple(X.shape)}, y shape={tuple(y.shape)}")

    metrics = train_and_eval_probe(
        X,
        y,
        seed=args.seed,
        hidden_dim=args.probe_hidden_dim,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        batch_size=args.batch_size,
        device=device,
    )
    result = {
        "control": "final_image_to_predicted_class",
        "num_samples": args.num_samples,
        "num_inference_steps": args.num_inference_steps,
        "probe_hidden_dim": args.probe_hidden_dim,
        "probe_epochs": args.probe_epochs,
        **metrics,
    }
    out_path = Path(args.output_json).expanduser().resolve()
    if out_path.suffix.lower() == ".json":
        out_path = out_path.with_name(out_path.stem + "_control.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
