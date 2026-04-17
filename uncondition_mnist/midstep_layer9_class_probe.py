#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from uncondition_mnist.mnist_classifier_reward import load_mnist_classifier


class HiddenProbe(nn.Module):
    def __init__(self, input_channels: int = 64, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=2, stride=2),  # Cx14x14 -> 16x7x7
            nn.GELU(),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 7 * 7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Probe final image class from middle-step hidden")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/1aurent_ddpm-mnist")
    p.add_argument(
        "--classifier_ckpt",
        type=str,
        default="/root/autodl-tmp/mnist_classifier/resnet18/classifier_best.pt",
    )
    p.add_argument("--layer_idx", type=int, default=9)
    p.add_argument("--step_idx", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe_hidden_dim", type=int, default=128)
    p.add_argument("--probe_epochs", type=int, default=10)
    p.add_argument("--probe_lr", type=float, default=1e-4)
    p.add_argument(
        "--output_json",
        type=str,
        default="/root/autodl-tmp/mnist_midstep_layer9_probe_result.json",
    )
    return p.parse_args()


@torch.inference_mode()
def _classify_final_images(classifier: nn.Module, x_final: torch.Tensor) -> torch.Tensor:
    # x_final: [B,1,28,28], roughly in [-1, 1]
    x = (x_final / 2 + 0.5).clamp(0, 1)
    x = x.repeat(1, 3, 1, 1)  # -> RGB-like
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=x.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    logits = classifier(x).float()
    return logits.argmax(dim=-1)


@torch.inference_mode()
def collect_hidden_and_labels(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    pipe = DDPMPipeline.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    classifier = load_mnist_classifier(args.classifier_ckpt, device)

    resnet_blocks = [(n, m) for n, m in pipe.unet.named_modules() if m.__class__.__name__ == "ResnetBlock2D"]
    if args.layer_idx < 0 or args.layer_idx >= len(resnet_blocks):
        raise IndexError(f"layer_idx={args.layer_idx} out of range, total={len(resnet_blocks)}")
    if args.step_idx < 0 or args.step_idx >= len(timesteps):
        raise IndexError(f"step_idx={args.step_idx} out of range, total_steps={len(timesteps)}")
    layer_name, layer_module = resnet_blocks[args.layer_idx]
    print(f"[info] target layer_idx={args.layer_idx}, name={layer_name}, step_idx={args.step_idx}")

    state = {"cur_step": -1, "hidden": None}

    def _hook(_m, _inp, _out):
        if state["cur_step"] != args.step_idx:
            return
        t = _out[0] if isinstance(_out, (tuple, list)) else _out
        state["hidden"] = t.detach()

    handle = layer_module.register_forward_hook(_hook)
    n_batches = math.ceil(args.num_samples / args.batch_size)

    all_hiddens: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    gen = torch.Generator(device=device).manual_seed(args.seed)

    for bi in tqdm(range(n_batches), desc="collect hidden+labels"):
        bsz = min(args.batch_size, args.num_samples - bi * args.batch_size)
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
        state["hidden"] = None
        for si, t in enumerate(timesteps):
            state["cur_step"] = si
            eps = pipe.unet(x, t).sample
            x = pipe.scheduler.step(eps, t, x, generator=gen).prev_sample

        if state["hidden"] is None:
            raise RuntimeError("hidden not captured; check step_idx/layer_idx setup")

        labels = _classify_final_images(classifier, x).cpu()
        hidden = state["hidden"].float().cpu()
        all_hiddens.append(hidden)
        all_labels.append(labels)

    handle.remove()
    X = torch.cat(all_hiddens, dim=0)
    y = torch.cat(all_labels, dim=0)
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

    in_ch = int(X.size(1))
    probe = HiddenProbe(input_channels=in_ch, hidden_dim=hidden_dim, num_classes=10).to(device)
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
        # Validate every epoch and keep the checkpoint with best val accuracy.
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
            f"[probe] epoch {ep+1}/{epochs}, train_ce={loss_sum / max(cnt, 1):.4f}, "
            f"val_acc={val_acc:.4f}, best_val_acc={best_val_acc:.4f} (epoch={best_epoch})"
        )

    if best_state is None:
        raise RuntimeError("best_state is None; training did not run correctly")

    # Report metrics using best-val checkpoint instead of the last epoch.
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
    acc = correct / max(total, 1)

    binc = torch.bincount(y_va, minlength=10)
    majority = int(binc.argmax().item())
    majority_acc = float((y_va == majority).float().mean().item())

    return {
        "val_acc": float(acc),
        "best_epoch": float(best_epoch),
        "best_val_acc": float(best_val_acc),
        "last_val_acc": float(last_val_acc),
        "majority_val_acc": float(majority_acc),
        "val_size": float(total),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = collect_hidden_and_labels(args, device)
    print(f"[info] collected X shape={tuple(X.shape)}, y shape={tuple(y.shape)}")

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
        "layer_idx": args.layer_idx,
        "step_idx": args.step_idx,
        "num_samples": args.num_samples,
        "num_inference_steps": args.num_inference_steps,
        "probe_hidden_dim": args.probe_hidden_dim,
        "probe_epochs": args.probe_epochs,
        **metrics,
    }
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
