#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from cifar10_classifier_reward import build_cifar10_resnet18


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train CIFAR-10 classifier for reward modeling")
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp")
    p.add_argument("--save_dir", type=str, default="/root/autodl-tmp/cifar10_classifier/resnet18")
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.shape[0]
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_num += int(x.shape[0])
    return total_loss / max(1, total_num), total_correct / max(1, total_num)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_set = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Hard enforce ImageNet pretrained initialization for finetuning.
    model = build_cifar10_resnet18(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    best_acc = -1.0
    history: list[dict[str, float]] = []
    for ep in range(args.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train {ep+1}/{args.num_epochs}", leave=False)
        running_loss = 0.0
        running_num = 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.shape[0]
            running_num += int(x.shape[0])
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(1, running_num)
        val_loss, val_acc = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": float(ep + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )
        print(f"Epoch {ep+1:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": ep + 1,
            "val_acc": val_acc,
        }
        torch.save(ckpt, save_dir / "classifier_last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, save_dir / "classifier_best.pt")

    (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"done. best_acc={best_acc:.4f}, ckpt_dir={save_dir}")


if __name__ == "__main__":
    main()

