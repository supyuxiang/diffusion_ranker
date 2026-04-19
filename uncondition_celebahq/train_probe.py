#!/usr/bin/env python3

'''

python /root/diffusion_ranker/uncondition_celebahq/train_probe.py \
  --data_path /root/autodl-tmp/ddpm_hiddens_and_rewards_shards \
  --step_idx 24 \
  --layer_idx 19 \
  --num_epochs 30 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --save_interval 1 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --train_ratio 0.9 \
  --split_seed 42 \
  --num_layers 2 \
  --optimizer adamw \
  --scheduler cosine \
  --dropout 0.1 \
  --hidden_dim 128 \
  --save_dir /root/autodl-tmp/probe_model/s24_l19/hd128_lr1e-4_wd1e-4

'''
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from ranker import Probe_MLP


class ProbeDataset(Dataset):
    def __init__(self, data_path: str | Path, step_idx: int = 12, layer_idx: int = 19):
        raw = torch.load(Path(data_path).expanduser().resolve(), map_location="cpu", weights_only=False)
        hidden_store = raw["hidden_store_target"]
        reward_store = raw["reward_store"]

        hidden_key = f"s{int(step_idx)}_l{int(layer_idx)}"
        reward_key = f"s{int(step_idx)}"
        if hidden_key not in hidden_store:
            raise KeyError(f"missing hidden key: {hidden_key}; available={list(hidden_store.keys())[:10]}")
        if reward_key not in reward_store:
            raise KeyError(f"missing reward key: {reward_key}; available={list(reward_store.keys())[:10]}")

        self.hiddens = hidden_store[hidden_key].float().contiguous()
        self.rewards = reward_store[reward_key].float().contiguous()
        if self.hiddens.shape[0] != self.rewards.shape[0]:
            raise ValueError(f"N mismatch: hidden={self.hiddens.shape[0]}, reward={self.rewards.shape[0]}")

    def __len__(self) -> int:
        return int(self.hiddens.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hiddens[idx], self.rewards[idx]


class ProbeShardDataset(Dataset):
    """
    Read probe training data directly from shard directory:
      hidden_s{step}_l{layer}_bXXXXX.pt
      reward_s{step}_bXXXXX.pt
    """

    def __init__(self, shard_dir: str | Path, step_idx: int = 12, layer_idx: int = 19):
        shard_dir = Path(shard_dir).expanduser().resolve()
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"shard_dir is not a directory: {shard_dir}")

        hidden_paths = sorted(shard_dir.glob(f"hidden_s{int(step_idx)}_l{int(layer_idx)}_b*.pt"))
        reward_paths = sorted(shard_dir.glob(f"reward_s{int(step_idx)}_b*.pt"))
        if not hidden_paths:
            raise FileNotFoundError(f"no hidden shards found in {shard_dir}")
        if not reward_paths:
            raise FileNotFoundError(f"no reward shards found in {shard_dir}")

        reward_map = {p.stem.split("_b")[-1]: p for p in reward_paths}
        self.records: list[tuple[Path, Path, int, int]] = []
        self.cum_ends: list[int] = []
        total = 0

        for hp in hidden_paths:
            bid = hp.stem.split("_b")[-1]
            rp = reward_map.get(bid)
            if rp is None:
                raise ValueError(f"missing reward shard for hidden batch id {bid}")
            h = torch.load(hp, map_location="cpu", weights_only=False)
            r = torch.load(rp, map_location="cpu", weights_only=False)
            if h.shape[0] != r.shape[0]:
                raise ValueError(f"batch size mismatch in shard {bid}: hidden={h.shape[0]} reward={r.shape[0]}")
            n = int(h.shape[0])
            start = total
            total += n
            end = total
            self.records.append((hp, rp, start, end))
            self.cum_ends.append(end)
            del h, r

        self.total = total
        self._cache_bid: str | None = None
        self._cache_h: torch.Tensor | None = None
        self._cache_r: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.total)

    def _load_record(self, rec_idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        hp, rp, start, _end = self.records[rec_idx]
        bid = hp.stem.split("_b")[-1]
        if self._cache_bid != bid:
            self._cache_h = torch.load(hp, map_location="cpu", weights_only=False).float().contiguous()
            self._cache_r = torch.load(rp, map_location="cpu", weights_only=False).float().contiguous()
            self._cache_bid = bid
        assert self._cache_h is not None and self._cache_r is not None
        return self._cache_h, self._cache_r, start

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        rec_idx = bisect.bisect_right(self.cum_ends, idx)
        h, r, start = self._load_record(rec_idx)
        local_idx = idx - start
        return h[local_idx], r[local_idx]


def build_datasets(
    *,
    data_path: str,
    step_idx: int,
    layer_idx: int,
    train_ratio: float,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    data_p = Path(data_path).expanduser().resolve()
    if data_p.is_dir():
        full = ProbeShardDataset(shard_dir=data_p, step_idx=step_idx, layer_idx=layer_idx) # shard的情况
    else:
        full = ProbeDataset(data_path=data_p, step_idx=step_idx, layer_idx=layer_idx) # 全量的情况
    n_total = len(full)
    n_train = int(n_total * train_ratio)
    n_train = max(1, min(n_total - 1, n_train))
    n_eval = n_total - n_train
    gen = torch.Generator().manual_seed(split_seed)
    train_ds, eval_ds = random_split(full, [n_train, n_eval], generator=gen)
    return train_ds, eval_ds


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: LambdaLR | None,
    device: torch.device,
    loss_fn: nn.Module,
    *,
    desc: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_steps = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch_hs, target_reward in pbar:
        batch_hs = batch_hs.to(device, non_blocking=True)
        target_reward = target_reward.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_hs)
            loss = loss_fn(pred, target_reward)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.inference_mode():
                pred = model(batch_hs)
                loss = loss_fn(pred, target_reward)

        total_loss += float(loss.item())
        n_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, n_steps)


def get_optimizer(
    model: nn.Module,
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    n = name.lower().strip()
    if n == "adamw":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if n == "sgd":
        return SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    num_epochs: int,
    num_steps_per_epoch: int,
    warmup_ratio: float = 0.01,
) -> LambdaLR:
    total_steps = max(1, int(num_epochs) * int(num_steps_per_epoch))
    warmup_steps = min(total_steps - 1, max(1, int(total_steps * warmup_ratio)))
    n = name.lower().strip()

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if n == "linear":
            return max(0.0, 1.0 - progress)
        if n == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"unknown scheduler: {name}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train Probe_MLP on DDPM rollout features")
    p.add_argument("--data_path", type=str, default="/root/autodl-tmp/ddpm_hiddens_and_rewards_shards")
    p.add_argument("--step_idx", type=int, default=12)
    p.add_argument("--layer_idx", type=int, default=19)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--input_channels", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=("adamw", "sgd"))
    p.add_argument("--scheduler", type=str, default="cosine", choices=("cosine", "linear"))
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--save_interval", type=int, default=2)
    p.add_argument("--save_dir", type=str, default="/root/autodl-tmp/probe_model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, eval_ds = build_datasets(
        data_path=args.data_path,
        step_idx=args.step_idx,
        layer_idx=args.layer_idx,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = Probe_MLP(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        input_channels=args.input_channels,
        dropout=args.dropout,
    ).to(device)
    optimizer = get_optimizer(
        model,
        name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(
        optimizer,
        name=args.scheduler,
        num_epochs=args.num_epochs,
        num_steps_per_epoch=len(train_loader),
        warmup_ratio=0.01,
    )
    loss_fn = nn.MSELoss()

    best_eval = float("inf")
    history: list[dict[str, float]] = []
    for ep in range(args.num_epochs):
        train_loss = run_one_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn, desc=f"train {ep+1}/{args.num_epochs}"
        )
        eval_loss = run_one_epoch(
            model, eval_loader, None, None, device, loss_fn, desc=f"eval  {ep+1}/{args.num_epochs}"
        )
        history.append({"epoch": float(ep + 1), "train_loss": float(train_loss), "eval_loss": float(eval_loss)})
        print(f"Epoch {ep+1:03d}: train={train_loss:.6f} eval={eval_loss:.6f}")

        if eval_loss < best_eval:
            best_eval = eval_loss
            torch.save(model.state_dict(), save_dir / "probe_best.pt")

        if (ep + 1) % max(1, args.save_interval) == 0:
            torch.save(model.state_dict(), save_dir / f"probe_epoch_{ep+1}.pt")

    torch.save(model.state_dict(), save_dir / "probe_last.pt")
    (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done. best_eval={best_eval:.6f}, ckpt_dir={save_dir}")


if __name__ == "__main__":
    main()
