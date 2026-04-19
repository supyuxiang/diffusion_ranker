#!/usr/bin/env python3
"""
Train Ranker_MLP on DDPM rollout features.

Unlike Probe (pointwise), the Ranker sees K candidates jointly and
predicts per-candidate scores. Training uses a combined loss:
  L = alpha_kl * KL(softmax(pred/tau) || softmax(target/tau))
    + alpha_mse * MSE(pred, target)

Example (single merged .pth):

python /root/diffusion_ranker/uncondition_celebahq/train_ranker.py \
  --data_path /root/autodl-tmp/ddpm_hidden_store_demo.pth \
  --step_idx 12 \
  --layer_idx 19 \
  --num_candidates 10 \
  --num_epochs 30 \
  --batch_size 32 \
  --save_dir /root/autodl-tmp/ranker_model/s12_l19/hd128_lr1e-4

Example (shard directory):

python /root/diffusion_ranker/uncondition_celebahq/train_ranker.py \
  --data_path /root/autodl-tmp/ddpm_hiddens_and_rewards_shards \
  --step_idx 12 \
  --layer_idx 19 \
  --num_candidates 10 \
  --num_epochs 30 \
  --batch_size 32 \
  --save_dir /root/autodl-tmp/ranker_model/s12_l19/hd128_lr1e-4
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from ranker import Ranker_MLP


def _load_flat_hidden_reward(
    data_path: str | Path,
    *,
    step_idx: int,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load flat hidden/reward tensors from either:
      1) merged rollout .pth, or
      2) shard directory with
         hidden_s{step}_l{layer}_bXXXXX.pt / reward_s{step}_bXXXXX.pt.
    Returns:
      hiddens: (N, C, H, W), rewards: (N,)
    """
    p = Path(data_path).expanduser().resolve()
    if p.is_dir():
        hidden_paths = sorted(p.glob(f"hidden_s{int(step_idx)}_l{int(layer_idx)}_b*.pt"))
        reward_paths = sorted(p.glob(f"reward_s{int(step_idx)}_b*.pt"))
        if not hidden_paths:
            raise FileNotFoundError(f"no hidden shards found in {p}")
        if not reward_paths:
            raise FileNotFoundError(f"no reward shards found in {p}")

        reward_map = {rp.stem.split("_b")[-1]: rp for rp in reward_paths}
        hs_chunks: list[torch.Tensor] = []
        rw_chunks: list[torch.Tensor] = []
        for hp in hidden_paths:
            bid = hp.stem.split("_b")[-1]
            rp = reward_map.get(bid)
            if rp is None:
                raise ValueError(f"missing reward shard for hidden batch id {bid}")
            h = torch.load(hp, map_location="cpu", weights_only=False).float().contiguous()
            r = torch.load(rp, map_location="cpu", weights_only=False).float().contiguous()
            if h.shape[0] != r.shape[0]:
                raise ValueError(f"batch size mismatch in shard {bid}: hidden={h.shape[0]} reward={r.shape[0]}")
            hs_chunks.append(h)
            rw_chunks.append(r)
        hiddens = torch.cat(hs_chunks, dim=0)
        rewards = torch.cat(rw_chunks, dim=0)
        return hiddens, rewards

    raw = torch.load(p, map_location="cpu", weights_only=False)
    hidden_store = raw["hidden_store_target"]
    reward_store = raw["reward_store"]
    hidden_key = f"s{int(step_idx)}_l{int(layer_idx)}"
    reward_key = f"s{int(step_idx)}"
    if hidden_key not in hidden_store:
        raise KeyError(f"missing hidden key: {hidden_key}; available={list(hidden_store.keys())[:10]}")
    if reward_key not in reward_store:
        raise KeyError(f"missing reward key: {reward_key}; available={list(reward_store.keys())[:10]}")
    hiddens = hidden_store[hidden_key].float().contiguous()
    rewards = reward_store[reward_key].float().contiguous()
    return hiddens, rewards


class RankerGroupDataset(Dataset):
    """
    Loads flat (N, C, H, W) hidden states and (N,) rewards from ddpm_rollout,
    then randomly groups them into (num_groups, K, C, H, W) / (num_groups, K).
    Each group of K candidates forms one training sample for the ranker.
    """

    def __init__(
        self,
        data_path: str | Path,
        step_idx: int = 12,
        layer_idx: int = 19,
        num_candidates: int = 10,
        group_seed: int = 0,
    ):
        hiddens, rewards = _load_flat_hidden_reward(
            data_path=data_path,
            step_idx=step_idx,
            layer_idx=layer_idx,
        )
        if hiddens.shape[0] != rewards.shape[0]:
            raise ValueError(f"N mismatch: hidden={hiddens.shape[0]}, reward={rewards.shape[0]}")

        N = hiddens.shape[0]
        K = int(num_candidates)
        if K <= 0:
            raise ValueError(f"num_candidates must be > 0, got {K}")
        rng = random.Random(group_seed)
        indices = list(range(N))
        rng.shuffle(indices)
        # trim to multiple of K
        n_use = (N // K) * K
        if n_use <= 0:
            raise ValueError(
                f"not enough samples to form one group: N={N}, num_candidates={K}. "
                f"Please reduce --num_candidates or provide more rollout data."
            )
        indices = indices[:n_use]

        idx_t = torch.tensor(indices, dtype=torch.long)
        self.hiddens = hiddens[idx_t].view(-1, K, *hiddens.shape[1:])  # (G, K, C, H, W)
        self.rewards = rewards[idx_t].view(-1, K)                       # (G, K)

    def __len__(self) -> int:
        return int(self.hiddens.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hiddens[idx], self.rewards[idx]


def build_datasets(
    *,
    data_path: str,
    step_idx: int,
    layer_idx: int,
    num_candidates: int,
    group_seed: int,
    train_ratio: float,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    full = RankerGroupDataset(
        data_path=data_path,
        step_idx=step_idx,
        layer_idx=layer_idx,
        num_candidates=num_candidates,
        group_seed=group_seed,
    )
    n_total = len(full)
    n_train = max(1, min(n_total - 1, int(n_total * train_ratio)))
    n_eval = n_total - n_train
    gen = torch.Generator().manual_seed(split_seed)
    return random_split(full, [n_train, n_eval], generator=gen)


class KLMSELoss(nn.Module):
    """alpha_kl * KL(softmax(pred/tau) || softmax(target/tau)) + alpha_mse * MSE(pred, target)"""

    def __init__(self, alpha_kl: float = 0.3, alpha_mse: float = 0.7, tau: float = 1.0):
        super().__init__()
        self.alpha_kl = alpha_kl
        self.alpha_mse = alpha_mse
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        log_p = F.log_softmax(pred / self.tau, dim=-1)
        q = F.softmax(target / self.tau, dim=-1)
        kl = F.kl_div(log_p, q, reduction="batchmean")
        return self.alpha_kl * kl + self.alpha_mse * mse


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
        # batch_hs: [B, K, C, H, W], target_reward: [B, K]
        batch_hs = batch_hs.to(device, non_blocking=True)
        target_reward = target_reward.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_hs)        # [B, K]
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
    p = argparse.ArgumentParser("Train Ranker_MLP on DDPM rollout features (listwise KL+MSE)")
    p.add_argument(
        "--data_path",
        type=str,
        default="/root/autodl-tmp/ddpm_hiddens_and_rewards_shards",
        help="path to merged .pth OR shard directory",
    )
    p.add_argument("--step_idx", type=int, default=12)
    p.add_argument("--layer_idx", type=int, default=19)
    p.add_argument("--num_candidates", type=int, default=10, help="K: group size for listwise ranking")
    p.add_argument("--group_seed", type=int, default=0, help="random seed for candidate grouping")
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--input_channels", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--alpha_kl", type=float, default=0.3)
    p.add_argument("--alpha_mse", type=float, default=0.7)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=("adamw", "sgd"))
    p.add_argument("--scheduler", type=str, default="cosine", choices=("cosine", "linear"))
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--save_interval", type=int, default=2)
    p.add_argument("--save_dir", type=str, default="/root/autodl-tmp/ranker_model")
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
        num_candidates=args.num_candidates,
        group_seed=args.group_seed,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    model = Ranker_MLP(
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
    loss_fn = KLMSELoss(alpha_kl=args.alpha_kl, alpha_mse=args.alpha_mse, tau=args.tau)

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
            torch.save(model.state_dict(), save_dir / "ranker_best.pt")

        if (ep + 1) % max(1, args.save_interval) == 0:
            torch.save(model.state_dict(), save_dir / f"ranker_epoch_{ep+1}.pt")

    torch.save(model.state_dict(), save_dir / "ranker_last.pt")
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done. best_eval={best_eval:.6f}, ckpt_dir={save_dir}")


if __name__ == "__main__":
    main()
