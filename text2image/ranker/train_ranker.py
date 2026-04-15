#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ranker.model import DiffusionRanker_MLP
from text2image.common import ensure_dir, set_seed


def _build_loader(train_pt: Path, batch_size: int, shuffle: bool) -> tuple[DataLoader, torch.Tensor]:
    obj = torch.load(train_pt, map_location="cpu", weights_only=False)
    cand = obj["candidate_tensor"].float()
    prompt_table = obj["prompt_table"].float()
    pidx = obj["prompt_index"].long()
    rew = obj["reward"].float()
    prompt = prompt_table[pidx]
    ds = TensorDataset(cand, prompt, rew)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), prompt_table


def main() -> None:
    p = argparse.ArgumentParser("Train pointwise ranker for text2image")
    p.add_argument("--train_pt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    train_pt = Path(args.train_pt).expanduser().resolve()
    loader, prompt_table = _build_loader(train_pt, int(args.batch_size), shuffle=True)

    sample = torch.load(train_pt, map_location="cpu", weights_only=False)
    in_c = int(sample["candidate_tensor"].shape[1])
    in_p = int(sample["prompt_table"].shape[1])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ranker = DiffusionRanker_MLP(
        candidate_hidden_dim=in_c,
        prompt_hidden_dim=in_p,
        hidden_dim=int(args.hidden_dim),
        output_dim=1,
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        half=False,
        criterion=None,
    ).to(device).float()

    opt = torch.optim.AdamW(ranker.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    losses: list[float] = []
    for ep in range(int(args.num_epochs)):
        ranker.train()
        ep_losses: list[float] = []
        for cand, prompt, reward in tqdm(loader, desc=f"epoch {ep+1}/{args.num_epochs}"):
            cand = cand.to(device)
            prompt = prompt.to(device)
            reward = reward.to(device).view(-1, 1)
            pred = ranker(cand.unsqueeze(1), prompt).view(-1, 1)
            loss = F.mse_loss(pred, reward)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))
        losses.append(sum(ep_losses) / max(len(ep_losses), 1))

    ckpt = {
        "model_state_dict": ranker.state_dict(),
        "args": {
            "ranker_type": "pointwise",
            "input_dim_candidate": in_c,
            "input_dim_prompt": in_p,
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "selected_repr_keys": ["clip_image"],
            "repr_key": "clip_image",
            "ranker_float32": True,
        },
        "train_loss_curve": losses,
    }
    torch.save(ckpt, out_dir / "ranker_final.pt")
    torch.save({"tensor": prompt_table.float()}, out_dir / "prompt_table.pt")
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "train_pt": str(train_pt),
                "num_epochs": int(args.num_epochs),
                "batch_size": int(args.batch_size),
                "final_loss": float(losses[-1] if losses else float("nan")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
