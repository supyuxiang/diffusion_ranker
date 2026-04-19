#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path("/root/diffusion_ranker")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ranker.model import DiffusionRanker_MLP
from utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train DDPM-tailored pointwise ranker")
    p.add_argument(
        "--train_data_pt",
        type=Path,
        default=Path("/root/autodl-tmp/ddpm_ranker_train_data_deno10.pt"),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/root/autodl-tmp/ddpm_ranker_ckpt_deno10"),
    )
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_freq", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(args.train_data_pt.expanduser().resolve(), map_location="cpu", weights_only=False)
    cand = obj["candidate_tensor"].float()
    rew = obj["reward"].float().view(-1, 1)
    ds = TensorDataset(cand, rew)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    in_c = int(cand.shape[1])
    ranker = DiffusionRanker_MLP(
        candidate_hidden_dim=in_c,
        prompt_hidden_dim=1,
        hidden_dim=int(args.hidden_dim),
        output_dim=1,
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        half=False,
        criterion=None,
    ).to(device).float()
    optimizer = torch.optim.AdamW(
        ranker.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    curve: list[float] = []
    for ep in range(int(args.num_epochs)):
        ranker.train()
        losses: list[float] = []
        for cand_b, rew_b in tqdm(loader, desc=f"ddpm ranker train {ep+1}/{args.num_epochs}"):
            cand_b = cand_b.to(device)
            rew_b = rew_b.to(device)
            pred = ranker(cand_b.unsqueeze(1), None).view(-1, 1)
            loss = F.mse_loss(pred, rew_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = float(sum(losses) / max(len(losses), 1))
        curve.append(mean_loss)
        print(f"[epoch {ep+1}] mse={mean_loss:.6f}")

        if (ep + 1) % max(int(args.save_freq), 1) == 0:
            torch.save(
                {
                    "epoch": int(ep + 1),
                    "model_state_dict": ranker.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": {
                        "ranker_type": "pointwise",
                        "input_dim_candidate": in_c,
                        "input_dim_prompt": 1,
                        "hidden_dim": int(args.hidden_dim),
                        "num_layers": int(args.num_layers),
                        "dropout": float(args.dropout),
                        "selected_repr_keys": ["ddpm_clip_image"],
                        "repr_key": "ddpm_clip_image",
                        "ranker_float32": True,
                        "ddpm_unconditional": True,
                    },
                },
                out_dir / f"ranker_epoch{ep+1}.pt",
            )

    torch.save(
        {
            "model_state_dict": ranker.state_dict(),
            "args": {
                "ranker_type": "pointwise",
                "input_dim_candidate": in_c,
                "input_dim_prompt": 1,
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.num_layers),
                "dropout": float(args.dropout),
                "selected_repr_keys": ["ddpm_clip_image"],
                "repr_key": "ddpm_clip_image",
                "ranker_float32": True,
                "ddpm_unconditional": True,
            },
            "train_loss_curve": curve,
            "train_data_pt": str(args.train_data_pt.expanduser().resolve()),
        },
        out_dir / "ranker_final.pt",
    )
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "train_data_pt": str(args.train_data_pt.expanduser().resolve()),
                "num_samples": int(cand.shape[0]),
                "input_dim_candidate": in_c,
                "num_epochs": int(args.num_epochs),
                "batch_size": int(args.batch_size),
                "final_mse": float(curve[-1] if curve else float("nan")),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out_dir / 'ranker_final.pt'}")


if __name__ == "__main__":
    main()
