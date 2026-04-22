from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from class_condition.ranker import (  # noqa: E402
    DiffusionRanker_CrossAttention,
    DiffusionRanker_MLP,
    DiffusionRanker_Transformer,
)
from class_condition.utils import RankerLoss, js_loss, kl_loss, mse_loss  # noqa: E402
from utils import set_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train diffusion ranker on single-class rollout data")
    p.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to one rollout shard file with keys: hiddens, reward, labels.",
    )
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/ranker_ckpts")
    p.add_argument("--run_name", type=str, default="ranker_single_class")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--save_interval", type=int, default=10)

    p.add_argument("--ranker_type", type=str, default="listwise", choices=("listwise", "pointwise", "cross_attention"))
    p.add_argument("--prompt_dim", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--half", action="store_true")
    p.add_argument("--criterion", type=str, default=None, help="Only used by pointwise MLP ranker.")

    p.add_argument("--parallel_size", type=int, default=16, help="Candidate count K in each listwise mini-batch.")
    p.add_argument("--num_bootstrap", type=int, default=8, help="How many list samples to draw from each dataloader batch.")
    p.add_argument("--kl_type", type=str, default="kl", choices=("kl", "js"))
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=("cosine", "none"))
    return p.parse_args()


def _build_ranker(args: argparse.Namespace, input_dim_candidate: int, input_dim_prompt: int) -> torch.nn.Module:
    if args.ranker_type == "listwise":
        return DiffusionRanker_Transformer(
            input_dim_candidate=input_dim_candidate,
            input_dim_prompt=input_dim_prompt,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_layers=args.num_layers,
            half=args.half,
        )
    if args.ranker_type == "pointwise":
        return DiffusionRanker_MLP(
            candidate_hidden_dim=input_dim_candidate,
            prompt_hidden_dim=input_dim_prompt,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            num_layers=args.num_layers,
            dropout=args.dropout,
            half=args.half,
            criterion=args.criterion,
        )
    if args.ranker_type == "cross_attention":
        return DiffusionRanker_CrossAttention(
            candidate_hidden_dim=input_dim_candidate,
            prompt_hidden_dim=input_dim_prompt,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            half=args.half,
        )
    raise NotImplementedError(f"Invalid ranker_type: {args.ranker_type}")


def _load_single_class_data(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    hiddens = obj["hiddens"].to(torch.float32)
    reward = obj["reward"].to(torch.float32).reshape(-1)
    labels = obj["labels"].to(torch.long).reshape(-1)
    meta = obj.get("meta", {})
    if hiddens.ndim != 2:
        raise ValueError(f"hiddens must be 2D (N,D), got shape={tuple(hiddens.shape)}")
    if reward.ndim != 1:
        raise ValueError(f"reward must be 1D (N,), got shape={tuple(reward.shape)}")
    if hiddens.shape[0] != reward.shape[0] or hiddens.shape[0] != labels.shape[0]:
        raise ValueError(
            f"row mismatch: hiddens={hiddens.shape[0]} reward={reward.shape[0]} labels={labels.shape[0]}"
        )
    return hiddens, reward, labels, meta


def _build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.num_epochs))


def _build_prompt_batch(batch_size: int, prompt_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(batch_size, prompt_dim, device=device, dtype=dtype)


def _process_batch(
    ranker: torch.nn.Module,
    ranker_type: str,
    batch_hidden: torch.Tensor,
    batch_reward: torch.Tensor,
    parallel_size: int,
    num_bootstrap: int,
    prompt_dim: int,
    loss_fn: RankerLoss,
    kl_type: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch_hidden.shape[0] < parallel_size:
        raise ValueError(
            f"batch_size ({batch_hidden.shape[0]}) is smaller than parallel_size ({parallel_size}). "
            "Increase --batch_size or lower --parallel_size."
        )

    w_dtype = next(ranker.parameters()).dtype
    batch_hidden = batch_hidden.to(device=device, dtype=w_dtype)
    batch_reward = batch_reward.to(device=device, dtype=w_dtype)

    cand_ls: list[torch.Tensor] = []
    rew_ls: list[torch.Tensor] = []
    for _ in range(num_bootstrap):
        idx = torch.randperm(batch_hidden.shape[0], device=device)[:parallel_size]
        cand_ls.append(batch_hidden[idx])   # (K, D)
        rew_ls.append(batch_reward[idx])    # (K,)

    mini_cand = torch.stack(cand_ls, dim=0)   # (Bmini, K, D)
    mini_rew = torch.stack(rew_ls, dim=0)     # (Bmini, K)

    mini_b = mini_cand.shape[0]
    prompt_2d = _build_prompt_batch(mini_b, prompt_dim, device=device, dtype=w_dtype)

    if ranker_type == "listwise":
        prompt_3d = prompt_2d.unsqueeze(1)  # (Bmini, 1, Dp)
        student_logits = ranker(mini_cand, prompt_3d)  # (Bmini, K)
    else:
        student_logits = ranker(mini_cand, prompt_2d)  # (Bmini, K)

    target = mini_rew.reshape_as(student_logits)
    loss = loss_fn(student_logits, target)
    mse_term = mse_loss(student_logits, target).detach()
    div_term = (kl_loss if kl_type == "kl" else js_loss)(student_logits, target).detach()
    return loss, mse_term, div_term


@torch.inference_mode()
def _evaluate(
    ranker: torch.nn.Module,
    ranker_type: str,
    val_loader: DataLoader,
    parallel_size: int,
    num_bootstrap: int,
    prompt_dim: int,
    loss_fn: RankerLoss,
    kl_type: str,
    device: torch.device,
) -> tuple[float, float, float]:
    ranker.eval()
    sum_loss = 0.0
    sum_mse = 0.0
    sum_div = 0.0
    num_batches = 0

    for batch_hidden, batch_reward in val_loader:
        if batch_hidden.shape[0] < parallel_size:
            continue
        loss, mse_term, div_term = _process_batch(
            ranker=ranker,
            ranker_type=ranker_type,
            batch_hidden=batch_hidden,
            batch_reward=batch_reward,
            parallel_size=parallel_size,
            num_bootstrap=num_bootstrap,
            prompt_dim=prompt_dim,
            loss_fn=loss_fn,
            kl_type=kl_type,
            device=device,
        )
        sum_loss += float(loss.item())
        sum_mse += float(mse_term.item())
        sum_div += float(div_term.item())
        num_batches += 1

    if num_batches == 0:
        return float("nan"), float("nan"), float("nan")
    return sum_loss / num_batches, sum_mse / num_batches, sum_div / num_batches


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device()

    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hiddens, reward, labels, meta = _load_single_class_data(data_path)
    input_dim_candidate = int(hiddens.shape[1])
    ranker = _build_ranker(args, input_dim_candidate=input_dim_candidate, input_dim_prompt=int(args.prompt_dim)).to(device)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args)
    loss_fn = RankerLoss(alpha=args.alpha, kl_type=args.kl_type)

    ds = TensorDataset(hiddens, reward)
    val_ratio = min(max(float(args.val_ratio), 0.0), 0.5)
    n_val = int(round(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    if n_train <= 0:
        raise ValueError(f"Not enough samples for training: total={len(ds)}, val={n_val}")
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False) if n_val > 0 else None

    best_train_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(args.num_epochs):
        ranker.train()
        sum_loss = 0.0
        sum_mse = 0.0
        sum_div = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for batch_hidden, batch_reward in pbar:
            if batch_hidden.shape[0] < args.parallel_size:
                continue
            loss, mse_term, div_term = _process_batch(
                ranker=ranker,
                ranker_type=args.ranker_type,
                batch_hidden=batch_hidden,
                batch_reward=batch_reward,
                parallel_size=args.parallel_size,
                num_bootstrap=args.num_bootstrap,
                prompt_dim=args.prompt_dim,
                loss_fn=loss_fn,
                kl_type=args.kl_type,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sum_loss += float(loss.item())
            sum_mse += float(mse_term.item())
            sum_div += float(div_term.item())
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        if num_batches == 0:
            raise RuntimeError(
                "No train batch was used. This usually means --batch_size < --parallel_size for all batches."
            )

        train_loss = sum_loss / num_batches
        train_mse = sum_mse / num_batches
        train_div = sum_div / num_batches

        if val_loader is not None:
            val_loss, val_mse, val_div = _evaluate(
                ranker=ranker,
                ranker_type=args.ranker_type,
                val_loader=val_loader,
                parallel_size=args.parallel_size,
                num_bootstrap=max(1, args.num_bootstrap // 2),
                prompt_dim=args.prompt_dim,
                loss_fn=loss_fn,
                kl_type=args.kl_type,
                device=device,
            )
        else:
            val_loss, val_mse, val_div = float("nan"), float("nan"), float("nan")

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} train_mse={train_mse:.6f} "
            f"train_{args.kl_type}={train_div:.6f} val_loss={val_loss:.6f} lr={lr_now:.3e}"
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_mse": float(train_mse),
                f"train_{args.kl_type}": float(train_div),
                "val_loss": float(val_loss),
                "val_mse": float(val_mse),
                f"val_{args.kl_type}": float(val_div),
                "lr": float(lr_now),
            }
        )

        ckpt_obj = {
            "state_dict": ranker.state_dict(),
            "args": vars(args),
            "data_meta": meta,
            "data_path": str(data_path),
            "input_dim_candidate": int(input_dim_candidate),
            "input_dim_prompt": int(args.prompt_dim),
            "history": history,
            "labels_unique": labels.unique().tolist(),
        }

        if epoch % args.save_interval == 0:
            torch.save(ckpt_obj, output_dir / f"{args.run_name}_epoch_{epoch}.pth")
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(ckpt_obj, output_dir / f"{args.run_name}_best.pth")

    torch.save(
        {
            "state_dict": ranker.state_dict(),
            "args": vars(args),
            "data_meta": meta,
            "data_path": str(data_path),
            "history": history,
            "best_train_loss": float(best_train_loss),
            "labels_unique": labels.unique().tolist(),
        },
        output_dir / f"{args.run_name}_last.pth",
    )
    (output_dir / f"{args.run_name}_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "best_train_loss": float(best_train_loss),
                "output_dir": str(output_dir),
                "run_name": args.run_name,
                "data_path": str(data_path),
                "num_samples": int(hiddens.shape[0]),
                "hidden_dim": int(hiddens.shape[1]),
                "labels_unique": labels.unique().tolist(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
