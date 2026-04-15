#!/usr/bin/env python3
r"""
example:

export CUDA_VISIBLE_DEVICES=0
python /root/diffusion_ranker/ranker/train.py \
    --num_epochs 100 \
    --output_dir /root/autodl-tmp/ranker_ckpt_listwise_l06_s00 \
    --image_dir /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10 \
    --candidate_repr_pt /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/mean/step_0_layer_0.6.pt \
    --prompt_repr_pt /root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_logits.pt \
    --reward_path /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/reward_ls.json \
    --batch_size 32 \
    --num_cand 10 \
    --input_dim_candidate 1024 \
    --input_dim_prompt 768 \
    --hidden_dim 64 \
    --num_attention_heads 4 \
    --num_layers 2 \
    --dropout 0.1 \
    --optimizer adamw \
    --scheduler cosine \
    --warmup_ratio 0.03 \
    --lr_min_ratio 0.01 \
    --alpha 0.8 \
    --tau 1.0 \
    --seed 42 \


"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LambdaLR, SequentialLR
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))



from ranker.eval_ranker import eval_epoch
from ranker.loss import mse_kl_combined_loss
from ranker.model import DiffusionRanker_Transformer
from ranker.repr_training_common import (
    RankerReprGroupDataset,
    build_flat_indices,
    bucket_by_class,
    forward_listwise_transformer_ranker,
    iter_training_groups,
    load_imagenet_labels,
    load_prompt_repr_table,
    load_reward_lookup_from_json,
    tokens_for_sample,
)
from ranker.dataset import diffusion_ranker_dataloader
from utils import Constants_default, ensure_image_reward_on_path, set_device, set_seed




def get_optimizer(
    model,
    name,
    lr,
    weight_decay,
    sgd_momentum=None,
) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=sgd_momentum,
            nesterov=sgd_momentum > 0,
        )
    raise ValueError(f"unknown optimizer: {name}")


def get_scheduler(
    name,
    optimizer,
    num_epochs,
    warmup_ratio,
    base_lr,
    lr_min_ratio,
) -> torch.optim.lr_scheduler._LRScheduler:
    name = (name or "none").lower()
    warmup_epochs = max(0, int(round(num_epochs * float(warmup_ratio))))
    eta_min = max(0.0, float(base_lr) * float(lr_min_ratio))
    end_factor = eta_min / base_lr if base_lr > 0 else 1.0
    if name == "none":
        return None
    if warmup_epochs >= num_epochs:
        return LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(num_epochs, 1))
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(int(num_epochs), 1), eta_min=eta_min)
    w = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=eta_min)
    return SequentialLR(optimizer, [w, cosine], milestones=[warmup_epochs])
    



def set_args() -> argparse.Namespace:
    cd = Constants_default
    _data = "/root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10"
    p = argparse.ArgumentParser(description="Diffusion ranker（repr.pt + 图；默认单键 layer=0.6 step=0）")
    p.add_argument(
        "--candidate_repr_pt",
        type=str,
        default=f"{_data}/repr.pt",
        help="repr.pt 路径",
    )
    p.add_argument(
        "--image_dir",
        type=str,
        default=_data,
        help="与 repr 中 image_save_names 一致的图目录",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/ranker_train_l06_s00",
        help="checkpoint 输出目录",
    )
    p.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="DiffusionRanker_Transformer 内 TransformerBlock 层数",
    )
    p.add_argument(
        "--num_attention_heads",
        type=int,
        default=4,
        help="MultiheadAttention 头数（须整除 --proj_dim 即内部 hidden_dim）",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="TransformerBlock 内 dropout",
    )
    p.add_argument(
        "--prompt_repr_pt",
        type=str,
        default="",
        help="可选：预计算类别 prompt 向量 ``(1000, D)``；空则 prompt 位置用全零（与 model 中 else 分支一致）",
    )
    p.add_argument("--num_epochs", type=int, default=int(cd.get("num_epochs", 100)))
    p.add_argument("--lr", type=float, default=float(cd.get("lr", 1e-4)))
    p.add_argument("--weight_decay", type=float, default=float(cd.get("weight_decay", 1e-4)))
    p.add_argument(
        "--optimizer",
        type=str,
        default='adamw',
        choices=("adamw", "sgd"),
    )
    p.add_argument("--sgd_momentum", type=float, default=0.9)
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default='cosine',
        choices=("cosine", "linear", "none"),
    )
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.01,
        help="余弦/线性衰减末端 lr / 初始 lr",
    )
    p.add_argument("--alpha", type=float, default=0.8, help="MSE 权重；KL 权重为 1-alpha")
    p.add_argument("--tau", type=float, default=1.0, help="KL 温度")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_freq", type=int, default=1, help="每 N 个 epoch 存 checkpoint")
    p.add_argument("--input_dim_candidate", type=int, default=1024)
    p.add_argument("--input_dim_prompt", type=int, default=768)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--reward_path", type=str, default="/root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/reward_ls.json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_cand", type=int, default=10)
    p.add_argument("--num_candidate_groups", type=int, default=8)
    return p.parse_args()




def split_batch_into_candidate_groups(
    batch,
    num_candidate_groups,
    num_cand,
):
    sample_indices_ls = []
    batch_size = batch[0].size(0)
    indices = torch.arange(batch_size)
    for i in range(num_candidate_groups):
        group_indices = random.sample(indices, num_cand)
        sample_indices_ls.append(group_indices)
    groups = [
        [batch[0][indices], batch[1][indices], batch[2][indices]] for indices in sample_indices_ls
    ]
    return groups



def process_single_candidate_group(
    ranker,
    loss_fn,
    group,
):
    candidate_hstates, prompt_hstates, rewards = group
    logits = ranker(candidate_hstates, prompt_hstates)
    group_loss, group_mse, group_kl = loss_fn(logits, rewards)
    return group_loss, group_mse, group_kl


def train_epoch(
    ranker,
    optimizer,
    scheduler,
    train_loader,
    loss_fn,
) -> dict[str, float]:
    ranker.train()
    epoch_loss = epoch_mse = epoch_kl = 0.0
    n_batch = 0
    pbar = tqdm(train_loader, desc="train", leave=False, total=len(train_loader))
    for batch in pbar:
        batch_loss = batch_mse = batch_kl = 0.0
        groups = split_batch_into_candidate_groups(batch, args.num_candidate_groups, args.num_cand)
        for group in groups:
            group_loss, group_mse, group_kl = process_single_candidate_group(ranker, loss_fn, group)
            batch_loss += group_loss
            batch_mse += group_mse
            batch_kl += group_kl
        pbar.set_postfix(
            batch_loss=batch_loss / args.num_candidate_groups,
            batch_mse_loss=batch_mse / args.num_candidate_groups,
            batch_kl_loss=batch_kl / args.num_candidate_groups,
        )
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()
        n_batch += 1
        epoch_loss += float(batch_loss.detach())
        epoch_mse += float(batch_mse.detach())
        epoch_kl += float(batch_kl.detach())
    epoch_loss /= n_batch
    epoch_mse /= n_batch
    epoch_kl /= n_batch

    out = {
        "epoch_loss": epoch_loss,
        "epoch_mse": epoch_mse,
        "epoch_kl": epoch_kl,
    }

    if scheduler is not None:
        scheduler.step()

    return out




def main() -> None:
    args = set_args()
    set_seed(args.seed)
    device = set_device()

    candidate_repr_path = Path(args.candidate_repr_pt).expanduser().resolve()
    prompt_repr_path = Path(args.prompt_repr_pt).expanduser().resolve()
    reward_path = Path(args.reward_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # build dataoader
    train_loader = diffusion_ranker_dataloader(
        candidate_repr_path,
        reward_path,
        prompt_repr_path,
        args.batch_size,
        shuffle=True,
    )
    print(f"train_loader built")

    ranker = DiffusionRanker_Transformer(
        input_dim_candidate=args.input_dim_candidate,
        input_dim_prompt=args.input_dim_prompt,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_attention_heads,
        dropout=args.dropout,
        num_layers=args.num_layers,
        half=True,
    ).to(device)

    optimizer = get_optimizer(
        ranker,
        args.optimizer,
        args.lr,
        args.weight_decay,
        args.sgd_momentum,
    )

    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        args.num_epochs,
        args.warmup_ratio,
        args.lr,
        args.lr_min_ratio,
    )

    loss_fn = mse_kl_combined_loss(
        alpha=args.alpha,
        tau=args.tau,
    )

    for epoch in range(args.num_epochs):
        o = train_epoch(
            ranker,
            optimizer,
            scheduler,
            train_loader,
            loss_fn,
        )
        print(f"epoch {epoch + 1}/{args.num_epochs}  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"epoch_loss={o['epoch_loss']:.6f}  epoch_mse_loss={o['epoch_mse_loss']:.6f}  epoch_kl_loss={o['epoch_kl_loss']:.6f}")
        if (epoch + 1) % max(args.save_freq, 1) == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": ranker.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, output_dir / f"ranker_epoch{epoch + 1}.pt")
            print(f"checkpoint saved -> {output_dir / f'ranker_epoch{epoch + 1}.pt'}")

    

if __name__ == "__main__":
    main()
