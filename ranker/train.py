#!/usr/bin/env python3
r"""
example:

export CUDA_VISIBLE_DEVICES=0
python /root/diffusion_ranker/ranker/train.py \
    --num_epochs 10 \
    --output_dir /root/autodl-tmp/ranker_ckpt_listwise_l06_s1 \
    --image_dir /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10 \
    --candidate_repr_pt /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/mean/step_1_layer_0.6.pt \
    --prompt_repr_pt /root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt \
    --reward_path /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/reward_ls.json \
    --batch_size 32 \
    --num_cand 10 \
    --input_dim_candidate 1152 \
    --input_dim_prompt 3584 \
    --hidden_dim 128 \
    --num_attention_heads 4 \
    --num_layers 4 \
    --dropout 0.1 \
    --optimizer adamw \
    --lr_scheduler cosine \
    --warmup_ratio 0.03 \
    --lr_min_ratio 0.01 \
    --alpha 0.8 \
    --tau 1.0 \
    --seed 42 \
    --num_candidate_groups 8 \
    --ranker_type pointwise \
    --has_prompt_hstates


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



from ranker.loss import mse_kl_combined_loss
from ranker.model import DiffusionRanker_Transformer, DiffusionRanker_MLP, DiffusionRanker_CrossAttention
from ranker.dataset import diffusion_ranker_dataloader
from utils import Constants_default, set_device, set_seed




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
        "--ranker_type",
        type=str,
        default='listwise',
        choices=('listwise', 'pointwise', 'cross_attention'),
        help="ranker_type",
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
        default="/root/autodl-tmp/ranker_ckpt_l06_s00",
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
    p.add_argument(
        "--has_prompt_hstates",
        action="store_true",
        help="传入则从 prompt_repr 取 prompt；不传则 False（勿写 --has_prompt_hstates True，store_true 不接收值）",
    )
    return p.parse_args()




def split_batch_into_candidate_groups(
    batch,
    num_candidate_groups,
    num_cand,
):
    sample_indices_ls = []
    batch_size = batch[0].size(0)
    for i in range(num_candidate_groups):
        group_indices = random.sample(range(batch_size), num_cand)
        sample_indices_ls.append(group_indices)
    groups = [
        [
            batch[0][indices],
            batch[1][indices] if batch[1] is not None else None,
            batch[2][indices],
        ]
        for indices in sample_indices_ls
    ]
    return groups



def process_single_candidate_group(
    ranker,
    loss_fn,
    group,
):
    candidate_hstates, prompt_hstates, rewards = group

    # breakpoint()
    # Ranker 约定：candidate (B, K, D_c)，prompt (B, D_p)；分组切片得到的是 (K, D)
    if candidate_hstates.dim() == 2:
        candidate_hstates = candidate_hstates.unsqueeze(0)  # (1, K, D_c)
    if prompt_hstates is not None and prompt_hstates.dim() == 2:
        # 同一 listwise 组共用一个上下文向量（多行 prompt 通常来自同 class 重复，取首行即可）
        prompt_hstates = prompt_hstates[:1]  # (1, D_p)

    logits = ranker(candidate_hstates, prompt_hstates)  # (1, K)
    # loss 里 KL 在 dim=0 上对「候选维」做 softmax，须为 (K, 1)
    logits = logits.transpose(0, 1).contiguous()
    rewards = rewards.view(-1, 1)
    if logits.shape[0] != rewards.shape[0]:
        raise ValueError(
            f"logits 与 reward 候选数不一致: {logits.shape} vs {rewards.shape}"
        )
    group_loss, group_mse, group_kl = loss_fn(logits, rewards)
    return group_loss, group_mse, group_kl


def train_epoch(
    ranker,
    optimizer,
    scheduler,
    train_loader,
    loss_fn,
    num_candidate_groups: int,
    num_cand: int,
) -> dict[str, float]:
    ranker.train()
    device = next(ranker.parameters()).device
    dtype = next(ranker.parameters()).dtype
    num_candidate_groups_show = max(int(num_candidate_groups), 1)
    epoch_loss = epoch_mse = epoch_kl = 0.0
    n_batch = 0
    pbar = tqdm(train_loader, desc="train", leave=False, total=len(train_loader))
    for batch in pbar:
        batch_c, batch_p, batch_r = batch
        batch = (
            batch_c.to(device=device, dtype=dtype, non_blocking=True),
            batch_p.to(device=device, dtype=dtype, non_blocking=True)
            if batch_p is not None
            else None,
            batch_r.to(device=device, dtype=dtype, non_blocking=True),
        )
        batch_loss = batch_mse = batch_kl = 0.0
        groups = split_batch_into_candidate_groups(batch, num_candidate_groups, num_cand)
        for group in groups:
            group_loss, group_mse, group_kl = process_single_candidate_group(ranker, loss_fn, group)
            batch_loss += group_loss
            batch_mse += group_mse
            batch_kl += group_kl
        pbar.set_postfix(
            batch_loss=float(batch_loss.detach() / num_candidate_groups_show),
            batch_mse_loss=float(batch_mse.detach() / num_candidate_groups_show),
            batch_kl_loss=float(batch_kl.detach() / num_candidate_groups_show),
        )
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()
        n_batch += 1
        epoch_loss += float(batch_loss.detach())
        epoch_mse += float(batch_mse.detach())
        epoch_kl += float(batch_kl.detach())
    if n_batch == 0:
        out = {"epoch_loss": 0.0, "epoch_mse": 0.0, "epoch_kl": 0.0}
    else:
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
        has_prompt_hstates=args.has_prompt_hstates,
    )
    print(f"train_loader built")

    if args.ranker_type == 'listwise':
        ranker = DiffusionRanker_Transformer(
            input_dim_candidate=args.input_dim_candidate,
            input_dim_prompt=args.input_dim_prompt,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_attention_heads,
            dropout=args.dropout,
            num_layers=args.num_layers,
            half=True,
        ).to(device)
    elif args.ranker_type == 'pointwise':
        ranker = DiffusionRanker_MLP(
            candidate_hidden_dim=args.input_dim_candidate,
            prompt_hidden_dim=args.input_dim_prompt,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            num_layers=args.num_layers,
            dropout=args.dropout,
            half=True,
        ).to(device)
        ranker.criterion = 'cos_t1.0'
    elif args.ranker_type == 'cross_attention':
        ranker = DiffusionRanker_CrossAttention(
            candidate_hidden_dim=args.input_dim_candidate,
            prompt_hidden_dim=args.input_dim_prompt,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_attention_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            half=True,
        ).to(device)
    else:
        raise ValueError(f"unknown ranker_type: {args.ranker_type}")


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
            args.num_candidate_groups,
            args.num_cand,
        )
        lr_s = (
            scheduler.get_last_lr()[0]
            if scheduler is not None
            else optimizer.param_groups[0]["lr"]
        )
        print(
            f"epoch {epoch + 1}/{args.num_epochs}  lr={lr_s:.2e}  "
            f"epoch_loss={o['epoch_loss']:.6f}  epoch_mse={o['epoch_mse']:.6f}  epoch_kl={o['epoch_kl']:.6f}"
        )
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
