#!/usr/bin/env python3
r"""
DiT rollout + ImageReward 教师 + **Point-wise** ranker（仅依赖单条样本的 GAP 表征）。

与 ``train_ranker_imagenet.py`` 的数据管线一致：按类 rollout、缓存 ``(h, ImageReward)`` 扁平池；
训练时对池子打乱，按 ``--train_batch_size`` 抽 **独立样本** 的 batch，**无跨样本 attention**。

损失与 listwise 脚本相同形式：
  ``L = α · MSE(s, R) + (1-α) · KL( softmax(R/τ) || softmax(s/τ) )``，
其中 KL 在 **当前 mini-batch** 的维度上做 softmax（鼓励 batch 内相对排序与教师一致）。
默认 ``α=0.7``，更偏重 MSE（标量对齐）。

``--group_size`` 仍表示 **每个 ImageNet 类并行 rollout 的条数**（决定池子大小 = 1000×group_size），
与 ranker 结构无关。


cd /data/home/yxfeng/diffusion_ranker
CUDA_VISIBLE_DEVICES=7 python3 train_point-wise_ranker_imagenet.py \
  --dit_path /data/yxfeng/models/dit_model \
  --output_dir ./outputs/pointwise_ranker \
  --epochs 15 --group_size 8 --train_batch_size 256 \
  --alpha 0.7 --listwise_tau 1.0 \
  --lr_scheduler cosine \
  --lr 1e-4 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import setup_seed, load_dit

import train_ranker_imagenet as tr

NUM_IMAGENET_CLASSES = tr.NUM_IMAGENET_CLASSES


class PointWiseDiffusionRanker(nn.Module):
    """``h ∈ R^D`` → MLP → 标量分数；不依赖同 batch 其他样本的前向结构（batch 内样本独立）。"""

    def __init__(
        self,
        dit_hidden_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")
        layers: list[nn.Module] = [
            nn.Linear(dit_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        """
        pooled_hidden: [B, D]
        返回 scores [B]
        """
        return self.net(pooled_hidden).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(
        description="DiT point-wise ranker + ImageReward (α·MSE + (1-α)·batch KL)"
    )
    parser.add_argument("--dit_path", type=str, default="/data/yxfeng/models/dit_model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_ROOT / "outputs" / "rollout_ranker_pointwise"),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="每类并行 rollout 条数；扁平池大小 = 1000 × group_size",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="point-wise 训练 micro-batch 大小；KL 在该 batch 维上做 softmax，需 >=2 才有意义",
    )
    parser.add_argument("--deno_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=("none", "cosine", "cosine_warmup", "step", "plateau"),
    )
    parser.add_argument("--lr_warmup_epochs", type=int, default=0)
    parser.add_argument("--lr_min_ratio", type=float, default=0.01)
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--lr_plateau_patience", type=int, default=5)
    parser.add_argument("--lr_plateau_factor", type=float, default=0.5)
    parser.add_argument("--layer_ratio", type=float, default=0.6)
    parser.add_argument("--hidden_layer_idx", type=int, default=None)
    parser.add_argument(
        "--ranker_hidden_dim",
        type=int,
        default=512,
        help="MLP 隐层宽度（类似 listwise 的 d_r）",
    )
    parser.add_argument(
        "--ranker_mlp_extra_layers",
        type=int,
        default=1,
        help="hidden_dim→hidden_dim 的额外残差块数（0 则仅一层宽隐层再接 score）",
    )
    parser.add_argument("--ranker_dropout", type=float, default=0.1)
    parser.add_argument("--listwise_tau", type=float, default=1.0, help="KL 中 softmax 温度 τ")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="MSE 权重；(1-α) 为 batch 维 KL( q || p ) 权重，默认 0.7 偏 MSE",
    )
    parser.add_argument("--prompt_template", type=str, default="a high-quality picture of a {name}")
    parser.add_argument("--imagereward_device", type=str, default=None)
    parser.add_argument(
        "--grad_accum_k",
        type=int,
        default=1,
        help="累积 k 个 micro-batch 再 optimizer.step",
    )
    parser.add_argument(
        "--cache_features_once",
        action="store_true",
        help="仅首轮构建 DiT+ImageReward 缓存，之后 epoch 只打乱训练",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="每多少 epoch 存 pointwise_ranker_diffusion_epoch_{epoch:04d}.pt 并覆盖 latest",
    )
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")
    if args.save_freq < 1:
        raise ValueError("--save_freq must be >= 1")
    if args.train_batch_size < 1:
        raise ValueError("--train_batch_size must be >= 1")
    if args.alpha < 1.0 and args.train_batch_size < 2:
        raise ValueError("当 alpha<1 使用 KL 时，请设 --train_batch_size >= 2")
    if not (0.0 <= args.lr_min_ratio < 1.0):
        raise ValueError("--lr_min_ratio must be in [0, 1)")
    if args.lr_warmup_epochs < 0:
        raise ValueError("--lr_warmup_epochs must be >= 0")
    if args.lr_scheduler == "cosine" and args.lr_warmup_epochs > 0:
        args.lr_scheduler = "cosine_warmup"
    if args.lr_scheduler == "cosine_warmup" and args.lr_warmup_epochs >= args.epochs:
        raise ValueError("--lr_warmup_epochs must be < --epochs for cosine_warmup")

    num_rollouts = NUM_IMAGENET_CLASSES
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ir_device = args.imagereward_device or str(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_dit(args.dit_path, str(device), fp16=args.fp16)
    n_blocks = len(pipe.transformer.transformer_blocks)
    hidden_layer_idx = tr.resolve_hidden_layer_idx(n_blocks, args.layer_ratio, args.hidden_layer_idx)
    dit_hidden_dim = int(pipe.transformer.inner_dim)

    meta = {
        "ranker_type": "pointwise_mlp",
        "num_transformer_blocks": n_blocks,
        "hidden_layer_idx": hidden_layer_idx,
        "layer_ratio": args.layer_ratio,
        "dit_hidden_dim": dit_hidden_dim,
        "ranker_hidden_dim": args.ranker_hidden_dim,
        "ranker_mlp_extra_layers": args.ranker_mlp_extra_layers,
        "group_size": args.group_size,
        "train_batch_size": args.train_batch_size,
        "listwise_tau": args.listwise_tau,
        "loss": "alpha*mse + (1-alpha)*kl_batch_softmax_teacher_ranker",
        "alpha": args.alpha,
        "rollouts_per_epoch": num_rollouts,
        "flat_pool_size": num_rollouts * args.group_size,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({**meta, **vars(args)}, f, indent=2)

    if args.grad_accum_k < 1:
        raise ValueError("--grad_accum_k must be >= 1")
    cache_mode = "feature cache: once" if args.cache_features_once else "feature cache: each epoch"
    print(
        f"[point-wise] DiT blocks={n_blocks}, capture block {hidden_layer_idx}; "
        f"MLP hidden={args.ranker_hidden_dim}, extra_layers={args.ranker_mlp_extra_layers}; "
        f"pool={num_rollouts * args.group_size}, train_batch={args.train_batch_size}; "
        f"loss=α·MSE+(1-α)·batch_KL, α={args.alpha}, τ={args.listwise_tau}; "
        f"grad_accum_k={args.grad_accum_k}; save_freq={args.save_freq}; {cache_mode}"
    )

    labels_1k = tr.load_imagenet_simple_labels()
    tr._ensure_imagereward_path()
    import ImageReward as RM

    teacher = RM.load("ImageReward-v1.0", device=ir_device)
    teacher.eval()

    ranker = PointWiseDiffusionRanker(
        dit_hidden_dim=dit_hidden_dim,
        hidden_dim=args.ranker_hidden_dim,
        num_hidden_layers=args.ranker_mlp_extra_layers,
        dropout=args.ranker_dropout,
    ).to(device)
    opt = torch.optim.AdamW(ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    eta_min = float(args.lr) * float(args.lr_min_ratio)
    scheduler = None
    sched_name = args.lr_scheduler
    if sched_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=eta_min)
    elif sched_name == "cosine_warmup":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        w = int(args.lr_warmup_epochs)
        if w <= 0:
            scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=eta_min)
        else:
            t_cos = max(1, args.epochs - w)
            warmup_sched = LinearLR(opt, start_factor=1e-2, end_factor=1.0, total_iters=w)
            cosine_sched = CosineAnnealingLR(opt, T_max=t_cos, eta_min=eta_min)
            scheduler = SequentialLR(opt, [warmup_sched, cosine_sched], milestones=[w])
    elif sched_name == "step":
        from torch.optim.lr_scheduler import StepLR

        scheduler = StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif sched_name == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
        )

    loss_history: list[dict] = []
    global_step = 0
    optimizer_step = 0
    k_acc = args.grad_accum_k
    accum_in_block = 0
    persistent_feature_cache: dict | None = None

    def _do_optimizer_step(*, remainder: int | None = None):
        nonlocal optimizer_step, accum_in_block
        if remainder is not None and 0 < remainder < k_acc:
            scale = k_acc / remainder
            for p in ranker.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        opt.step()
        opt.zero_grad(set_to_none=True)
        optimizer_step += 1
        accum_in_block = 0

    gs = args.group_size
    tb = args.train_batch_size

    for epoch in range(args.epochs):
        epoch_losses: list[float] = []
        opt.zero_grad(set_to_none=True)
        accum_in_block = 0

        need_feature_extract = persistent_feature_cache is None or not args.cache_features_once
        if need_feature_extract:
            d_hid = dit_hidden_dim
            pool_h = torch.empty(num_rollouts * gs, d_hid, dtype=torch.float32)
            pool_t = torch.empty(num_rollouts * gs, dtype=torch.float32)
            ext_pbar = tqdm(
                range(num_rollouts),
                desc=f"epoch {epoch+1}/{args.epochs} pre-extract",
                leave=False,
            )
            for b in ext_pbar:
                if args.cache_features_once:
                    batch_global = b
                else:
                    batch_global = epoch * num_rollouts + b
                seed = args.seed + batch_global * 9973
                class_id = int(b)

                pil_list, pooled_h = tr.rollout_dit_same_class_pil_and_pooled_hidden(
                    pipe,
                    gs,
                    class_id,
                    args.deno_steps,
                    str(device),
                    seed,
                    hidden_layer_idx,
                    guidance_scale=args.guidance_scale,
                )

                prompt = tr.prompt_for_class(labels_1k, class_id, args.prompt_template)
                ref_scores: list[float] = []
                for img in pil_list:
                    with torch.inference_mode():
                        ref_scores.append(float(teacher.score(prompt, img)))
                teacher_t = torch.tensor(ref_scores, dtype=torch.float32)

                pool_h[b * gs : (b + 1) * gs] = pooled_h.detach().cpu().float()
                pool_t[b * gs : (b + 1) * gs] = teacher_t.cpu()
                ext_pbar.set_postfix(rollout_cid=class_id)

            epoch_cache = {"pooled_h_all": pool_h, "teacher_t_all": pool_t}
            if args.cache_features_once:
                persistent_feature_cache = epoch_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            epoch_cache = persistent_feature_cache
            assert epoch_cache is not None

        pool_h = epoch_cache["pooled_h_all"]
        pool_t = epoch_cache["teacher_t_all"]
        pool_size = int(pool_h.shape[0])

        rng_perm = torch.Generator()
        rng_perm.manual_seed(int(args.seed) + epoch)
        perm = torch.randperm(pool_size, generator=rng_perm)

        n_full = pool_size // tb
        batches: list[torch.Tensor] = []
        for j in range(n_full):
            batches.append(perm[j * tb : (j + 1) * tb])
        rem = pool_size - n_full * tb
        if rem > 0:
            batches.append(perm[n_full * tb :])

        train_pbar = tqdm(
            range(len(batches)),
            desc=f"epoch {epoch+1}/{args.epochs} train point-wise",
            leave=True,
        )
        for bi in train_pbar:
            sl = batches[bi]
            if sl.numel() < 2 and args.alpha < 1.0:
                continue
            pooled_h = pool_h[sl].to(device, non_blocking=True)
            teacher_t = pool_t[sl].to(device, non_blocking=True)

            ranker.train()
            pred_scores = ranker(pooled_h)
            loss, mse_term, kl_term = tr.mse_kl_combined_loss(
                pred_scores, teacher_t, args.listwise_tau, args.alpha
            )

            (loss / k_acc).backward()
            accum_in_block += 1
            loss_f = loss.item()
            mse_f = mse_term.item()
            kl_f = kl_term.item()
            epoch_losses.append(loss_f)
            global_step += 1
            train_pbar.set_postfix(
                loss=f"{loss_f:.4f}",
                mse=f"{mse_f:.4f}",
                kl=f"{kl_f:.4f}",
                bi=bi,
                accum=f"{accum_in_block}/{k_acc}",
            )

            stepped_now = False
            if accum_in_block == k_acc:
                _do_optimizer_step()
                stepped_now = True

            row = {
                "epoch": epoch,
                "batch": bi,
                "step": global_step,
                "optimizer_step": optimizer_step,
                "loss": loss_f,
                "loss_mse": mse_f,
                "loss_kl": kl_f,
                "alpha": args.alpha,
                "hidden_layer_idx": hidden_layer_idx,
                "batch_size": int(sl.numel()),
                "accum_in_block_after_backward": accum_in_block,
                "did_optimizer_step": stepped_now,
            }
            loss_history.append(row)
            with open(out_dir / "loss_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        flush_remainder = accum_in_block
        if flush_remainder > 0:
            _do_optimizer_step(remainder=flush_remainder)

        mean_ep = sum(epoch_losses) / max(len(epoch_losses), 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1} mean loss: {mean_ep:.6f} | lr={lr_now:.2e}")

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(mean_ep)
            else:
                scheduler.step()

        ep1 = epoch + 1
        if ep1 % args.save_freq == 0 or ep1 == args.epochs:
            payload = {
                "ranker_state_dict": ranker.state_dict(),
                "meta": meta,
                "config": vars(args),
                "epoch": ep1,
            }
            ckpt_path = out_dir / f"pointwise_ranker_diffusion_epoch_{ep1:04d}.pt"
            torch.save(payload, ckpt_path)
            latest = out_dir / "pointwise_ranker_diffusion.pt"
            shutil.copy2(ckpt_path, latest)
            print(f"  checkpoint -> {ckpt_path.name} (epoch {ep1}); latest -> {latest.name}")

    try:
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in loss_history]
        losses = [r["loss"] for r in loss_history]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, losses, "-", alpha=0.35, label="per-batch total")
        if len(losses) >= 3:
            k = min(7, len(losses) // 3 + 1)
            pad = losses[:1] * (k // 2) + losses + losses[-1:] * (k // 2)
            smooth = [sum(pad[i : i + k]) / k for i in range(len(losses))]
            plt.plot(steps, smooth, "-", linewidth=2, label=f"moving avg (k={k})")
        plt.xlabel("train step")
        plt.ylabel("α·MSE + (1-α)·KL")
        plt.title(f"Point-wise Ranker (α={args.alpha}, block {hidden_layer_idx})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig_path = out_dir / "loss_curve.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Loss curve saved to {fig_path}")
    except ImportError:
        print("matplotlib 未安装，跳过绘图；loss 已写入 loss_history.jsonl")


if __name__ == "__main__":
    main()
