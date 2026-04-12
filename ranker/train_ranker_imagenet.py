#!/usr/bin/env python3
r"""
DiT rollout + ImageReward 教师 + Diffusion Ranker 式 **跨样本 Transformer ranker**。

对齐论文设定：
  - 从选定 Transformer 层提取最后一步去噪的 patch 表征，GAP 得到 h_i^(l)；
  - x_i = W_proj h_i + b（d_r 维），**不加位置编码**，将 N 个候选作为序列送入 M 层
    Transformer Encoder，再做打分头 s_i；
  - 同一条件 c 下并行 N 条轨迹（此处为 **同一 ImageNet 类**、**不同随机噪声**）；
  - 训练目标：``L = α·MSE(s,R) + (1-α)·KL(q||p)``，``q=softmax(R/τ)``，``p=softmax(s/τ)``。

**每个 epoch（默认）**：先对本 epoch 全部 micro-batch 做 DiT rollout + ImageReward，将 GAP 表征与教师标量缓存到 CPU；
再仅对 ranker 做反向与优化。

**``--cache_features_once``**：只在**第 1 个 epoch** 做一次预提取，之后各 epoch 只打乱训练池、训练 ranker。

**数据流**：按 ImageNet 类 0..999 各 rollout 一组（每组 ``--group_size`` 张），将 ``(GAP 表征 h, ImageReward 分数)``
展平为单一池；训练时每 epoch **打乱**后按 ``group_size`` 切块送入 ranker。

层索引：``--layer_ratio``（默认 0.6）或 ``--hidden_layer_idx``。

示例（全 1000 类：1000 次 DiT rollout，训练池 1000×group_size 条，再 shuffle 切块）:
  CUDA_VISIBLE_DEVICES=0 python3 /root/diffusion_ranker/train_ranker_imagenet.py \
    --model_name dit-xl-2-256 \
    --output_dir /hy-tmp/diffusion_ranker/outputs/ranker_mse_kl \
    --epochs 100 \
    --layer_ratio 0.6 \
    --group_size 32 \
    --ranker_tf_layers 2 \
    --listwise_tau 1.0 \
    --alpha 0.3 \
    --deno_steps 10 \
    --grad_accum_k 1 \
    --guidance_scale 1.0 \
    --ranker_proj_dim 512 \
    --output_dir /hy-tmp/diffusion_ranker/outputs/ranker_mse_kl/g32_alpha0.3_layer_cfg1.0_ratio0.6_d512_lr1e-4_cosine \
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
import urllib.request
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils import ensure_image_reward_on_path, model_loader, set_seed

NUM_IMAGENET_CLASSES = 1000
NULL_CLASS_ID = 1000
OUTPUT_IMAGE_SIZE = 256
# 同组内各条轨迹的子种子间隔（与历史行为一致）
LATENT_TRAJECTORY_SEED_STRIDE = 10_007
# rollout 索引 -> DiT 随机种子（与历史行为一致）
ROLLOUT_GLOBAL_SEED_MULTIPLIER = 9973


def _ensure_imagereward_path() -> None:
    """See ``utils.ensure_image_reward_on_path``（默认识别 ``/root/ImageReward-main``）。"""
    ensure_image_reward_on_path()


def resolve_hidden_layer_idx(num_blocks: int, layer_ratio: float | None, hidden_layer_idx: int | None) -> int:
    if hidden_layer_idx is not None:
        return int(max(0, min(hidden_layer_idx, num_blocks - 1)))
    if layer_ratio is None:
        layer_ratio = 0.6
    r = max(0.0, min(1.0, float(layer_ratio)))
    if num_blocks <= 1:
        return 0
    return int(round(r * (num_blocks - 1)))


def load_imagenet_simple_labels(cache_dir: Path | None = None) -> list[str]:
    cache_dir = cache_dir or (Path.home() / ".cache" / "diffusion_ranker")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "imagenet-simple-labels.json"
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    if not path.exists():
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"警告: 无法下载 ImageNet 标签 ({e})，将使用占位 prompt。")
            return [f"imagenet class {i}" for i in range(1000)]
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if len(labels) != 1000:
        return [f"imagenet class {i}" for i in range(1000)]
    return labels


def prompt_for_class(labels: list[str], class_id: int, template: str) -> str:
    name = labels[class_id] if 0 <= class_id < len(labels) else f"class {class_id}"
    return template.format(name=name, id=class_id)


def _expand_timestep(t: torch.Tensor | int, batch_size: int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(t):
        return torch.tensor([t], dtype=torch.long, device=device).expand(batch_size)
    return (t.expand(batch_size) if t.dim() == 0 else t).to(device)


def _merge_cfg_noise_pred(
    noise_pred: torch.Tensor,
    latent_channels: int,
    guidance_scale: float,
    *,
    duplicate_merged_eps_for_double_batch: bool,
) -> torch.Tensor:
    """DiT CFG：与 extract_dit_token_repr / diffusers 一致时 merged eps 需拼回 2B batch。"""
    if guidance_scale <= 1:
        return noise_pred
    eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    merged = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    if duplicate_merged_eps_for_double_batch:
        merged = torch.cat([merged, merged], dim=0)
    return torch.cat([merged, rest], dim=1)


def _scheduler_model_output(noise_pred: torch.Tensor, pipe, latent_channels: int) -> torch.Tensor:
    if pipe.transformer.config.out_channels // 2 == latent_channels:
        model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        return model_output
    return noise_pred


@contextmanager
def _forward_hook(module: torch.nn.Module, hook_fn: Callable) -> AbstractContextManager[None]:
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _forward_hooks(block_modules: list, hook_fns: list) -> AbstractContextManager[None]:
    handles = [m.register_forward_hook(fn) for m, fn in zip(block_modules, hook_fns)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _cat_latents_same_class(
    pipe,
    group_size: int,
    device: str,
    seed: int,
    class_id: int,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size
    chunks = []
    for i in range(group_size):
        g = torch.Generator(device=device).manual_seed(seed + i * LATENT_TRAJECTORY_SEED_STRIDE)
        z = randn_tensor(
            (1, latent_channels, latent_size, latent_size),
            generator=g,
            device=device,
            dtype=pipe.transformer.dtype,
        )
        chunks.append(z)
    latents = torch.cat(chunks, dim=0)
    if guidance_scale > 1:
        latents = torch.cat([latents] * 2, dim=0)
    class_labels = torch.full((group_size,), class_id, device=device, dtype=torch.long)
    class_null = torch.tensor([NULL_CLASS_ID] * group_size, device=device)
    labels = torch.cat([class_labels, class_null], dim=0) if guidance_scale > 1 else class_labels
    return latents, labels, latent_channels


def _dit_denoise_loop(
    pipe,
    num_steps: int,
    latents: torch.Tensor,
    class_labels_input: torch.Tensor,
    latent_channels: int,
    guidance_scale: float,
    step_forward_ctx: Callable[[int, int], AbstractContextManager[None]],
) -> torch.Tensor:
    """``step_forward_ctx(step_i, num_steps)`` 包住该步的 transformer 前向（用于最后一步 hook）。"""
    pipe.scheduler.set_timesteps(num_steps)
    timesteps = list(pipe.scheduler.timesteps)
    n_t = len(timesteps)
    for step_i, t in enumerate(timesteps):
        if guidance_scale > 1:
            half = latents[: len(latents) // 2]
            latents = torch.cat([half, half], dim=0)
        latent_input = pipe.scheduler.scale_model_input(latents, t)
        ts = _expand_timestep(t, latent_input.shape[0], latent_input.device)
        with step_forward_ctx(step_i, n_t):
            noise_pred = pipe.transformer(latent_input, timestep=ts, class_labels=class_labels_input).sample
        noise_pred = _merge_cfg_noise_pred(
            noise_pred,
            latent_channels,
            guidance_scale,
            duplicate_merged_eps_for_double_batch=True,
        )
        model_output = _scheduler_model_output(noise_pred, pipe, latent_channels)
        latents = pipe.scheduler.step(model_output, t, latents).prev_sample
    return latents


def _decode_latents_to_pil(pipe, latents: torch.Tensor, group_size: int, guidance_scale: float) -> list[Image.Image]:
    to_decode = latents[:group_size] if guidance_scale > 1 else latents
    to_decode = to_decode / pipe.vae.config.scaling_factor
    samples = pipe.vae.decode(to_decode).sample
    samples = (samples / 2 + 0.5).clamp(0, 1)
    samples_np = samples.cpu().permute(0, 2, 3, 1).float().numpy()
    images = pipe.numpy_to_pil(samples_np)
    return [img.resize((OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), Image.LANCZOS) for img in images]


@torch.no_grad()
def rollout_dit_same_class_pil_and_pooled_hidden(
    pipe,
    group_size: int,
    class_id: int,
    num_steps: int,
    device: str,
    seed: int,
    hidden_layer_idx: int,
    guidance_scale: float = 1.0,
) -> tuple[list[Image.Image], torch.Tensor]:
    """
    同一类别 c、group_size 条独立噪声轨迹；最后一步捕获指定层 GAP 表征 [N, D]。
    返回 (pil 列表, pooled_hidden)。

    使用 ``torch.no_grad()`` 而非 ``inference_mode()``：后者得到 inference tensor，
    不能作为 ranker 前向的输入参与对 ranker 参数的反传。
    """
    blocks = pipe.transformer.transformer_blocks
    assert 0 <= hidden_layer_idx < len(blocks)

    latents, class_labels_input, latent_channels = _cat_latents_same_class(
        pipe, group_size, device, seed, class_id, guidance_scale
    )
    capture: list[torch.Tensor | None] = [None]

    def hook_fn(_module, _inp, out):
        capture[0] = out.detach()

    def step_forward_ctx(step_i: int, n_steps: int) -> AbstractContextManager[None]:
        if step_i == n_steps - 1:
            return _forward_hook(blocks[hidden_layer_idx], hook_fn)
        return nullcontext()

    latents = _dit_denoise_loop(
        pipe, num_steps, latents, class_labels_input, latent_channels, guidance_scale, step_forward_ctx
    )

    hidden = capture[0]
    if hidden is None:
        raise RuntimeError("未能捕获中间层表征（请确认 deno_steps>=1）。")
    if guidance_scale > 1:
        hidden = hidden[: hidden.shape[0] // 2]
    pooled = hidden.float().mean(dim=1)

    images = _decode_latents_to_pil(pipe, latents, group_size, guidance_scale)
    return images, pooled


@torch.no_grad()
def rollout_dit_same_class_pil_and_all_layers_pooled_hidden(
    pipe,
    group_size: int,
    class_id: int,
    num_steps: int,
    device: str,
    seed: int,
    guidance_scale: float = 7.5,
) -> tuple[list[Image.Image], torch.Tensor]:
    """
    与 ``rollout_dit_same_class_pil_and_pooled_hidden`` 相同采样流程，但在最后一步对 **每一个**
    transformer block 注册 hook，一次性得到各层 token 输出并做 GAP。

    返回 (pil 列表, pooled_all)，``pooled_all`` 形状 ``[num_blocks, N, D]``（N=group_size）。
    """
    blocks = pipe.transformer.transformer_blocks
    n_blocks = len(blocks)

    latents, class_labels_input, latent_channels = _cat_latents_same_class(
        pipe, group_size, device, seed, class_id, guidance_scale
    )

    captures: list[torch.Tensor | None] = [None] * n_blocks

    def _make_hook(block_idx: int):
        def hook_fn(_module, _inp, out):
            captures[block_idx] = out.detach()

        return hook_fn

    hook_fns = [_make_hook(bi) for bi in range(n_blocks)]

    def step_forward_ctx(step_i: int, n_steps: int) -> AbstractContextManager[None]:
        if step_i == n_steps - 1:
            return _forward_hooks(list(blocks), hook_fns)
        return nullcontext()

    latents = _dit_denoise_loop(
        pipe, num_steps, latents, class_labels_input, latent_channels, guidance_scale, step_forward_ctx
    )

    pooled_layers: list[torch.Tensor] = []
    for bi in range(n_blocks):
        hidden = captures[bi]
        if hidden is None:
            raise RuntimeError(f"未能捕获 block {bi} 的表征（请确认 deno_steps>=1）。")
        if guidance_scale > 1:
            hidden = hidden[: hidden.shape[0] // 2]
        pooled_layers.append(hidden.float().mean(dim=1))

    pooled_all = torch.stack(pooled_layers, dim=0)
    images = _decode_latents_to_pil(pipe, latents, group_size, guidance_scale)
    return images, pooled_all


class CrossSampleDiffusionRanker(nn.Module):
    """
    论文 Sec. Ranker: x_i = W_proj h_i + b ∈ R^{d_r}；N 个 x_i 作为序列（**无位置编码**）
    过 M 层 Transformer Encoder，再 f_psi: x~_i -> s_i。PyTorch 的 TransformerEncoder 默认不注入位置编码。
    """

    def __init__(
        self,
        dit_hidden_dim: int,
        proj_dim: int = 64,
        num_tf_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if proj_dim % n_heads != 0:
            raise ValueError(f"proj_dim ({proj_dim}) must be divisible by n_heads ({n_heads})")
            
        self.proj = nn.Linear(dit_hidden_dim, proj_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * proj_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_tf_layers)
        self.score_head = nn.Linear(proj_dim, 1)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        """
        pooled_hidden: [N, D]，N 条候选的 GAP 表征（rollout 时同类；训练池 shuffle 后可为跨类混合 chunk）。
        返回 scores [N]（logits，未 softmax）。
        """
        x = self.proj(pooled_hidden).unsqueeze(0)
        x = self.encoder(x)
        s = self.score_head(x).squeeze(0).squeeze(-1)
        return s


def kl_reward_ranker_loss(pred_scores: torch.Tensor, teacher_scores: torch.Tensor, tau: float) -> torch.Tensor:
    """
    KL( q || p )，q=softmax(R/τ) 为 ImageReward 诱导的分布，p=softmax(s/τ) 为 ranker。
    对 pred_scores 的梯度与 listwise CE -sum q log p 一致（q 与 log q 项对 s 无梯度）。
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    q = F.softmax(teacher_scores / tau, dim=0)
    log_p = F.log_softmax(pred_scores / tau, dim=0)
    log_q = torch.log(q + 1e-8)
    return (q * (log_q - log_p)).sum()


def mse_kl_combined_loss(
    pred_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    tau: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (total, mse_term, kl_term)，total = alpha*mse + (1-alpha)*kl。"""
    mse = F.mse_loss(pred_scores, teacher_scores)
    kl = kl_reward_ranker_loss(pred_scores, teacher_scores, tau)
    a = float(alpha)
    total = a * mse + (1.0 - a) * kl
    return total, mse, kl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiT cross-sample ranker + ImageReward (MSE + KL)")
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--output_dir", type=str, default=str(_ROOT / "outputs" / "rollout_ranker_listwise"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="并行候选数 N（同类别、不同噪声）；论文常用 16，显存不足可调小",
    )
    p.add_argument("--deno_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DiT 半精度（默认开）；可用 --no-fp16 关闭",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=("none", "cosine", "cosine_warmup", "step", "plateau"),
        help="学习率调度：每个 epoch 末 step 一次；cosine_warmup 先用 LinearLR 再 CosineAnnealing",
    )
    p.add_argument(
        "--lr_warmup_epochs",
        type=int,
        default=0,
        help="仅 cosine_warmup：线性 warmup 的 epoch 数（若 >0 且选 cosine，会自动当作 cosine_warmup）",
    )
    p.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.01,
        help="cosine 的 eta_min = lr * lr_min_ratio",
    )
    p.add_argument(
        "--lr_step_size",
        type=int,
        default=10,
        help="step：每多少个 epoch 将 lr 乘以 gamma",
    )
    p.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR 的 gamma")
    p.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=5,
        help="plateau：验证 loss 无改善等待的 epoch 数",
    )
    p.add_argument(
        "--lr_plateau_factor",
        type=float,
        default=0.5,
        help="plateau：lr 衰减因子",
    )
    p.add_argument("--layer_ratio", type=float, default=0.6, help="相对深度 [0,1] → block 下标")
    p.add_argument("--hidden_layer_idx", type=int, default=None)
    p.add_argument("--ranker_proj_dim", type=int, default=512, help="d_r")
    p.add_argument("--ranker_tf_layers", type=int, default=2, help="跨样本 Transformer 层数 M")
    p.add_argument("--ranker_heads", type=int, default=4)
    p.add_argument("--ranker_ff_mult", type=int, default=4)
    p.add_argument("--listwise_tau", type=float, default=1.0, help="KL 中 q、p 的温度 τ")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="总损失中 MSE 权重；KL(softmax(R/τ) || softmax(s/τ)) 权重为 1-alpha",
    )
    p.add_argument("--prompt_template", type=str, default="a high-quality picture of a {name}")
    p.add_argument("--imagereward_device", type=str, default=None)
    p.add_argument(
        "--grad_accum_k",
        type=int,
        default=1,
        help="累积 k 个 shuffle 块（每块 group_size 条）后再 optimizer.step；反传 loss/k 以对齐全 k 时的平均梯度",
    )
    p.add_argument(
        "--cache_features_once",
        action="store_true",
        help="仅在第 1 个 epoch 做 DiT+ImageReward 预提取并缓存；之后各 epoch 复用同一 CPU 缓存，只训练 ranker",
    )
    p.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="每多少个 epoch 保存 ranker_diffusion_epoch_{epoch:04d}.pt，并同步覆盖 ranker_diffusion.pt；最后一 epoch 若未命中也会保存",
    )
    p.add_argument(
        "--num_rollouts",
        type=int,
        default=NUM_IMAGENET_CLASSES,
        help="每 epoch 做多少类 rollout（默认 1000=全类；调小可冒烟/调试）",
    )
    return p.parse_args()


def validate_training_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")
    if args.save_freq < 1:
        raise ValueError("--save_freq must be >= 1")
    if not (0.0 <= args.lr_min_ratio < 1.0):
        raise ValueError("--lr_min_ratio must be in [0, 1)")
    if args.lr_warmup_epochs < 0:
        raise ValueError("--lr_warmup_epochs must be >= 0")
    if args.lr_scheduler == "cosine" and args.lr_warmup_epochs > 0:
        args.lr_scheduler = "cosine_warmup"
    if args.lr_scheduler == "cosine_warmup" and args.lr_warmup_epochs >= args.epochs:
        raise ValueError("--lr_warmup_epochs must be < --epochs for cosine_warmup")
    if args.grad_accum_k < 1:
        raise ValueError("--grad_accum_k must be >= 1")
    if not (1 <= args.num_rollouts <= NUM_IMAGENET_CLASSES):
        raise ValueError("--num_rollouts must be in [1, 1000]")


def build_lr_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    eta_min = float(args.lr) * float(args.lr_min_ratio)
    name = args.lr_scheduler
    if name == "none":
        return None
    if name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)
    if name == "cosine_warmup":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        w = int(args.lr_warmup_epochs)
        if w <= 0:
            return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)
        t_cos = max(1, args.epochs - w)
        warmup_sched = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=w)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=t_cos, eta_min=eta_min)
        return SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[w])
    if name == "step":
        from torch.optim.lr_scheduler import StepLR

        return StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    if name == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
        )
    raise ValueError(f"unknown lr_scheduler: {name}")


def _save_ranker_checkpoint(
    out_dir: Path,
    ranker: nn.Module,
    meta: dict,
    args: argparse.Namespace,
    epoch_1based: int,
) -> None:
    payload = {
        "ranker_state_dict": ranker.state_dict(),
        "meta": meta,
        "config": vars(args),
        "epoch": epoch_1based,
    }
    ckpt_path = out_dir / f"ranker_diffusion_epoch_{epoch_1based:04d}.pt"
    torch.save(payload, ckpt_path)
    shutil.copy2(ckpt_path, out_dir / "ranker_diffusion.pt")
    print(f"  checkpoint saved -> {ckpt_path.name} (epoch {epoch_1based}); latest -> ranker_diffusion.pt")


def _try_plot_loss_curve(out_dir: Path, loss_history: list[dict], args: argparse.Namespace, hidden_layer_idx: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图；loss 已写入 loss_history.jsonl")
        return

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
    plt.title(
        f"Diffusion Ranker (α={args.alpha}, block {hidden_layer_idx}, "
        f"N={args.group_size}, M={args.ranker_tf_layers})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig_path = out_dir / "loss_curve.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {fig_path}")


def main() -> None:
    args = parse_args()
    validate_training_args(args)

    num_rollouts = int(args.num_rollouts)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ir_device = args.imagereward_device or str(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = model_loader(
        args.model_name,
        device,
        dtype="fp16" if args.fp16 else "fp32",
    )
    n_blocks = len(pipe.transformer.transformer_blocks)
    hidden_layer_idx = resolve_hidden_layer_idx(n_blocks, args.layer_ratio, args.hidden_layer_idx)
    dit_hidden_dim = int(pipe.transformer.inner_dim)

    meta = {
        "num_transformer_blocks": n_blocks,
        "hidden_layer_idx": hidden_layer_idx,
        "layer_ratio": args.layer_ratio,
        "dit_hidden_dim": dit_hidden_dim,
        "ranker_proj_dim": args.ranker_proj_dim,
        "ranker_tf_layers": args.ranker_tf_layers,
        "group_size": args.group_size,
        "listwise_tau": args.listwise_tau,
        "loss": "alpha*mse + (1-alpha)*kl_teacher_ranker",
        "alpha": args.alpha,
        "rollouts_per_epoch": num_rollouts,
        "train_chunks_per_epoch": num_rollouts,
        "flat_pool_size": num_rollouts * args.group_size,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({**meta, **vars(args)}, f, indent=2)

    cache_mode = "feature cache: once (reuse all epochs)" if args.cache_features_once else "feature cache: refresh each epoch"
    print(
        f"DiT blocks={n_blocks}, capture block {hidden_layer_idx} (layer_ratio={args.layer_ratio}); "
        f"ranker: d_r={args.ranker_proj_dim}, M={args.ranker_tf_layers}, N={args.group_size}, "
        f"loss=α·MSE+(1-α)·KL, α={args.alpha}; "
        f"{num_rollouts} rollouts/epoch -> pool {num_rollouts * args.group_size}, "
        f"{num_rollouts} train chunks/epoch (shuffled); "
        f"grad_accum_k={args.grad_accum_k}; save_freq={args.save_freq}; "
        f"lr_scheduler={args.lr_scheduler}; {cache_mode}"
    )

    labels_1k = load_imagenet_simple_labels()
    _ensure_imagereward_path()
    import ImageReward as RM

    teacher = RM.load("ImageReward-v1.0", device=ir_device)
    teacher.eval()

    ranker = CrossSampleDiffusionRanker(
        dit_hidden_dim=dit_hidden_dim,
        proj_dim=args.ranker_proj_dim,
        num_tf_layers=args.ranker_tf_layers,
        n_heads=args.ranker_heads,
        ff_mult=args.ranker_ff_mult,
    ).to(device)
    opt = torch.optim.AdamW(ranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(opt, args)

    loss_history: list[dict] = []
    global_step = 0
    optimizer_step = 0
    k_acc = args.grad_accum_k
    accum_in_block = 0
    persistent_feature_cache: dict | None = None

    def _do_optimizer_step(*, remainder: int | None = None) -> None:
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

    for epoch in range(args.epochs):
        epoch_losses: list[float] = []
        opt.zero_grad(set_to_none=True)
        accum_in_block = 0

        need_feature_extract = persistent_feature_cache is None or not args.cache_features_once
        if need_feature_extract:
            gs = args.group_size
            pool_h = torch.empty(num_rollouts * gs, dit_hidden_dim, dtype=torch.float32)
            pool_t = torch.empty(num_rollouts * gs, dtype=torch.float32)
            ext_pbar = tqdm(
                range(num_rollouts),
                desc=f"epoch {epoch+1}/{args.epochs} pre-extract (build flat pool)",
                leave=False,
            )
            for b in ext_pbar:
                batch_global = b if args.cache_features_once else epoch * num_rollouts + b
                seed = args.seed + batch_global * ROLLOUT_GLOBAL_SEED_MULTIPLIER
                class_id = int(b)

                pil_list, pooled_h = rollout_dit_same_class_pil_and_pooled_hidden(
                    pipe,
                    gs,
                    class_id,
                    args.deno_steps,
                    str(device),
                    seed,
                    hidden_layer_idx,
                    guidance_scale=args.guidance_scale,
                )

                prompt = prompt_for_class(labels_1k, class_id, args.prompt_template)
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
        gs = args.group_size
        if pool_size % gs != 0:
            raise RuntimeError(f"pool_size={pool_size} 不能被 group_size={gs} 整除")
        n_train_chunks = pool_size // gs

        rng_perm = torch.Generator()
        rng_perm.manual_seed(int(args.seed) + epoch)
        perm = torch.randperm(pool_size, generator=rng_perm)

        train_pbar = tqdm(
            range(n_train_chunks),
            desc=f"epoch {epoch+1}/{args.epochs} train ranker (shuffled pool)",
            leave=True,
        )
        for bi in train_pbar:
            sl = perm[bi * gs : (bi + 1) * gs]
            pooled_h_batch = pool_h[sl].to(device, non_blocking=True)
            teacher_t_batch = pool_t[sl].to(device, non_blocking=True)

            ranker.train()
            pred_scores = ranker(pooled_h_batch)
            loss, mse_term, kl_term = mse_kl_combined_loss(
                pred_scores, teacher_t_batch, args.listwise_tau, args.alpha
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
                chunk=bi,
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
                "accum_in_block_after_backward": accum_in_block,
                "did_optimizer_step": stepped_now,
            }
            loss_history.append(row)
            with open(out_dir / "loss_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        if accum_in_block > 0:
            _do_optimizer_step(remainder=accum_in_block)

        mean_ep = sum(epoch_losses) / max(len(epoch_losses), 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1} mean total loss (α·MSE+(1-α)·KL): {mean_ep:.6f} | lr={lr_now:.2e}")

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(mean_ep)
            else:
                scheduler.step()

        ep1 = epoch + 1
        if ep1 % args.save_freq == 0 or ep1 == args.epochs:
            _save_ranker_checkpoint(out_dir, ranker, meta, args, ep1)

    _try_plot_loss_curve(out_dir, loss_history, args, hidden_layer_idx)


if __name__ == "__main__":
    main()
