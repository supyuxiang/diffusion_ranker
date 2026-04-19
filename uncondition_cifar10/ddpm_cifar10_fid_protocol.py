#!/usr/bin/env python3
"""
复现 Hugging Face ``google/ddpm-cifar10-32`` 讨论 #10 中的评测习惯（见下）。

- **speiqin**：EMA 权重 + **1000 步 DDPM** 曾报 FID≈5.23；后说明若 FID 只用 **1 万张** 图会偏差，
  改为 **5 万张** 后与论文更接近。
- **Vinnnf**：提到修改 diffusers 与 `GitHub huggingface/diffusers#2585` 相关的问题（**DDIM 时间步**），
  并给出 Diff-Pruning 里的采样脚本；其 **FID≈4.5** 的结果是 **100 步 DDIM + 50k 样本**。
  Issue #2585 讨论的是 **DDIMScheduler.set_timesteps**，与 **DDPMPipeline 走满训练步数 T=1000**
  的离散 DDPM 采样不是同一条代码路径；若你只用 ``DDPMPipeline`` + ``num_inference_steps=1000``，
  一般**不**需要为 #2585 单独打补丁。

EMA 权重需自行放到本地 diffusers 目录（例如从 ``pesser/pytorch_diffusion`` 或
``VainF/Diff-Pruning`` 发布的 ``ddpm_ema_cifar10`` 转换结果），再通过 ``--model_path`` 指定。
Hub 默认 ``google_ddpm-cifar10-32`` 多为 **非 EMA**，FID 常在 ~12 量级（讨论帖）。

Example（50k 生成 vs 50k train，对齐常见论文 FID 设定）:

  CUDA_VISIBLE_DEVICES=0 python /root/diffusion_ranker/uncondition_cifar10/ddpm_cifar10_fid_protocol.py \\
      --model_path /root/autodl-tmp/ddpm_ema_cifar10 \\
      --deno_steps 1000 \\
      --num_gen 50000 \\
      --real_split train \\
      --save_dir /root/autodl-tmp/outputs/ddpm_cifar10_ema_fid

非 EMA 快照快速自检（步数少、张数少）::

  CUDA_VISIBLE_DEVICES=0 python .../ddpm_cifar10_fid_protocol.py \\
      --model_path /root/autodl-tmp/google_ddpm-cifar10-32 \\
      --deno_steps 50 --num_gen 512 --real_split test --batch_size 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

_ROOT_ = Path(__file__).resolve().parent.parent
if str(_ROOT_) not in sys.path:
    sys.path.insert(0, str(_ROOT_))

from uncondition_cifar10.cifar10_scaling_with_different_steps import (
    _load_images_from_dir_uint8,
    _resolve_cifar10_torchvision_root,
)
from utils import (
    compute_fid,
    compute_is,
    ddpm_loader,
    rollout_ddpm_with_steps,
    set_device,
    set_seed,
)


def _load_cifar10_split_uint8(cifar_root: Path, *, train: bool) -> torch.Tensor:
    ds = CIFAR10(root=str(cifar_root), train=train, download=False)
    arr = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    return arr.to(torch.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CIFAR-10 DDPM：指定本地模型目录 + 步数 + 张数，算 FID/IS（可对齐 HF #10 讨论）",
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="本地 diffusers 快照目录（含 model_index.json），如 EMA 转换目录或 google_ddpm-cifar10-32",
    )
    p.add_argument("--deno_steps", type=int, default=1000, help="num_inference_steps（DDPM 常为 1000）")
    p.add_argument("--num_gen", type=int, default=50_000, help="生成张数；FID 讨论中建议与统计量一致用 50k")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/outputs/ddpm_cifar10_fid_protocol",
    )
    p.add_argument(
        "--real_split",
        type=str,
        choices=("train", "test"),
        default="train",
        help="FID 参考集：train=50k（常见论文）；test=10k（与 cifar10_scaling_with_different_steps 默认一致）",
    )
    p.add_argument("--data_root", type=str, default="/root/autodl-pub/cifar-10")
    p.add_argument("--extract_cache", type=str, default="/root/autodl-tmp/cifar-10_from_autodl_pub")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device()

    model_path = Path(args.model_path).expanduser().resolve()
    if not (model_path / "model_index.json").is_file():
        raise FileNotFoundError(f"不是有效的 diffusers 目录: {model_path}")

    cifar_root = _resolve_cifar10_torchvision_root(
        Path(args.data_root),
        Path(args.extract_cache),
    )
    train_mode = args.real_split == "train"
    real_images = _load_cifar10_split_uint8(cifar_root, train=train_mode)
    print(
        f"[info] real CIFAR-10 split={args.real_split!r} shape={tuple(real_images.shape)} "
        f"(train=50k, test=10k)"
    )

    dtype = args.dtype
    if dtype == "bf16":
        print("[warn] bf16 在 DDPMPipeline numpy 路径上常不可用，改用 fp16。")
        dtype = "fp16"

    pipe = ddpm_loader(model_path, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    out_dir = Path(args.save_dir).expanduser().resolve() / f"ddpm_steps{args.deno_steps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[info] rollout: deno_steps={args.deno_steps} num_gen={args.num_gen} -> {out_dir}"
    )
    rollout_ddpm_with_steps(
        pipe,
        out_dir,
        args.deno_steps,
        args.num_gen,
        args.batch_size,
        device,
        seed=args.seed,
        image_size=32,
    )

    gen_images = _load_images_from_dir_uint8(out_dir)
    if gen_images.shape[0] != args.num_gen:
        print(
            f"[warn] 目录内图像数 {gen_images.shape[0]} 与 --num_gen {args.num_gen} 不一致，"
            "请检查是否被其它文件干扰或生成中断。"
        )

    fid = float(compute_fid(real_images, gen_images, device=device))
    is_mean, is_std = compute_is(gen_images, device=device)
    metrics = {"fid": fid, "is": float(is_mean), "is_std": float(is_std)}
    meta = {
        "model_path": str(model_path),
        "deno_steps": args.deno_steps,
        "num_gen": int(args.num_gen),
        "real_split": args.real_split,
        "real_shape": list(real_images.shape),
        "gen_shape": list(gen_images.shape),
        "note_issue_2585": (
            "diffusers#2585 concerns DDIM timestep spacing; full-step DDPM (this script) is a different path."
        ),
    }
    out_json = out_dir / "metrics_fid_protocol.json"
    out_json.write_text(
        json.dumps({"metrics": metrics, "meta": meta}, indent=2),
        encoding="utf-8",
    )
    print(f"[done] metrics={metrics}")
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
