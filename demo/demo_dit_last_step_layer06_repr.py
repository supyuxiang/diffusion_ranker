#!/usr/bin/env python3
r"""
演示：用 ``DiTPipeline`` 生成图像时，只抓取 **最后一个去噪步** 上 **layer ratio = 0.6**
对应 block 的输出，并得到与 rollout 一致的 ``mean`` / ``last`` 池化向量（与 ``DitRolloutReprCapture`` 相同逻辑）。

运行（仓库根目录）::

  CUDA_VISIBLE_DEVICES=0 python3 demo/demo_dit_last_step_layer06_repr.py \\
    --dit_path /root/autodl-tmp/DiT-XL-2-256 \\
    --num_inference_steps 10 \\
    --batch_size 4

说明：
- ``step_ratio_ls=[1.0]`` → 目标步下标 ``round(1.0 * (T-1)) = T-1``，即最后一步。
- ``layer_ratios=(0.6,)`` → ``round(0.6 * (n_blocks - 1))`` 对应 DiT block。
- 字典键格式与 ``utils.DitRolloutReprCapture`` 一致，例如 ``s9_r0.6``（当 ``T=10`` 时）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

from utils import DitRolloutReprCapture, dit_loader, set_device, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="DiT 末步 layer=0.6 表征 demo")
    p.add_argument(
        "--dit_path",
        type=str,
        default=str(Path("/root/autodl-tmp/DiT-XL-2-256")),
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DiT 用 fp16（关则用 fp32）",
    )
    p.add_argument(
        "--layer_ratio",
        type=float,
        default=0.6,
        help="层比例（与训练导出 step_X_layer_0.6 一致）",
    )
    p.add_argument(
        "--step_ratio",
        type=float,
        default=1.0,
        help="去噪步比例；1.0 = 最后一步",
    )
    args = p.parse_args()

    dit_path = Path(args.dit_path).expanduser().resolve()
    if not dit_path.is_dir():
        raise SystemExit(f"DiT 目录不存在: {dit_path}")

    set_seed(args.seed)
    device = set_device()
    dtype_s = "fp16" if args.fp16 else "fp32"
    store_dtype = torch.float16 if args.fp16 else torch.float32

    pipe = dit_loader(dit_path, device, dtype=dtype_s)
    pipe.set_progress_bar_config(disable=True)

    T = args.num_inference_steps
    last_step_idx = min(T - 1, max(0, round(float(args.step_ratio) * (T - 1))))
    n_blocks = len(pipe.transformer.transformer_blocks)
    layer_block_idx = int(round(float(args.layer_ratio) * (n_blocks - 1))) # 7
    layer_block_idx = max(0, min(n_blocks - 1, layer_block_idx)) # 7

    print(
        f"DiT blocks={n_blocks}  |  抓取 layer_ratio={args.layer_ratio} → block_idx={layer_block_idx}"
    )
    print(
        f"num_inference_steps={T}  |  step_ratio={args.step_ratio} → step_idx={last_step_idx}（末步应为 {T - 1}）"
    )

    repr_cap = DitRolloutReprCapture(
        pipe,
        num_inference_steps=T,
        guidance_scale=args.guidance_scale,
        layer_ratios=(float(args.layer_ratio),),
        step_ratio_ls=[float(args.step_ratio)],
        store_dtype=store_dtype,
        rollout_base_seed=args.seed,
    )
    repr_cap.install()

    labels = torch.randint(0, 1000, (args.batch_size,), device=device).tolist()
    gen = torch.Generator(device=device).manual_seed(args.seed)

    try:
        repr_cap.set_capturing(True)
        repr_cap.begin_batch(labels)
        out = pipe(
            class_labels=labels,
            num_inference_steps=T,
            guidance_scale=args.guidance_scale,
            generator=gen,
        )
        repr_cap.end_batch(0, 0, generator_seed=args.seed)
        rec = repr_cap.records[-1]
        repr_dict = rec["repr"]
    finally:
        repr_cap.remove()

    print(f"生成图像数: {len(out.images)}  repr 键: {list(repr_dict.keys())}")

    for k, v in repr_dict.items():
        mean_t = v["mean"]
        last_t = v["last"]
        print(
            f"  [{k}]  mean shape={tuple(mean_t.shape)} dtype={mean_t.dtype}  |  "
            f"last shape={tuple(last_t.shape)}"
        )
        # 示例：单条样本的 mean 向量前 5 维
        print(f"        mean[0,:5] = {mean_t[0, :5].float()}")

    print("\n完成。可将 repr_dict[k]['mean'] 作为与训练 ``mean/step_*_layer_0.6`` 同语义的表征。")


if __name__ == "__main__":
    main()
