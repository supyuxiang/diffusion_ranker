#!/usr/bin/env python3
"""
演示：在 **完整 DiTPipeline 采样**（``pipe(class_labels, ...)``）过程中，用 ``register_forward_hook``
抓取 ``transformer_blocks[layer_idx]`` 每次前向的输出，**无需手写 scheduler 循环**。

说明：
- DiT 每去噪步会调用一次 ``pipe.transformer``；hook 会在**每一步**触发，列表长度 ≈ ``num_inference_steps``。
- 若 ``guidance_scale > 1``，实现里可能对 batch 做 double，forward 次数可能多于步数，以实际 ``len(captures)`` 为准。
- 正式 rollout 若对 ``pipe.transformer`` 使用 ``torch.compile``，与 hook 同开时需自行验证；本 demo **不** 开 compile，便于调试。

运行（仓库根目录）::

  python3 demo/demo_dit_pipeline_representation_hook.py \\
      --num_inference_steps 5 --num_images 2 --layer_idx 14

依赖与 ``rollout.py`` / ``demo_dit.py`` 相同：本地 DiT 权重、``utils.model_loader``。
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

from utils import Constants_default, model_loader, set_device, set_seed


def _block_output_tensor(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"Unexpected block output type: {type(out)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--class_id", type=int, default=281, help="ImageNet class id（默认 tabby cat）")
    p.add_argument("--num_images", type=int, default=2)
    p.add_argument("--num_inference_steps", type=int, default=5)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--dtype", default="fp16", choices=("fp32", "fp16", "bf16"))
    p.add_argument(
        "--layer_idx",
        type=int,
        default=-1,
        help="transformer_blocks 下标；-1 表示中间层",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save_pt",
        type=str,
        default="",
        help="若非空，将 captures 存为 torch.save（list[Tensor]，可能较大）",
    )
    args = p.parse_args()

    device = set_device()
    set_seed(args.seed)

    pipe = model_loader(Constants_default["model_name"], device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)

    tf = pipe.transformer
    n_blocks = len(tf.transformer_blocks)
    layer_idx = args.layer_idx if args.layer_idx >= 0 else n_blocks // 2
    if not (0 <= layer_idx < n_blocks):
        raise SystemExit(f"layer_idx must be in [0, {n_blocks - 1}], got {layer_idx}")

    captures: list[torch.Tensor] = []
    step_counter = {"i": 0}

    def forward_hook(_module, _inp, out) -> None:
        h = _block_output_tensor(out)
        captures.append(h.detach().float().cpu())
        step_counter["i"] += 1

    block = tf.transformer_blocks[layer_idx]
    handle = block.register_forward_hook(forward_hook)

    try:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        labels = [args.class_id] * args.num_images

        print(
            f"Running pipe: batch={args.num_images}, steps={args.num_inference_steps}, "
            f"guidance_scale={args.guidance_scale}, hook on transformer_blocks[{layer_idx}]"
        )
        _ = pipe(
            labels,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
    finally:
        handle.remove()

    print(f"Hook fired {len(captures)} times (transformer forwards).")
    for i, t in enumerate(captures[:3]):
        print(f"  capture[{i}] shape = {tuple(t.shape)}")
    if len(captures) > 3:
        print(f"  ... capture[{len(captures) - 1}] shape = {tuple(captures[-1].shape)}")

    # 示例：对最后一步的 patch 维做 GAP，得到每张图一个向量
    last = captures[-1]
    if last.dim() == 3:
        pooled = last.mean(dim=1)
        print(f"Last capture GAP -> {tuple(pooled.shape)} (batch, dim)")

    if args.save_pt:
        out_path = Path(args.save_pt).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "captures": captures,
                "layer_idx": layer_idx,
                "n_blocks": n_blocks,
                "class_id": args.class_id,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "num_hook_fires": len(captures),
            },
            out_path,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
