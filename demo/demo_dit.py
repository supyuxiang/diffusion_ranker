#!/usr/bin/env python3
"""
DiT (Diffusion Transformer) 演示：类别条件图像生成
支持离线加载（local_files_only=True）

额外：``--extract_hidden`` 对 **单次** transformer 前向（随机噪声 latent + 调度器中间时刻）
在指定 ``transformer_blocks[layer_idx]`` **之后** 注册 hook，导出该层输出的 patch-token 表征（形状约为 ``(B, N_patches, dim)``），便于下游探针 / ranker 实验。这不是整段去噪轨迹的聚合。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import DiTPipeline
from diffusers.utils.torch_utils import randn_tensor

from utils import Constants_default, model_loader, set_device, set_seed







@torch.inference_mode()
def demo_extract_transformer_block_hidden(
    pipe: DiTPipeline,
    *,
    class_id: int,
    layer_idx: int,
    num_inference_steps: int,
    device: torch.device,
    seed: int,
    out_pt: Path,
) -> torch.Size:
    """一次 ``pipe.transformer`` 前向，在某一 DiT block 输出处抓取 hidden states并 ``torch.save``。"""
    tf = pipe.transformer
    n_blocks = len(tf.transformer_blocks)
    if layer_idx < 0:
        layer_idx = n_blocks // 2
    if not (0 <= layer_idx < n_blocks):
        raise ValueError(f"layer_idx must be in [0, {n_blocks - 1}], got {layer_idx}")

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out: torch.Tensor) -> None:
        captured["h"] = out.detach().float().cpu()

    handle = tf.transformer_blocks[layer_idx].register_forward_hook(_hook)
    try:
        latent_channels = tf.config.in_channels
        latent_size = tf.config.sample_size
        gen = torch.Generator(device=device).manual_seed(seed)
        latents = randn_tensor(
            (1, latent_channels, latent_size, latent_size),
            generator=gen,
            device=device,
            dtype=tf.dtype,
        )
        pipe.scheduler.set_timesteps(num_inference_steps)
        mid = min(num_inference_steps - 1, num_inference_steps // 2)
        t = pipe.scheduler.timesteps[mid]
        latent_model_input = pipe.scheduler.scale_model_input(latents, t)
        timesteps = torch.tensor([t], dtype=torch.long, device=device).expand(latent_model_input.shape[0])
        labels = torch.tensor([class_id], dtype=torch.long, device=device)

        _ = tf(
            latent_model_input,
            timestep=timesteps,
            class_labels=labels,
            return_dict=True,
        )
    finally:
        handle.remove()

    if "h" not in captured:
        raise RuntimeError("Forward hook did not run; check diffusers / DiT version.")

    h = captured["h"]
    t_int = int(t) if not torch.is_tensor(t) else int(t.item())
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_states_after_block": h,
            "shape": list(h.shape),
            "layer_idx": layer_idx,
            "n_transformer_blocks": n_blocks,
            "class_id": class_id,
            "scheduler_step_index": mid,
            "timestep": t_int,
            "num_inference_steps": num_inference_steps,
            "note": (
                "Single forward: random latent, mid schedule timestep. "
                "Tensor is patch sequence before final norm/proj_out in DiT."
            ),
        },
        out_pt,
    )
    return h.shape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiT demo: sampling + optional hidden extract")
    p.add_argument("--class_id", type=int, default=666)
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument(
        "--extract_hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one transformer forward with a hook and save hidden states (default: on).",
    )
    p.add_argument(
        "--hidden_layer",
        type=int,
        default=-1,
        help="transformer_blocks index; -1 = middle block.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = set_device()
    pipe_dit = model_loader(Constants_default["model_name"], device, dtype="fp16")
    seed = Constants_default["seed"]
    set_seed(seed)

    class_id = args.class_id
    num_images = args.num_images
    num_inference_steps = args.num_inference_steps
    output_dir = Path(__file__).parent / "outputs_demo" / f"class{class_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(seed)
    labels = [class_id] * num_images

    images = pipe_dit(
        labels,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images

    for i, img in enumerate(images):
        path = output_dir / f"dit_class{class_id}_{i}.png"
        img.save(path)
        print(f"  保存: {path}")

    if args.extract_hidden:
        n_bl = len(pipe_dit.transformer.transformer_blocks)
        resolved_layer = args.hidden_layer if args.hidden_layer >= 0 else n_bl // 2
        pt_path = output_dir / f"dit_hidden_after_block_{resolved_layer}_demo.pt"
        try:
            shp = demo_extract_transformer_block_hidden(
                pipe_dit,
                class_id=class_id,
                layer_idx=args.hidden_layer,
                num_inference_steps=num_inference_steps,
                device=device,
                seed=seed + 1,
                out_pt=pt_path,
            )
            print(f"\n[hidden demo] saved {pt_path} tensor shape {tuple(shp)}")
        except Exception as e:
            print(f"\n[hidden demo] skipped: {type(e).__name__}: {e}")

    print(f"\n完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()
