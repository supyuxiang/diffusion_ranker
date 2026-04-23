#!/usr/bin/env python3
"""
从 DiT 的最后一个去噪步提取每个 token 的 representation，然后取平均
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor


def load_dit(offline: bool = True):
    dit_path = Path("/data/home/yxfeng/data/models/dit_model")
    
    pipe = DiTPipeline.from_pretrained(
        str(dit_path), local_files_only=offline, torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def extract_last_step_token_repr(
    pipe: DiTPipeline,
    class_id: int,
    num_images: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 1.0,
    seed: int = 42,
):
    """
    运行 DiT 生成，在最后一个去噪步提取每个 token 的 hidden state，返回 mean-pooled 表示。
    返回: (images, token_repr_mean)
    - images: 生成的图像 list
    - token_repr_mean: [num_images, hidden_dim] 每个样本的 token 平均表示
    """
    device = pipe._execution_device
    batch_size = num_images
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels
    inner_dim = pipe.transformer.inner_dim  # hidden size per token

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = randn_tensor(
        (batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=pipe.transformer.dtype,
    )
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

    class_labels = torch.tensor([class_id] * batch_size, device=device)
    class_null = torch.tensor([1000] * batch_size, device=device)
    class_labels_input = (
        torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels
    )

    # 用于存储最后一个去噪步的 token 表示
    last_step_hidden = [None]

    def capture_hook(module, input, output):
        """捕获最后一个 transformer block 输出的 hidden states"""
        last_step_hidden[0] = output.detach().float()

    handle = pipe.transformer.transformer_blocks[-1].register_forward_hook(capture_hook)

    pipe.scheduler.set_timesteps(num_inference_steps)
    for t in pipe.scheduler.timesteps:
        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t if torch.is_tensor(t) else torch.tensor([t], device=device, dtype=torch.long)
        if len(timesteps.shape) == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.to(device).expand(latent_model_input.shape[0])

        noise_pred = pipe.transformer(
            latent_model_input, timestep=timesteps, class_labels=class_labels_input
        ).sample

        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            noise_pred = torch.cat([eps, rest], dim=1)

        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        latent_model_input = pipe.scheduler.step(model_output, t, latent_model_input).prev_sample

    handle.remove()

    if guidance_scale > 1:
        latents, _ = latent_model_input.chunk(2, dim=0)
    else:
        latents = latent_model_input

    # last_step_hidden[0]: [B, num_tokens, inner_dim]，若 guidance>1 则 B=2*batch_size
    hidden = last_step_hidden[0]
    if guidance_scale > 1:
        hidden = hidden[: hidden.shape[0] // 2]  # 只取 conditioned 部分

    # 对每个样本，对所有 token 取平均
    token_repr_mean = hidden.mean(dim=1)  # [num_images, inner_dim]

    # 解码图像
    latents = 1 / pipe.vae.config.scaling_factor * latents
    samples = pipe.vae.decode(latents).sample
    samples = (samples / 2 + 0.5).clamp(0, 1)
    images = pipe.numpy_to_pil(samples.detach().cpu().permute(0, 2, 3, 1).float().numpy())

    return images, token_repr_mean


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_id", type=int, default=207)
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--output_dir", default="/data/home/yxfeng/project1/idm/dit_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", default=True)
    args = parser.parse_args()

    print("加载 DiT 模型...")
    pipe = load_dit(offline=args.offline)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    print(f"生成 {args.num_images} 张图像，提取最后去噪步的 token 表示并取平均...")
    images, token_repr_mean = extract_last_step_token_repr(
        pipe,
        class_id=args.class_id,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=1.0,
        seed=args.seed,
    )

    print(f"\nToken 表示形状: {token_repr_mean.shape}  (num_images, hidden_dim)")
    print(f"Hidden dim: {token_repr_mean.shape[-1]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = out_dir / f"dit_class{args.class_id}_repr_{i:02d}.png"
        img.save(path)
        print(f"  保存图像: {path}")

    # 保存 token 表示
    repr_path = out_dir / f"dit_class{args.class_id}_token_repr_mean.pt"
    torch.save({"token_repr_mean": token_repr_mean, "class_id": args.class_id}, repr_path)
    print(f"  保存表示: {repr_path}")
    print(f"\n完成！")


if __name__ == "__main__":
    main()
