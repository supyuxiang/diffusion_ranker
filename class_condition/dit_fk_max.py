#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from class_condition.reward import labels_1k, load_clip, load_imagereward
from utils import dit_loader, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DiT FK-Steering (max potential) for ImageNet class-conditional generation")
    p.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/class_condition/dit_fk_max_steps10_ps8_mc5",
    )
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=50_000, help="50/class protocol uses 50,000 samples")
    p.add_argument("--batch_size", type=int, default=2, help="number of sample slots per round (bz)")
    p.add_argument("--parallel_size", type=int, default=8, help="number of particles per sample slot (K)")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0, help="currently only supports 1.0")
    p.add_argument("--resample_step_idxs", type=str, default="1,6", help="comma-separated denoise step indices")
    p.add_argument("--fk_lambda", type=float, default=10.0)
    p.add_argument("--mc", type=int, default=5, help="MC rollout count per particle at each resample step")
    p.add_argument("--label_reward", type=str, default="imagereward", choices=("imagereward", "clipscore"))
    p.add_argument("--save_all_particles", action="store_true", help="save final K particles for each sample slot")
    return p.parse_args()


def _parse_step_indices(s: str, max_steps: int) -> list[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= max_steps:
            raise ValueError(f"resample step idx {idx} out of range [0, {max_steps - 1}]")
        vals.append(idx)
    if not vals:
        raise ValueError("resample_step_idxs is empty")
    return sorted(set(vals))


def _build_labels_50k_class_major(num_samples: int) -> list[int]:
    labels = [c for c in range(1000) for _ in range(50)]
    if num_samples != 50_000:
        if num_samples > len(labels):
            raise ValueError(f"num_samples={num_samples} > 50,000 class-major protocol size")
        labels = labels[:num_samples]
    return labels


def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        dt = torch.float64 if isinstance(t, float) else torch.int64
        timesteps = torch.tensor([t], dtype=dt, device=latent_model_input.device)
    elif len(t.shape) == 0:
        timesteps = t[None].to(latent_model_input.device)
    else:
        timesteps = t
    return timesteps.expand(latent_model_input.shape[0])


def _scheduler_step_with_optional_noise(
    scheduler,
    model_output: torch.Tensor,
    t,
    sample: torch.Tensor,
    generator: torch.Generator | None,
    add_variance_noise: bool,
) -> torch.Tensor:
    if add_variance_noise:
        variance_noise = torch.randn(
            sample.shape,
            generator=generator,
            device=sample.device,
            dtype=sample.dtype,
        )
        try:
            out = scheduler.step(
                model_output, t, sample, generator=generator, variance_noise=variance_noise
            )
            return out.prev_sample
        except TypeError:
            pass
    out = scheduler.step(model_output, t, sample, generator=generator)
    return out.prev_sample


def _denoise_one_step(
    *,
    pipe,
    scheduler,
    x: torch.Tensor,
    labels_flat: torch.Tensor,
    t,
    latent_channels: int,
    generator: torch.Generator | None = None,
    add_variance_noise: bool = False,
) -> torch.Tensor:
    x_in = scheduler.scale_model_input(x, t)
    t_b = _broadcast_timesteps(t, x_in)
    noise_pred = pipe.transformer(x_in, timestep=t_b, class_labels=labels_flat).sample
    if pipe.transformer.config.out_channels // 2 == latent_channels:
        model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
    else:
        model_output = noise_pred
    return _scheduler_step_with_optional_noise(
        scheduler=scheduler,
        model_output=model_output,
        t=t,
        sample=x,
        generator=generator,
        add_variance_noise=add_variance_noise,
    )


@torch.inference_mode()
def _decode_latents_to_images(pipe, x: torch.Tensor) -> list[Image.Image]:
    lat = 1 / pipe.vae.config.scaling_factor * x
    samples = pipe.vae.decode(lat).sample
    samples = (samples / 2 + 0.5).clamp(0, 1)
    out: list[Image.Image] = []
    for i in range(samples.shape[0]):
        arr = samples[i].detach().cpu().permute(1, 2, 0).float().numpy()
        out.append(Image.fromarray((arr * 255.0).round().clip(0, 255).astype("uint8")))
    return out


def _prepare_reward_context(reward_type: str, device: torch.device) -> dict:
    if reward_type == "imagereward":
        return {"type": reward_type, "model": load_imagereward(device)}
    if reward_type == "clipscore":
        clip_model, clip_processor = load_clip(device)
        return {"type": reward_type, "model": clip_model, "processor": clip_processor}
    raise ValueError(f"unsupported reward_type={reward_type}")


@torch.inference_mode()
def _score_images(
    images: list[Image.Image],
    labels: list[int],
    reward_ctx: dict,
    device: torch.device,
) -> torch.Tensor:
    if len(images) != len(labels):
        raise ValueError(f"images size {len(images)} != labels size {len(labels)}")
    reward_type = reward_ctx["type"]
    prompts = [f"A photo of a {labels_1k[int(lab)]}." for lab in labels]
    if reward_type == "imagereward":
        rm = reward_ctx["model"]
        vals = [float(rm.score(p, img)) for p, img in zip(prompts, images)]
        return torch.tensor(vals, dtype=torch.float32)
    if reward_type == "clipscore":
        clip_model = reward_ctx["model"]
        clip_processor = reward_ctx["processor"]
        inputs = clip_processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = clip_model(**inputs)
        img_f = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_f = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (img_f * txt_f).sum(dim=-1)
        return sim.detach().float().cpu()
    raise ValueError(f"unsupported reward_type={reward_type}")


def _gather_matrix_by_idx(mat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(mat, dim=1, index=idx)


def _resample_by_potential(
    x: torch.Tensor,
    potential_rewards: torch.Tensor,
    fk_lambda: float,
    bsz: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    logits = fk_lambda * potential_rewards
    logits = logits - logits.max(dim=1, keepdim=True).values
    w = torch.exp(logits).clamp(min=1e-20)
    probs = w / w.sum(dim=1, keepdim=True)

    idx_rows = []
    for bi in range(bsz):
        idx_rows.append(torch.multinomial(probs[bi], num_samples=k, replacement=True))
    idx = torch.stack(idx_rows, dim=0)

    x_view = x.view(bsz, k, *x.shape[1:])
    gather_index = idx.view(bsz, k, 1, 1, 1).expand(-1, -1, x.shape[1], x.shape[2], x.shape[3])
    x_resampled = torch.gather(x_view, dim=1, index=gather_index).reshape(bsz * k, *x.shape[1:])

    ess = (1.0 / (probs.pow(2).sum(dim=1))).mean().item()
    diag = {
        "weights_max_mean": float(probs.max(dim=1).values.mean().item()),
        "weights_min_mean": float(probs.min(dim=1).values.mean().item()),
        "ess_mean": float(ess),
        "potential_mean": float(potential_rewards.mean().item()),
    }
    return x_resampled, idx, diag


@torch.inference_mode()
def _mc_rollout_rewards(
    *,
    pipe,
    scheduler_state_after_step,
    x_after_step: torch.Tensor,
    labels_flat: torch.Tensor,
    timesteps,
    current_step_idx: int,
    mc: int,
    reward_ctx: dict,
    device: torch.device,
    seed_base: int,
) -> torch.Tensor:
    bk = x_after_step.shape[0]
    latent_channels = pipe.transformer.config.in_channels
    labels_list = [int(x) for x in labels_flat.detach().cpu().tolist()]
    reward_sum = torch.zeros(bk, dtype=torch.float32)

    for mi in range(int(mc)):
        scheduler_mc = copy.deepcopy(scheduler_state_after_step)
        x = x_after_step.clone()
        gen_mc = torch.Generator(device=device).manual_seed(int(seed_base + mi * 1_000_003))

        for st in range(current_step_idx + 1, len(timesteps)):
            t = timesteps[st]
            x = _denoise_one_step(
                pipe=pipe,
                scheduler=scheduler_mc,
                x=x,
                labels_flat=labels_flat,
                t=t,
                latent_channels=latent_channels,
                generator=gen_mc,
                add_variance_noise=True,
            )

        images = _decode_latents_to_images(pipe, x)
        reward_sum += _score_images(images, labels_list, reward_ctx, device)
        del scheduler_mc, x, images

    return reward_sum / float(mc)


def main() -> None:
    args = parse_args()
    if float(args.guidance_scale) != 1.0:
        raise ValueError("Current implementation supports guidance_scale=1.0 only")
    if int(args.parallel_size) <= 0 or int(args.batch_size) <= 0:
        raise ValueError("parallel_size and batch_size must be positive")
    if int(args.mc) <= 0:
        raise ValueError("mc must be positive")

    set_seed(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    best_dir = output_dir / "best_images"
    particle_dir = output_dir / "all_particles"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    if args.save_all_particles:
        particle_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = dit_loader(args.model_name, device=device, dtype=args.dtype)
    pipe.set_progress_bar_config(disable=True)
    reward_ctx = _prepare_reward_context(str(args.label_reward), device)

    labels_all = _build_labels_50k_class_major(int(args.num_samples))
    num_rounds = (len(labels_all) + int(args.batch_size) - 1) // int(args.batch_size)
    resample_steps = _parse_step_indices(str(args.resample_step_idxs), int(args.num_inference_steps))
    if len(resample_steps) < 2:
        raise ValueError("max potential run expects at least two resample steps (e.g., 1,6)")

    img_counter = 0
    final_rewards_all: list[float] = []
    fk_logs: list[dict] = []
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels
    k = int(args.parallel_size)
    first_resample_step = int(resample_steps[0])

    pbar = tqdm(range(num_rounds), desc="FK max 50k")
    for r in pbar:
        st = r * int(args.batch_size)
        ed = min(st + int(args.batch_size), len(labels_all))
        labels_this = labels_all[st:ed]
        bsz = len(labels_this)
        if bsz <= 0:
            break

        labels_flat = torch.tensor(labels_this, device=device, dtype=torch.long).repeat_interleave(k)
        gen = torch.Generator(device=device).manual_seed(int(args.seed) + r * 10_007)
        x = torch.randn(
            (bsz * k, latent_channels, latent_size, latent_size),
            generator=gen,
            device=device,
            dtype=pipe.transformer.dtype,
        )

        step1_rewards: torch.Tensor | None = None
        pipe.scheduler.set_timesteps(int(args.num_inference_steps))
        timesteps = pipe.scheduler.timesteps

        for step_idx, t in enumerate(timesteps):
            x = _denoise_one_step(
                pipe=pipe,
                scheduler=pipe.scheduler,
                x=x,
                labels_flat=labels_flat,
                t=t,
                latent_channels=latent_channels,
                generator=gen,
                add_variance_noise=False,
            )

            if step_idx in resample_steps:
                scheduler_state = copy.deepcopy(pipe.scheduler)
                rewards_flat = _mc_rollout_rewards(
                    pipe=pipe,
                    scheduler_state_after_step=scheduler_state,
                    x_after_step=x,
                    labels_flat=labels_flat,
                    timesteps=timesteps,
                    current_step_idx=step_idx,
                    mc=int(args.mc),
                    reward_ctx=reward_ctx,
                    device=device,
                    seed_base=int(args.seed) + r * 100_003 + step_idx * 1_009,
                )
                rewards_now = rewards_flat.view(bsz, k)

                if step_idx == first_resample_step:
                    potential_rewards = rewards_now
                else:
                    if step1_rewards is None:
                        raise RuntimeError("step1 rewards missing before later resample step")
                    potential_rewards = torch.maximum(step1_rewards, rewards_now)

                x, idx, diag = _resample_by_potential(
                    x=x,
                    potential_rewards=potential_rewards,
                    fk_lambda=float(args.fk_lambda),
                    bsz=bsz,
                    k=k,
                )
                rewards_now = _gather_matrix_by_idx(rewards_now, idx)
                if step1_rewards is None:
                    step1_rewards = rewards_now
                else:
                    step1_rewards = _gather_matrix_by_idx(step1_rewards, idx)

                fk_logs.append(
                    {
                        "round": int(r),
                        "step_idx": int(step_idx),
                        "reward_now_mean": float(rewards_now.mean().item()),
                        "reward_step1_mean": float(step1_rewards.mean().item()),
                        **diag,
                    }
                )

        final_images = _decode_latents_to_images(pipe, x)
        final_rewards = _score_images(
            final_images,
            [int(v) for v in labels_flat.detach().cpu().tolist()],
            reward_ctx,
            device,
        ).view(bsz, k)
        best_idx = final_rewards.argmax(dim=1)

        for bi in range(bsz):
            lab = int(labels_this[bi])
            ki = int(best_idx[bi].item())
            best_img = final_images[bi * k + ki]
            best_img.save(best_dir / f"image_{img_counter:05d}_{lab:05d}.png")
            final_rewards_all.append(float(final_rewards[bi, ki].item()))
            if args.save_all_particles:
                for pj in range(k):
                    final_images[bi * k + pj].save(
                        particle_dir / f"image_{img_counter:05d}_{lab:05d}_p{pj:02d}.png"
                    )
            img_counter += 1

        pbar.set_postfix({"saved": img_counter})
        del x, final_images, final_rewards
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if img_counter != len(labels_all):
        raise RuntimeError(f"saved {img_counter} images, expected {len(labels_all)}")

    arr = np.asarray(final_rewards_all, dtype=np.float64)
    summary = {
        "potential_type": "max",
        "num_images": int(img_counter),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "parallel_size": int(args.parallel_size),
        "num_inference_steps": int(args.num_inference_steps),
        "resample_step_idxs": [int(x) for x in resample_steps],
        "fk_lambda": float(args.fk_lambda),
        "mc": int(args.mc),
        "label_reward": str(args.label_reward),
        "final_best_reward_mean": float(arr.mean()) if arr.size else None,
        "final_best_reward_std": float(arr.std(ddof=0)) if arr.size else None,
        "output_dir": str(output_dir),
        "best_images_dir": str(best_dir),
        "seed": int(args.seed),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "fk_diagnostics.json").write_text(
        json.dumps(fk_logs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
