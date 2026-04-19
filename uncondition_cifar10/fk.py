#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
from pathlib import Path

import torch
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm

_ROOT_ = Path(__file__).parent.parent.resolve()
if str(_ROOT_) not in sys.path:
    sys.path.insert(0, str(_ROOT_))

from random_search_ranker_trans_reward import (  # noqa: E402
    _dtype_to_torch,
    _latents_to_pil,
    _load_cifar10_test_images,
    _load_images_from_dir_uint8,
    _load_ranker,
    compute_fid,
    compute_is,
    set_device,
    set_seed,
)


def _resolve_step_pair(mode: str, first_step_idx: int, second_step_idx: int) -> tuple[int, int, str]:
    if mode == "early_mid":
        return 100, 600, "Early-step (Top-N/2) + Mid-step"
    if mode == "mid_last":
        return 600, 999, "Mid-step (Top-N/2) + Last-step"
    if mode == "early_last":
        return 100, 999, "Early-step (Top-N/2) + Last-step"
    if mode == "custom":
        return first_step_idx, second_step_idx, "Custom (Top-N/2) + second-step"
    raise ValueError(f"unknown mode: {mode}")


@torch.inference_mode()
def _run_one_candidate_two_steps(
    *,
    pipe: DDPMPipeline,
    base_scheduler,
    timesteps: torch.Tensor,
    target_layer_module,
    first_step_idx: int,
    second_step_idx: int,
    batch_n: int,
    generator_seed: int,
    device: torch.device,
) -> tuple[list[Image.Image], dict[int, torch.Tensor]]:
    capture_steps = {first_step_idx, second_step_idx}
    state: dict[str, int | dict[int, torch.Tensor]] = {"cur_step": -1, "hidden_map": {}}

    def _hook(_m, _inp, _out):
        cur_step = int(state["cur_step"])  # type: ignore[arg-type]
        if cur_step not in capture_steps:
            return
        hidden_map = state["hidden_map"]  # type: ignore[assignment]
        if cur_step in hidden_map:
            return
        t = _out[0] if isinstance(_out, (tuple, list)) else _out
        hidden_map[cur_step] = t.detach().float()

    h = target_layer_module.register_forward_hook(_hook)
    try:
        sched = copy.deepcopy(base_scheduler)
        x = torch.randn(
            (
                batch_n,
                pipe.unet.config.in_channels,
                pipe.unet.config.sample_size,
                pipe.unet.config.sample_size,
            ),
            generator=torch.Generator(device=device).manual_seed(generator_seed),
            device=device,
            dtype=next(pipe.unet.parameters()).dtype,
        )
        for i, t in enumerate(timesteps):
            state["cur_step"] = i
            eps = pipe.unet(x, t).sample
            x = sched.step(eps, t, x, generator=None).prev_sample
            del eps
        images = _latents_to_pil(x)
        hidden_map = state["hidden_map"]  # type: ignore[assignment]
        for s in capture_steps:
            if s not in hidden_map:
                raise RuntimeError(f"hook did not capture hidden at step_idx={s}")
        return images, hidden_map
    finally:
        h.remove()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CIFAR-10 FK difference two-stage random search")
    p.add_argument("--first_ranker_ckpt", type=str, required=True)
    p.add_argument("--second_ranker_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/ddpm_ema_cifar10")
    p.add_argument("--data_root", type=str, default="/root/autodl-tmp/cifar-10_from_autodl_pub")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--parallel_size", type=int, default=8)
    p.add_argument("--first_keep", type=int, default=0, help="top-k kept after first split; 0 means N/2")
    p.add_argument("--num_inference_steps", type=int, default=1000)
    p.add_argument("--layer_idx", type=int, default=12)
    p.add_argument("--mode", type=str, default="early_mid", choices=("early_mid", "mid_last", "early_last", "custom"))
    p.add_argument("--first_step_idx", type=int, default=100)
    p.add_argument("--second_step_idx", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32", "bf16"))
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--input_channels", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.parallel_size <= 1:
        raise ValueError(f"parallel_size must be > 1, got {args.parallel_size}")

    first_step_idx, second_step_idx, strategy_name = _resolve_step_pair(
        args.mode,
        args.first_step_idx,
        args.second_step_idx,
    )
    if not (0 <= first_step_idx < args.num_inference_steps):
        raise ValueError(f"first_step_idx={first_step_idx} out of range [0, {args.num_inference_steps - 1}]")
    if not (0 <= second_step_idx < args.num_inference_steps):
        raise ValueError(f"second_step_idx={second_step_idx} out of range [0, {args.num_inference_steps - 1}]")
    if first_step_idx >= second_step_idx:
        raise ValueError(
            f"first_step_idx must be < second_step_idx, got first={first_step_idx}, second={second_step_idx}"
        )

    keep_k = args.first_keep if args.first_keep > 0 else max(1, args.parallel_size // 2)
    keep_k = min(keep_k, args.parallel_size)

    set_seed(args.seed)
    device = set_device()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink()

    real_images = _load_cifar10_test_images(args.data_root)
    print(f"[info] loaded CIFAR10 test images: {tuple(real_images.shape)}")

    dtype = args.dtype
    if dtype == "bf16":
        print("[warn] bf16 is not supported in DDPMPipeline numpy conversion, fallback to fp16.")
        dtype = "fp16"
    pipe = DDPMPipeline.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        torch_dtype=_dtype_to_torch(dtype),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    base_scheduler = copy.deepcopy(pipe.scheduler)
    timesteps = base_scheduler.timesteps

    resnet_blocks = [(n, m) for n, m in pipe.unet.named_modules() if m.__class__.__name__ == "ResnetBlock2D"]
    if args.layer_idx < 0 or args.layer_idx >= len(resnet_blocks):
        raise IndexError(f"layer_idx out of range: {args.layer_idx}, total={len(resnet_blocks)}")
    layer_name, layer_module = resnet_blocks[args.layer_idx]
    print(
        f"[info] strategy={strategy_name}, layer_idx={args.layer_idx}({layer_name}), "
        f"steps=({first_step_idx}->{second_step_idx}), N={args.parallel_size}, keep={keep_k}"
    )

    ranker_first = _load_ranker(
        Path(args.first_ranker_ckpt).expanduser().resolve(),
        device,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        input_channels=args.input_channels,
        dropout=args.dropout,
    )
    ranker_second = _load_ranker(
        Path(args.second_ranker_ckpt).expanduser().resolve(),
        device,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        input_channels=args.input_channels,
        dropout=args.dropout,
    )

    n_rounds = math.ceil(args.num_samples / args.batch_size)
    img_counter = 0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    round_logs: list[dict[str, object]] = []

    pbar = tqdm(range(n_rounds), desc=f"fk-difference {strategy_name} N={args.parallel_size}->K={keep_k}")
    for ridx in pbar:
        batch_n = min(args.batch_size, args.num_samples - img_counter)
        if batch_n <= 0:
            break

        cand_images_all: list[list[Image.Image]] = []
        cand_hidden_first_all: list[torch.Tensor] = []
        cand_hidden_second_all: list[torch.Tensor] = []
        cand_seed_all: list[int] = []

        for _k in range(args.parallel_size):
            cand_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            cand_images, hidden_map = _run_one_candidate_two_steps(
                pipe=pipe,
                base_scheduler=base_scheduler,
                timesteps=timesteps,
                target_layer_module=layer_module,
                first_step_idx=first_step_idx,
                second_step_idx=second_step_idx,
                batch_n=batch_n,
                generator_seed=cand_seed,
                device=device,
            )
            cand_images_all.append(cand_images)
            cand_hidden_first_all.append(hidden_map[first_step_idx])
            cand_hidden_second_all.append(hidden_map[second_step_idx])
            cand_seed_all.append(cand_seed)

        first_hidden = torch.stack(cand_hidden_first_all, dim=1).to(device, non_blocking=True)
        first_scores = ranker_first(first_hidden).mean(dim=0).float().cpu()
        top_vals, top_idx = torch.topk(first_scores, k=keep_k, largest=True, sorted=True)
        kept_idx = [int(x.item()) for x in top_idx]

        second_hidden = torch.stack([cand_hidden_second_all[i] for i in kept_idx], dim=1).to(device, non_blocking=True)
        second_scores = ranker_second(second_hidden).mean(dim=0).float().cpu()
        best_sub_idx = int(torch.argmax(second_scores).item())
        best_idx = kept_idx[best_sub_idx]
        best_images = cand_images_all[best_idx]

        for im in best_images:
            if img_counter >= args.num_samples:
                break
            im.save(out_dir / f"{img_counter:05d}.png")
            img_counter += 1

        round_logs.append(
            {
                "round": ridx,
                "strategy": strategy_name,
                "first_step_idx": first_step_idx,
                "second_step_idx": second_step_idx,
                "parallel_size": args.parallel_size,
                "keep_k": keep_k,
                "selected_after_first": kept_idx,
                "final_best_idx": best_idx,
                "first_stage_scores": [float(v.item()) for v in first_scores],
                "first_stage_top_scores": [float(v.item()) for v in top_vals],
                "second_stage_scores_on_kept": [float(v.item()) for v in second_scores],
                "candidate_seeds": cand_seed_all,
            }
        )

        del first_hidden, second_hidden, first_scores, second_scores
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gen_images = _load_images_from_dir_uint8(out_dir)
    metrics = {
        "fid": float(compute_fid(real_images, gen_images, device=device)),
    }
    is_mean, is_std = compute_is(gen_images, device=device)
    metrics["is"] = float(is_mean)
    metrics["is_std"] = float(is_std)
    metrics["num_generated"] = float(img_counter)
    metrics["strategy"] = strategy_name
    metrics["first_step_idx"] = float(first_step_idx)
    metrics["second_step_idx"] = float(second_step_idx)
    metrics["parallel_size"] = float(args.parallel_size)
    metrics["first_keep"] = float(keep_k)

    cfg = vars(args).copy()
    cfg["resolved_strategy"] = strategy_name
    cfg["resolved_first_step_idx"] = first_step_idx
    cfg["resolved_second_step_idx"] = second_step_idx
    cfg["resolved_first_keep"] = keep_k

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "round_log.json").write_text(json.dumps(round_logs, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

