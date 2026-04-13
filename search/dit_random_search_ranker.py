#!/usr/bin/env python3
"""
DiT **在线** Random Search：每轮从 ``num_candidates`` 个不同随机种子的 batch 中选优。

- **``--verifier fid``**（默认）：与 ``dit_random_search_fid_reward.py`` 相同思路，用
  ``reward.fid_reward``（对 ImageNet val 的 FID 改善，越大越好）选 batch；维护已选样本的
  running Inception 统计量。
- **``--verifier ranker``**：用训练好的 **Diffusion ranker**（``ranker/model.py``，checkpoint
  与 ``ranker/train.py`` 一致）对 ``DitRolloutReprCapture`` 表征做 **batch 内平均分** 选优。

表征抓取与训练数据一致：``DitRolloutReprCapture`` 全层比例 + 默认步比例；推理时用 checkpoint 里的
``selected_repr_keys`` 子集拼 token。

用法::

  CUDA_VISIBLE_DEVICES=0 python3 search/dit_random_search_ranker.py \
    --verifier fid \
    --dit_path /root/autodl-tmp/DiT-XL-2-256 \
    --imagenet_val /root/autodl-tmp/imagenet/val \
    --output_dir /root/autodl-tmp/eval_dit_rs_fid \
    --num_samples 5000 \
    --init_random_samples 1024 \
    --batch_size 32 \
    --num_candidates 8 \
    --deno_steps 10 --guidance_scale 1.0

  # ranker 选优（需 ``--ranker_ckpt``）::
  python3 search/dit_random_search_ranker.py --verifier ranker \
    --ranker_ckpt /path/to/ranker_final.pt \
    ...
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from diffusers import DiTPipeline

from dit_random_search_common import (
    CANDIDATE_SEED_HIGH,
    init_random_samples,
    rollout_dit_batch_with_repr,
)
from ranker.model import ListwiseDiffusionRanker, PointwiseDiffusionRanker
from reward import (
    extract_features_from_images,
    fid_features_to_statistics,
    fid_reward,
    merge_fid_statistics,
)
from utils import (
    INCEPTION_WEIGHTS_PATH,
    DitRolloutReprCapture,
    compute_fid_is_from_dirs,
    dit_loader,
    set_seed,
)

try:
    from torch_fidelity.metric_fid import fid_input_id_to_statistics_cached
    from torch_fidelity.utils import create_feature_extractor
except ImportError as e:
    raise ImportError("random search fid verifier needs torch_fidelity") from e


def compute_ref_statistics(real_dir: Path, cache_name: str) -> dict:
    kwargs = dict(
        input2=str(real_dir),
        input2_cache_name=cache_name,
        samples_find_deep=True,
        samples_find_ext="png,jpg,jpeg,JPEG,JPG",
        samples_resize_and_crop=256,
        cuda=torch.cuda.is_available(),
        batch_size=512,
        verbose=False,
        cache=True,
        feature_extractor_weights_path=str(Path(INCEPTION_WEIGHTS_PATH).resolve()),
    )
    feat_extractor = create_feature_extractor("inception-v3-compat", ["2048"], **kwargs)
    return fid_input_id_to_statistics_cached(2, feat_extractor, "2048", **kwargs)


def init_pool_statistics(
    output_dir: Path,
    n_init: int,
    feat_extractor,
    device: torch.device,
) -> tuple[dict, int]:
    paths = sorted(
        output_dir.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )[:n_init]
    if len(paths) < n_init:
        raise RuntimeError(f"init images: need {n_init}, found {len(paths)} under {output_dir}")
    images = [Image.open(p).convert("RGB").copy() for p in paths]
    feats = extract_features_from_images(
        images, feat_extractor, device, batch_size=min(512, len(images))
    )
    return fid_features_to_statistics(feats), len(images)


def repr_dict_to_tokens(
    repr_dict: dict, keys: list[str], batch_size: int
) -> torch.Tensor:
    """``(B, N_tokens, H)``，顺序与 ``ranker/test_repr_ranker.py`` 一致（每 key：mean、last）。"""
    rows: list[torch.Tensor] = []
    for i in range(batch_size):
        parts: list[torch.Tensor] = []
        for k in keys:
            parts.append(repr_dict[k]["mean"][i].float())
            parts.append(repr_dict[k]["last"][i].float())
        rows.append(torch.stack(parts, dim=0))
    return torch.stack(rows, dim=0)


def load_diffusion_ranker(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, list[str], bool]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    selected: list[str] = list(ckpt.get("selected_repr_keys") or [])
    if not selected:
        raise ValueError(f"{ckpt_path}: missing selected_repr_keys")
    args_ckpt = ckpt.get("args") or {}
    listwise = any(k.startswith("list_encoder") for k in state)
    if listwise:
        w = state["candidate_encoder.proj.weight"]
        hidden_dim, proj_dim = w.shape[1], w.shape[0]
        layer_ids: list[int] = []
        for k in state:
            if k.startswith("list_encoder.layers."):
                parts = k.split(".")
                if len(parts) > 3 and parts[2].isdigit():
                    layer_ids.append(int(parts[2]))
        nl = (max(layer_ids) + 1) if layer_ids else int(args_ckpt.get("num_list_layers", 2))
        model = ListwiseDiffusionRanker(
            hidden_dim,
            proj_dim=proj_dim,
            num_list_layers=nl,
        )
    else:
        w = state["encoder.proj.weight"]
        hidden_dim, proj_dim = w.shape[1], w.shape[0]
        model = PointwiseDiffusionRanker(hidden_dim, proj_dim=proj_dim)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, selected, listwise


def mean_batch_ranker_score(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    *,
    listwise: bool,
) -> float:
    """tokens: ``(B, N, H)`` CPU；listwise 时按 K=1 展成 listwise 输入。"""
    with torch.no_grad():
        x = tokens.to(device=device, dtype=torch.float32)
        if listwise:
            logits = model(x.unsqueeze(1)).squeeze(-1).squeeze(-1)
        else:
            logits = model(x)
        return float(logits.float().mean().item())


def verify_keys_subset(repr_dict: dict, selected_keys: list[str]) -> None:
    missing = [k for k in selected_keys if k not in repr_dict]
    if missing:
        raise KeyError(
            "checkpoint selected_repr_keys 与当前 DitRolloutReprCapture 的 key 不一致 "
            f"(missing {len(missing)} keys, e.g. {missing[:3]})"
        )


@torch.inference_mode()
def run_random_search(
    pipe: DiTPipeline,
    repr_capture: DitRolloutReprCapture,
    verifier: str,
    ranker: torch.nn.Module | None,
    selected_keys: list[str] | None,
    listwise: bool,
    feat_extractor,
    ref_stats: dict,
    selected_stats: dict,
    n_selected: int,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    deno_steps: int,
    device: torch.device,
    seed: int,
    guidance_scale: float,
    n_init: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for _ in tqdm(range(num_rounds), desc=f"random search ({verifier})"):
        n_this = min(batch_size, num_samples - img_counter)
        if n_this <= 0:
            break

        candidate_seeds = rng.integers(0, CANDIDATE_SEED_HIGH, size=num_candidates).tolist()
        best_score = float("-inf")
        best_imgs: list[Image.Image] | None = None
        best_labels: list[int] | None = None

        for cand_seed in candidate_seeds:
            imgs, labels, repr_dict = rollout_dit_batch_with_repr(
                pipe,
                repr_capture,
                deno_steps,
                n_this,
                device,
                int(cand_seed),
                guidance_scale=guidance_scale,
            )

            if verifier == "fid":
                feats = extract_features_from_images(
                    imgs, feat_extractor, device, batch_size=min(512, n_this)
                )
                bstat = fid_features_to_statistics(feats)
                score = fid_reward(bstat, len(imgs), selected_stats, n_selected, ref_stats)
            elif verifier == "ranker":
                assert ranker is not None and selected_keys is not None
                verify_keys_subset(repr_dict, selected_keys)
                tok = repr_dict_to_tokens(repr_dict, selected_keys, len(imgs))
                score = mean_batch_ranker_score(
                    ranker, tok, device, listwise=listwise
                )
            else:
                raise ValueError(verifier)

            if score > best_score:
                best_score = score
                best_imgs = imgs
                best_labels = labels

        assert best_imgs is not None and best_labels is not None

        if verifier == "fid":
            feats = extract_features_from_images(
                best_imgs, feat_extractor, device, batch_size=min(512, n_this)
            )
            bstat = fid_features_to_statistics(feats)
            selected_stats = merge_fid_statistics(
                selected_stats, n_selected, bstat, len(best_imgs)
            )
            n_selected += len(best_imgs)

        for img, lab in zip(best_imgs, best_labels):
            if img_counter >= num_samples:
                break
            img.save(output_dir / f"image_{img_counter:05d}_{int(lab):05d}.png")
            img_counter += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--verifier",
        type=str,
        default="fid",
        choices=("fid", "ranker"),
        help="fid：FID reward；ranker：Diffusion ranker batch 均分",
    )
    p.add_argument(
        "--ranker_ckpt",
        type=str,
        default="",
        help="``verifier=ranker`` 时必填；``ranker/train.py`` 保存的 .pt",
    )
    p.add_argument(
        "--dit_path",
        type=str,
        default=str(Path("/root/autodl-tmp/DiT-XL-2-256")),
        help="DiT 本地权重目录",
    )
    p.add_argument("--imagenet_val", type=str, default="/root/autodl-tmp/imagenet/val")
    p.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/eval_dit_random_search_ranker",
    )
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DiT 推理 dtype（默认 fp16）",
    )
    p.add_argument("--ref_cache_name", type=str, default="imagenet_val")
    p.add_argument("--init_random_samples", type=int, default=1024)
    p.add_argument("--metrics_batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verifier == "ranker" and not str(args.ranker_ckpt).strip():
        raise SystemExit("--verifier ranker 需要 --ranker_ckpt")

    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()
    dit_path = Path(args.dit_path).expanduser().resolve()

    if not imagenet_val.is_dir():
        raise SystemExit(f"ImageNet val 未找到: {imagenet_val}")
    if not dit_path.is_dir():
        raise SystemExit(f"DiT 路径未找到: {dit_path}")

    assert 0 <= args.init_random_samples <= args.num_samples

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_s = "fp16" if args.fp16 else "fp32"

    ranker: torch.nn.Module | None = None
    selected_keys: list[str] | None = None
    listwise = False
    if args.verifier == "ranker":
        ckpt_p = Path(args.ranker_ckpt).expanduser().resolve()
        ranker, selected_keys, listwise = load_diffusion_ranker(ckpt_p, device)

    print("1. 加载 DiT …")
    pipe = dit_loader(dit_path, device, dtype=dtype_s)
    pipe.set_progress_bar_config(disable=True)

    store_dtype = torch.float16 if args.fp16 else torch.float32
    repr_capture = DitRolloutReprCapture(
        pipe,
        num_inference_steps=args.deno_steps,
        guidance_scale=args.guidance_scale,
        store_dtype=store_dtype,
        rollout_base_seed=args.seed,
    )
    repr_capture.install()
    try:
        feat_extractor = None
        ref_stats = None
        selected_stats: dict = {}
        n_sel = 0

        if args.verifier == "fid":
            print("   计算 ImageNet val 参考统计量 …")
            ref_stats = compute_ref_statistics(imagenet_val, args.ref_cache_name)
            kwargs_fe = dict(
                cuda=(device.type == "cuda"),
                verbose=False,
                feature_extractor_weights_path=str(Path(INCEPTION_WEIGHTS_PATH).resolve()),
            )
            feat_extractor = create_feature_extractor("inception-v3-compat", ["2048"], **kwargs_fe)

        print(f"2a. 随机初始化 {args.init_random_samples} 张 …")
        n_init = (
            init_random_samples(
                pipe,
                output_dir,
                args.init_random_samples,
                args.batch_size,
                args.deno_steps,
                device,
                args.seed,
                guidance_scale=args.guidance_scale,
            )
            if args.init_random_samples > 0
            else 0
        )
        print(f"   已写入 {n_init} 张")

        if args.verifier == "fid":
            if n_init > 0:
                selected_stats, n_sel = init_pool_statistics(
                    output_dir, n_init, feat_extractor, device
                )
                print(f"   init pool stats: n={n_sel}")
            else:
                selected_stats, n_sel = {}, 0

        print(
            f"2b. Random Search  verifier={args.verifier}  "
            f"num_candidates={args.num_candidates}  batch_size={args.batch_size}"
        )
        run_random_search(
            pipe,
            repr_capture,
            args.verifier,
            ranker,
            selected_keys,
            listwise,
            feat_extractor,
            ref_stats,
            selected_stats,
            n_sel,
            output_dir,
            args.num_samples,
            args.batch_size,
            args.num_candidates,
            args.deno_steps,
            device,
            args.seed,
            args.guidance_scale,
            n_init,
        )
    finally:
        repr_capture.remove()

    log_path = output_dir / "config.json"
    cfg = {
        "verifier": args.verifier,
        "ranker_ckpt": str(Path(args.ranker_ckpt).resolve())
        if str(args.ranker_ckpt).strip()
        else None,
        "dit_path": str(dit_path),
        "imagenet_val": str(imagenet_val),
        "output_dir": str(output_dir),
        "init_random_samples": args.init_random_samples,
        "deno_steps": args.deno_steps,
        "guidance_scale": args.guidance_scale,
        "num_candidates": args.num_candidates,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "fp16": args.fp16,
        "ref_cache_name": args.ref_cache_name,
    }
    log_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n完成。图像目录: {output_dir}，配置: {log_path}")

    print("\n3. FID / IS …")
    metrics = compute_fid_is_from_dirs(
        output_dir,
        imagenet_val,
        device,
        batch_size=args.metrics_batch_size,
    )
    print(
        f"   FID: {metrics['FID']:.2f}, IS: {metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}"
    )
    (output_dir / "final_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
