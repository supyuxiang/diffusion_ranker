#!/usr/bin/env python3
r"""
DiT **random search**：每轮从 K 个候选（预计算 suite 或在线 DiT）里用 **ranker 均分 logits** 选最优，
写入 ``output_dir``；可选 init；最后算 FID/IS。

打分实现为 ``reward.ranker_mean_logit_score``（单键 ``repr_key`` 的 ``mean`` 向量 + prompt 表）。

**默认**未指定 ``precomputed_*`` 时自动填与 ``dit_random_search_clipscore`` / ``rollout.py`` 相同路径。

``fk_search/dit_ranker.py`` 依赖本模块的 ``load_diffusion_ranker``、``repr_dict_to_tokens``、
``mean_batch_ranker_score``、``slice_repr_batch`` 等符号。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from diffusers import DiTPipeline

_SEARCH = Path(__file__).resolve().parent
_ROOT = _SEARCH.parent
# 与 ``python3 search/...py`` 一致：脚本目录在 path 上，``reward`` 可解析；项目根供 ``utils`` / ``ranker``。
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SEARCH))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from ranker.model import DiffusionRanker_CrossAttention, DiffusionRanker_MLP, DiffusionRanker_Transformer
from reward import ranker_mean_logit_score
from utils import (
    DitRolloutReprCapture,
    compute_fid_is_from_dirs,
    dit_loader,
    rollout_dit_with_steps,
    set_seed,
)

CANDIDATE_SEED_HIGH = 2**31 - 1


# ---------------------------------------------------------------------------
# 步比例 -> DitRolloutReprCapture 的 step_ratio_ls
# ---------------------------------------------------------------------------


def _step_ratio_ls_from_ratios(
    num_inference_steps: int, step_ratios: tuple[float, ...]
) -> list[float]:
    t = num_inference_steps
    if t <= 1:
        return [0.0]
    return [float(r) for r in step_ratios] if step_ratios else [0.0]


def make_repr_capture(
    pipe: DiTPipeline,
    *,
    deno_steps: int,
    guidance_scale: float,
    layer_ratios: tuple[float, ...],
    step_ratios: tuple[float, ...],
    store_dtype: torch.dtype,
    seed: int,
) -> DitRolloutReprCapture:
    return DitRolloutReprCapture(
        pipe,
        num_inference_steps=deno_steps,
        guidance_scale=guidance_scale,
        layer_ratios=layer_ratios,
        step_ratio_ls=_step_ratio_ls_from_ratios(deno_steps, step_ratios),
        store_dtype=store_dtype,
        rollout_base_seed=seed,
    )


@torch.inference_mode()
def rollout_dit_batch_with_repr(
    pipe: DiTPipeline,
    repr_capture: DitRolloutReprCapture,
    num_steps: int,
    n_this: int,
    device: torch.device,
    generator_seed: int,
    guidance_scale: float,
) -> tuple[list[Image.Image], list[int], dict]:
    batch_labels_ls = torch.randint(0, 1000, (n_this,), device=device).tolist()
    gen = torch.Generator(device=device).manual_seed(int(generator_seed))
    repr_capture.set_capturing(True)
    repr_capture.begin_batch(batch_labels_ls)
    out = pipe(
        class_labels=batch_labels_ls,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=gen,
    )
    repr_capture.end_batch(0, 0, generator_seed=int(generator_seed))
    rec = repr_capture.records.pop()
    repr_dict = rec["repr"]
    imgs = [im.copy() for im in out.images]
    del out, gen
    return imgs, batch_labels_ls, repr_dict


def load_prompt_table(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "tensor" in obj:
        t = obj["tensor"]
    else:
        t = obj
    if not isinstance(t, torch.Tensor) or t.dim() != 2:
        raise ValueError(f"prompt 表须为 2D Tensor，得到 {type(t)} shape={getattr(t, 'shape', None)}")
    if t.shape[0] != 1000:
        print(f"   注意: prompt 表首维为 {t.shape[0]}（非 1000），按 labels 索引，请保证不越界")
    return t.cpu()


def _count_transformer_layers(state: dict) -> int:
    layer_ids: list[int] = []
    for k in state:
        if k.startswith("layers.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit():
            layer_ids.append(int(k.split(".")[1]))
    return (max(layer_ids) + 1) if layer_ids else 2


def _count_mlp_ffn_layers(state: dict) -> int:
    layer_ids: list[int] = []
    for k in state:
        parts = k.split(".")
        if k.startswith("layers.") and len(parts) > 1 and parts[1].isdigit():
            layer_ids.append(int(parts[1]))
    return (max(layer_ids) + 1) if layer_ids else 2


def load_ranker_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    *,
    num_attention_heads: int,
    dropout: float,
) -> nn.Module:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    args_ckpt = ckpt.get("args") or {}
    half = not bool(args_ckpt.get("ranker_float32", False))
    drp = float(args_ckpt.get("dropout", dropout))

    if "output_weights" in state:
        if "linear_candidate.weight" not in state:
            raise ValueError(f"{ckpt_path}: Transformer ckpt 缺 linear_candidate.weight")
        wc = state["linear_candidate.weight"]
        hidden_internal, input_dim_candidate = int(wc.shape[0]), int(wc.shape[1])
        wp = state["linear_prompt.weight"]
        input_dim_prompt = int(wp.shape[1])
        nl = int(
            args_ckpt.get(
                "num_list_layers",
                args_ckpt.get("num_layers", _count_transformer_layers(state)),
            )
        )
        nh = int(
            args_ckpt.get(
                "num_list_heads", args_ckpt.get("num_attention_heads", num_attention_heads)
            )
        )
        model = DiffusionRanker_Transformer(
            input_dim_candidate=input_dim_candidate,
            input_dim_prompt=input_dim_prompt,
            hidden_dim=hidden_internal,
            num_heads=nh,
            dropout=drp,
            num_layers=nl,
            half=half,
        )
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model

    if "score_head.weight" in state:
        wc = state["process_candidate.0.weight"]
        hidden_internal, input_dim_candidate = int(wc.shape[0]), int(wc.shape[1])
        wp0 = state["process_prompt.0.weight"]
        input_dim_prompt = int(wp0.shape[1])
        nl = int(args_ckpt.get("num_layers", _count_mlp_ffn_layers(state)))
        nh = int(args_ckpt.get("num_attention_heads", num_attention_heads))
        model = DiffusionRanker_CrossAttention(
            candidate_hidden_dim=input_dim_candidate,
            prompt_hidden_dim=input_dim_prompt,
            hidden_dim=hidden_internal,
            num_heads=nh,
            num_layers=nl,
            dropout=drp,
            half=half,
        )
        model.load_state_dict(state, strict=True)
        model.input_dim_candidate = input_dim_candidate
        model.input_dim_prompt = input_dim_prompt
        model.to(device)
        model.eval()
        return model

    if "projection.weight" in state and "process_prompt.0.weight" in state:
        wc = state["linear_candidate.weight"]
        hidden_internal, input_dim_candidate = int(wc.shape[0]), int(wc.shape[1])
        wp0 = state["process_prompt.0.weight"]
        input_dim_prompt = int(wp0.shape[1])
        nl = int(args_ckpt.get("num_layers", _count_mlp_ffn_layers(state)))
        model = DiffusionRanker_MLP(
            candidate_hidden_dim=input_dim_candidate,
            prompt_hidden_dim=input_dim_prompt,
            hidden_dim=hidden_internal,
            output_dim=1,
            num_layers=nl,
            dropout=drp,
            half=half,
            criterion=None,
        )
        model.load_state_dict(state, strict=True)
        model.criterion = "cos_t1.0"
        model.input_dim_candidate = input_dim_candidate
        model.input_dim_prompt = input_dim_prompt
        model.to(device)
        model.eval()
        return model

    raise ValueError(
        f"{ckpt_path}: 无法识别 ranker（需 Transformer / MLP / CrossAttention 的 state_dict）"
    )


@torch.inference_mode()
def mean_ranker_score_batch(
    ranker: nn.Module,
    repr_dict: dict,
    repr_key: str,
    labels: list[int],
    prompt_table: torch.Tensor,
    device: torch.device,
) -> float:
    return ranker_mean_logit_score(ranker, repr_dict, repr_key, labels, prompt_table, device)


def slice_repr_batch(repr_dict: dict, n: int) -> dict:
    out: dict = {}
    for k, v in repr_dict.items():
        out[k] = {"mean": v["mean"][:n].clone(), "last": v["last"][:n].clone()}
    return out


def _batch_record_to_pool_entry(
    rec: dict,
    image_dir: Path,
    *,
    source_hint: str = "",
) -> tuple[list[Image.Image], list[int], dict]:
    names = rec.get("image_save_names") or []
    labels = list(rec.get("labels") or [])
    repr_dict = rec.get("repr")
    if not names or not labels or repr_dict is None:
        raise ValueError(f"{source_hint}: batch 缺少 image_save_names / labels / repr")
    if len(names) != len(labels):
        raise ValueError(f"{source_hint}: names/labels 长度不一致")
    image_dir = Path(image_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    imgs: list[Image.Image] = []
    for fn in names:
        p = image_dir / Path(fn).name
        if not p.is_file():
            raise FileNotFoundError(f"未找到: {p}")
        imgs.append(Image.open(p).convert("RGB").copy())
    return imgs, labels, repr_dict


def load_precomputed_candidate_pool(
    repr_pt: Path, image_dir: Path
) -> list[tuple[list[Image.Image], list[int], dict]]:
    try:
        obj = torch.load(repr_pt, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(repr_pt, map_location="cpu")
    batches = obj.get("batches")
    if not batches:
        raise ValueError(f"{repr_pt}: 缺少 batches")
    return [_batch_record_to_pool_entry(rec, image_dir, source_hint=str(repr_pt)) for rec in batches]


def load_precomputed_pool_from_rollout_suite(
    repr_dir: Path,
    image_root: Path,
    num_candidates: int,
    batch_index: int = 0,
) -> list[tuple[list[Image.Image], list[int], dict]]:
    repr_dir = Path(repr_dir).resolve()
    image_root = Path(image_root).resolve()
    pool: list[tuple[list[Image.Image], list[int], dict]] = []
    for idx in range(num_candidates):
        repr_pt = repr_dir / f"idx{idx}.pt"
        cand_dir = image_root / f"candidate_{idx:04d}"
        if not repr_pt.is_file():
            raise FileNotFoundError(f"缺少 {repr_pt}")
        try:
            obj = torch.load(repr_pt, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(repr_pt, map_location="cpu")
        batches = obj.get("batches")
        if not batches or batch_index >= len(batches):
            raise IndexError(f"{repr_pt}: batches 无效或 batch_index 越界")
        pool.append(
            _batch_record_to_pool_entry(
                batches[batch_index], cand_dir, source_hint=f"{repr_pt}"
            )
        )
    return pool


def copy_init_images_from_rollout_candidate(
    output_dir: Path, candidate_dir: Path, n_init: int
) -> int:
    candidate_dir = Path(candidate_dir).resolve()
    paths = sorted(
        candidate_dir.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if len(paths) < n_init:
        raise RuntimeError(f"init 需要 {n_init} 张，{candidate_dir} 仅 {len(paths)} 张")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(paths[:n_init]):
        parts = src.stem.split("_")
        lab = int(parts[2])
        dst = output_dir / f"image_{i:05d}_{lab:05d}.png"
        shutil.copy2(src, dst)
    return n_init


def verify_pool_repr_key(pool, repr_key: str) -> None:
    for i, (_, _, rd) in enumerate(pool):
        if repr_key not in rd:
            raise SystemExit(
                f"预计算 pool[{i}] 缺少 repr 键 {repr_key!r}；示例键: {list(rd.keys())[:6]}"
            )


def verify_keys_subset(repr_d: dict, keys: list[str]) -> None:
    for k in keys:
        if k not in repr_d:
            raise SystemExit(f"repr 缺键 {k!r}；当前有: {list(repr_d.keys())[:12]} …")


def repr_dict_to_tokens(repr_s: dict, selected_keys: list[str], n: int) -> torch.Tensor:
    """多键时在特征维拼接 ``mean``，形状 ``(n, 1, sum_d)``，供 ``mean_batch_ranker_score``（prompt=None）。"""
    parts: list[torch.Tensor] = []
    for k in selected_keys:
        if k not in repr_s:
            raise KeyError(k)
        parts.append(repr_s[k]["mean"].float()[:n])
    tok = torch.cat(parts, dim=-1)
    return tok.unsqueeze(1)


@torch.inference_mode()
def mean_batch_ranker_score(
    ranker: nn.Module,
    tok: torch.Tensor,
    device: torch.device,
    *,
    listwise: bool = True,
) -> float:
    del listwise  # API 兼容 FK；当前实现与 listwise 无关
    w = next(ranker.parameters())
    t = tok.to(device=device, dtype=w.dtype)
    logits = ranker(t, None)
    return float(logits.float().mean().item())


def load_diffusion_ranker(
    ckpt_path: Path | str,
    device: torch.device,
    *,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
) -> tuple[nn.Module, list[str], bool, dict]:
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    args_ckpt = ckpt.get("args") or {}
    keys = args_ckpt.get("selected_repr_keys")
    if not keys:
        rk = args_ckpt.get("repr_key", "")
        keys = [rk] if isinstance(rk, str) and rk.strip() else ["s0_r0.6"]
    if isinstance(keys, str):
        keys = [keys]
    ranker = load_ranker_from_ckpt(
        ckpt_path,
        device,
        num_attention_heads=int(
            args_ckpt.get("num_attention_heads", num_attention_heads)
        ),
        dropout=float(args_ckpt.get("dropout", dropout)),
    )
    listwise = isinstance(ranker, DiffusionRanker_Transformer)
    meta = {"selected_repr_keys": keys, "listwise": listwise}
    return ranker, keys, listwise, meta


def verify_precomputed_pool_ranker_keys(
    pool: list[tuple[list[Image.Image], list[int], dict]],
    keys: list[str],
    *,
    repr_dir_hint: Path | None = None,
) -> None:
    del repr_dir_hint
    for i, (_, _, rd) in enumerate(pool):
        try:
            verify_keys_subset(rd, keys)
        except SystemExit as e:
            raise SystemExit(f"pool[{i}]: {e}") from e


@torch.inference_mode()
def run_random_search_ranker(
    *,
    pipe: DiTPipeline | None,
    repr_capture: DitRolloutReprCapture | None,
    ranker: nn.Module,
    prompt_table: torch.Tensor,
    repr_key: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    deno_steps: int,
    device: torch.device,
    seed: int,
    guidance_scale: float,
    n_init: int,
    precomputed_pool: list[tuple[list[Image.Image], list[int], dict]] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    pool_n = len(precomputed_pool) if precomputed_pool is not None else 0
    if precomputed_pool is not None:
        if pool_n < num_candidates:
            raise ValueError(f"预计算候选数 {pool_n} < --num_candidates={num_candidates}")
        b0 = len(precomputed_pool[0][1])
        for i, (_, lab, _) in enumerate(precomputed_pool):
            if len(lab) != b0:
                raise ValueError(f"预计算各 candidate batch 大小须一致: 0={b0} vs {i}={len(lab)}")
        if b0 != batch_size:
            raise ValueError(f"预计算 batch 大小 {b0} != --batch_size={batch_size}")

    for _ in tqdm(range(num_rounds), desc="random search (ranker)"):
        n_this = min(batch_size, num_samples - img_counter)
        if n_this <= 0:
            break

        best_score = float("-inf")
        best_imgs: list[Image.Image] | None = None
        best_labels: list[int] | None = None

        if precomputed_pool is not None:
            idxs = rng.choice(pool_n, size=num_candidates, replace=False).tolist()
            candidate_iter: list[tuple[list[Image.Image], list[int], dict]] = []
            for j in idxs:
                imgs_f, labels_f, repr_f = precomputed_pool[j]
                if n_this < len(imgs_f):
                    imgs_s = imgs_f[:n_this]
                    labels_s = labels_f[:n_this]
                    repr_s = slice_repr_batch(repr_f, n_this)
                else:
                    imgs_s, labels_s, repr_s = imgs_f, labels_f, repr_f
                candidate_iter.append((imgs_s, labels_s, repr_s))
        else:
            assert pipe is not None and repr_capture is not None
            candidate_iter = []
            for _k in range(num_candidates):
                cand_seed = int(rng.integers(0, CANDIDATE_SEED_HIGH))
                imgs, labels, repr_d = rollout_dit_batch_with_repr(
                    pipe,
                    repr_capture,
                    deno_steps,
                    n_this,
                    device,
                    cand_seed,
                    guidance_scale,
                )
                candidate_iter.append((imgs, labels, repr_d))

        for imgs, labels, repr_d in candidate_iter:
            score = mean_ranker_score_batch(
                ranker, repr_d, repr_key, labels, prompt_table, device
            )
            if score > best_score:
                best_score = score
                best_imgs = imgs
                best_labels = labels

        assert best_imgs is not None and best_labels is not None
        for img, lab in zip(best_imgs, best_labels):
            if img_counter >= num_samples:
                break
            img.save(output_dir / f"image_{img_counter:05d}_{int(lab):05d}.png")
            img_counter += 1


def _parse_float_list(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def default_candidate_suite_paths(guidance_scale: float, deno_steps: int) -> tuple[str, str]:
    gs, T = float(guidance_scale), int(deno_steps)
    image_root = f"/root/autodl-tmp/dit_candidates_50k_cfg{gs}_deno{T}"
    repr_dir = f"/root/autodl-tmp/repr_candidate_deno{T}_cfg{gs}"
    return image_root, repr_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiT random search with trained diffusion ranker")
    p.add_argument("--ranker_ckpt", type=str, required=True)
    p.add_argument(
        "--prompt_repr_pt",
        type=str,
        required=True,
        help="(1000, D) 或带 tensor 字段的 pt",
    )
    p.add_argument(
        "--repr_key",
        type=str,
        required=True,
        help="repr 字典键，如 s1_r0.6（须与预计算 / 抓取一致）",
    )
    p.add_argument("--dit_path", type=str, default=str(Path("/root/autodl-tmp/DiT-XL-2-256")))
    p.add_argument("--imagenet_val", type=str, default="/root/autodl-tmp/imagenet/val")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/eval_dit_random_search_ranker")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--init_random_samples", type=int, default=1024)
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument(
        "--layer_ratios",
        type=str,
        default="0.6",
        help="逗号分隔；须含 repr_key 对应层比",
    )
    p.add_argument(
        "--step_ratios",
        type=str,
        default="0.0",
        help="逗号分隔去噪步比例（0..1）",
    )
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--precomputed_candidates_pt", type=str, default="")
    p.add_argument("--precomputed_candidates_image_dir", type=str, default="")
    p.add_argument("--precomputed_suite_image_root", type=str, default="")
    p.add_argument("--precomputed_suite_repr_dir", type=str, default="")
    p.add_argument("--precomputed_suite_num_candidates", type=int, default=16)
    p.add_argument("--precomputed_suite_batch_index", type=int, default=0)
    p.add_argument("--init_suite_candidate_idx", type=int, default=0)
    p.add_argument(
        "--online_rollout_candidates",
        action="store_true",
        help="search 每轮在线 DiT 采样；默认用预计算池",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if (
        not args.online_rollout_candidates
        and not str(args.precomputed_candidates_pt).strip()
        and not str(args.precomputed_suite_image_root).strip()
        and not str(args.precomputed_suite_repr_dir).strip()
    ):
        ir, rd = default_candidate_suite_paths(args.guidance_scale, args.deno_steps)
        args.precomputed_suite_image_root = ir
        args.precomputed_suite_repr_dir = rd
        print(
            "未指定 precomputed_*，使用默认候选路径：\n"
            f"  --precomputed_suite_image_root {ir}\n"
            f"  --precomputed_suite_repr_dir {rd}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()
    dit_path = Path(args.dit_path).expanduser().resolve()
    ckpt_p = Path(args.ranker_ckpt).expanduser().resolve()
    prompt_p = Path(args.prompt_repr_pt).expanduser().resolve()

    if not imagenet_val.is_dir():
        raise SystemExit(f"ImageNet val 未找到: {imagenet_val}")

    use_pt_single = bool(str(args.precomputed_candidates_pt).strip())
    use_suite = bool(str(args.precomputed_suite_image_root).strip())
    if use_pt_single and use_suite:
        raise SystemExit("不要同时指定 precomputed_candidates_pt 与 suite")
    if use_pt_single and not str(args.precomputed_candidates_image_dir).strip():
        raise SystemExit("precomputed_candidates_pt 需要 precomputed_candidates_image_dir")
    if use_suite and not str(args.precomputed_suite_repr_dir).strip():
        raise SystemExit("precomputed_suite_image_root 需要 precomputed_suite_repr_dir")

    precomputed = use_pt_single or use_suite
    if not args.online_rollout_candidates and not precomputed:
        raise SystemExit(
            "须指定预计算 suite/pt，或加 --online_rollout_candidates 做在线采样。"
        )

    init_from_suite = use_suite and args.init_random_samples > 0
    need_dit = (args.online_rollout_candidates and not precomputed) or (
        args.init_random_samples > 0 and not init_from_suite
    )
    if need_dit and not dit_path.is_dir():
        raise SystemExit(f"DiT 路径未找到: {dit_path}")

    assert 0 <= args.init_random_samples <= args.num_samples
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_s = "fp16" if args.fp16 else "fp32"

    print(f"加载 ranker: {ckpt_p}")
    ranker = load_ranker_from_ckpt(
        ckpt_p,
        device,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
    )
    ic = getattr(ranker, "input_dim_candidate", None) or ranker.linear_candidate.in_features
    ip = getattr(ranker, "input_dim_prompt", None)
    if ip is None:
        ip = (
            ranker.linear_prompt.in_features
            if hasattr(ranker, "linear_prompt")
            else ranker.process_prompt[0].in_features
        )
    print(f"   input_dim_candidate={ic} input_dim_prompt={ip} hidden={ranker.linear_candidate.out_features}")

    print(f"加载 prompt 表: {prompt_p}")
    prompt_table = load_prompt_table(prompt_p)
    if prompt_table.shape[1] != int(ip):
        raise SystemExit(
            f"prompt 维 {prompt_table.shape[1]} != ranker.input_dim_prompt={ip}"
        )

    precomputed_pool = None
    if use_suite:
        precomputed_pool = load_precomputed_pool_from_rollout_suite(
            Path(args.precomputed_suite_repr_dir).expanduser().resolve(),
            Path(args.precomputed_suite_image_root).expanduser().resolve(),
            int(args.precomputed_suite_num_candidates),
            batch_index=int(args.precomputed_suite_batch_index),
        )
        print(f"预计算 suite: {len(precomputed_pool)} candidates")
    elif use_pt_single:
        precomputed_pool = load_precomputed_candidate_pool(
            Path(args.precomputed_candidates_pt).expanduser().resolve(),
            Path(args.precomputed_candidates_image_dir).expanduser().resolve(),
        )
        print(f"预计算单 pt: {len(precomputed_pool)} batches")

    if precomputed_pool is not None:
        verify_pool_repr_key(precomputed_pool, args.repr_key)

    layer_ratios = _parse_float_list(args.layer_ratios)
    step_ratios = _parse_float_list(args.step_ratios)

    pipe: DiTPipeline | None = None
    repr_capture: DitRolloutReprCapture | None = None
    if need_dit:
        print(f"加载 DiT: {dit_path} ({dtype_s})")
        pipe = dit_loader(dit_path, device, dtype=dtype_s)
        pipe.set_progress_bar_config(disable=True)
        store_dtype = torch.float16 if args.fp16 else torch.float32
        repr_capture = make_repr_capture(
            pipe,
            deno_steps=args.deno_steps,
            guidance_scale=args.guidance_scale,
            layer_ratios=layer_ratios,
            step_ratios=step_ratios,
            store_dtype=store_dtype,
            seed=args.seed,
        )
        repr_capture.install()

    try:
        n_init = 0
        if args.init_random_samples > 0:
            if init_from_suite:
                root = Path(args.precomputed_suite_image_root).expanduser().resolve()
                icand = int(args.init_suite_candidate_idx)
                cand_dir = root / f"candidate_{icand:04d}"
                n_init = copy_init_images_from_rollout_candidate(
                    output_dir, cand_dir, args.init_random_samples
                )
                print(f"init: 从 {cand_dir} 复制 {n_init} 张")
            else:
                assert pipe is not None
                if repr_capture is not None:
                    repr_capture.set_capturing(False)
                n_init = rollout_dit_with_steps(
                    pipe,
                    output_dir,
                    args.deno_steps,
                    args.init_random_samples,
                    args.batch_size,
                    device,
                    args.seed,
                    guidance_scale=args.guidance_scale,
                    repr_capture=None,
                )
                print(f"init: DiT 写入 {n_init} 张")
                if repr_capture is not None:
                    repr_capture.set_capturing(True)
        else:
            print("init: 跳过")

        run_random_search_ranker(
            pipe=pipe,
            repr_capture=repr_capture,
            ranker=ranker,
            prompt_table=prompt_table,
            repr_key=args.repr_key,
            output_dir=output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_candidates=args.num_candidates,
            deno_steps=args.deno_steps,
            device=device,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            n_init=n_init,
            precomputed_pool=precomputed_pool,
        )
    finally:
        if repr_capture is not None:
            repr_capture.remove()

    cfg = {
        "ranker_ckpt": str(ckpt_p),
        "prompt_repr_pt": str(prompt_p),
        "repr_key": args.repr_key,
        "layer_ratios": list(layer_ratios),
        "step_ratios": list(step_ratios),
        "output_dir": str(output_dir),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_candidates": args.num_candidates,
        "deno_steps": args.deno_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n完成。图像: {output_dir}  config: {output_dir / 'config.json'}")

    print("\nFID / IS …")
    metrics = compute_fid_is_from_dirs(
        output_dir,
        imagenet_val,
        device,
        batch_size=args.metrics_batch_size,
    )
    print(f"   FID: {metrics['FID']:.2f}, IS: {metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}")
    (output_dir / "final_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
