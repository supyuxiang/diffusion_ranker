#!/usr/bin/env python3
r"""
离线 random search（ranker）：

- 每轮从预计算 candidate pool 中随机抽 K 个候选；
- 用 ``reward.ranker_mean_logit_score`` 打分；
- 选最优并写图到 ``output_dir``，最后计算 FID/IS。

本脚本只支持 **预计算表征/图像**，不包含在线 DiT rollout 与 init 逻辑。
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

_SEARCH = Path(__file__).resolve().parent
_ROOT = _SEARCH.parent
# 与 ``python3 search/...py`` 一致：脚本目录在 path 上，``reward`` 可解析；项目根供 ``utils`` / ``ranker``。
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SEARCH))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from ranker.model import DiffusionRanker_CrossAttention, DiffusionRanker_MLP, DiffusionRanker_Transformer
from reward import ranker_mean_logit_score
from utils import (
    compute_fid_is_from_dirs,
    set_seed,
)


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


def _load_merged_repr_tensor(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    except RuntimeError:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict) and "tensor" in obj:
        t = obj["tensor"]
    else:
        raise ValueError(f"{path}: 期望 Tensor 或含 tensor 字段的字典")
    if not isinstance(t, torch.Tensor) or t.dim() != 2:
        raise ValueError(f"{path}: 期望 2D Tensor，得到 {type(t)} shape={getattr(t, 'shape', None)}")
    return t


def _sorted_candidate_image_paths(candidate_dir: Path) -> list[Path]:
    return sorted(
        candidate_dir.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def _load_images_labels_slice(
    candidate_dir: Path,
    start: int,
    end: int,
) -> tuple[list[Image.Image], list[int]]:
    paths = _sorted_candidate_image_paths(candidate_dir)
    if end > len(paths):
        raise ValueError(
            f"{candidate_dir}: 切片 [{start}:{end}) 越界（仅 {len(paths)} 张）；"
            "请检查 --precomputed_suite_batch_index / --batch_size。"
        )
    sel = paths[start:end]
    imgs: list[Image.Image] = []
    labels: list[int] = []
    for p in sel:
        lab = int(p.stem.split("_")[2])
        imgs.append(Image.open(p).convert("RGB").copy())
        labels.append(lab)
    return imgs, labels


def _load_pool_from_merged_layout(
    repr_mean_dir: Path,
    repr_last_dir: Path,
    image_root: Path,
    num_candidates: int,
    repr_keys: list[str],
    *,
    batch_index: int,
    batch_size: int | None,
) -> list[tuple[list[Image.Image], list[int], dict]]:
    """
    兼容 scripts/merge_repr_candidate_by_key.py 产物：
    ``repr_dir/mean/{repr_key}.pt``、``repr_dir/last/{repr_key}.pt``。
    行顺序按 idx0 全量后接 idx1 …，这里切出每个 candidate 的一个小批次。
    """
    if not repr_keys:
        raise ValueError("merged layout 需要 repr_keys（至少 1 个）")

    mean_tensors: dict[str, torch.Tensor] = {}
    last_tensors: dict[str, torch.Tensor] = {}
    n_rows: int | None = None
    for k in repr_keys:
        mean_pt = repr_mean_dir / f"{k}.pt"
        last_pt = repr_last_dir / f"{k}.pt"
        if not mean_pt.is_file():
            raise FileNotFoundError(f"缺少 {mean_pt}")
        t_mean = _load_merged_repr_tensor(mean_pt)
        t_last = _load_merged_repr_tensor(last_pt) if last_pt.is_file() else t_mean
        if t_last.shape != t_mean.shape:
            raise ValueError(
                f"{k}: mean/last 形状不一致 {tuple(t_mean.shape)} vs {tuple(t_last.shape)}"
            )
        if n_rows is None:
            n_rows = int(t_mean.shape[0])
        elif t_mean.shape[0] != n_rows:
            raise ValueError(f"{k}: 行数 {t_mean.shape[0]} 与其它 key 的 {n_rows} 不一致")
        mean_tensors[k] = t_mean
        last_tensors[k] = t_last

    assert n_rows is not None
    if n_rows % num_candidates != 0:
        raise ValueError(
            f"合并表征总行数 {n_rows} 不能被 num_candidates={num_candidates} 整除。"
        )
    per_candidate = n_rows // num_candidates
    bs = per_candidate if batch_size is None else int(batch_size)
    start_local = int(batch_index) * bs
    end_local = start_local + bs
    if end_local > per_candidate:
        raise IndexError(
            f"merged 表征每个 candidate 仅 {per_candidate} 行，"
            f"请求切片 [{start_local}:{end_local}) 越界。"
        )

    pool: list[tuple[list[Image.Image], list[int], dict]] = []
    for idx in range(num_candidates):
        row_start = idx * per_candidate + start_local
        row_end = idx * per_candidate + end_local
        candidate_dir = image_root / f"candidate_{idx:04d}"
        imgs, labels = _load_images_labels_slice(candidate_dir, start_local, end_local)
        repr_dict = {
            k: {
                "mean": mean_tensors[k][row_start:row_end].float().clone(),
                "last": last_tensors[k][row_start:row_end].float().clone(),
            }
            for k in repr_keys
        }
        pool.append((imgs, labels, repr_dict))
    return pool


def load_precomputed_pool_from_rollout_suite(
    repr_dir: Path,
    image_root: Path,
    num_candidates: int,
    batch_index: int = 0,
    *,
    repr_keys: list[str] | None = None,
    batch_size: int | None = None,
) -> tuple[list[tuple[list[Image.Image], list[int], dict]], dict]:
    repr_dir = Path(repr_dir).resolve()
    image_root = Path(image_root).resolve()
    if repr_dir.name == "mean":
        repr_mean_dir = repr_dir
        repr_last_dir = repr_dir.parent / "last"
        repr_root = repr_dir.parent
    else:
        repr_root = repr_dir
        repr_mean_dir = repr_root / "mean"
        repr_last_dir = repr_root / "last"
    idx0 = repr_root / "idx0.pt"
    if idx0.is_file():
        pool: list[tuple[list[Image.Image], list[int], dict]] = []
        for idx in range(num_candidates):
            repr_pt = repr_root / f"idx{idx}.pt"
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
        return pool, {"layout": "idx_pt", "batch_index": int(batch_index)}

    if not repr_keys:
        raise FileNotFoundError(
            f"缺少 {idx0}，且未提供 repr_keys，无法从 merged 布局加载。"
        )
    first_mean = repr_mean_dir / f"{repr_keys[0]}.pt"
    if not first_mean.is_file():
        avail = sorted(repr_mean_dir.glob("*.pt"))
        avail_stems = [p.stem for p in avail[:20]]
        raise FileNotFoundError(
            f"缺少 {idx0}，且 merged 文件不存在: {first_mean}。\n"
            f"mean/ 下可用键示例: {avail_stems}"
        )
    pool = _load_pool_from_merged_layout(
        repr_mean_dir,
        repr_last_dir,
        image_root,
        num_candidates,
        repr_keys,
        batch_index=batch_index,
        batch_size=batch_size,
    )
    return pool, {
        "layout": "mean_last_merge",
        "batch_index": int(batch_index),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "repr_keys": list(repr_keys),
    }


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
    ranker: nn.Module,
    prompt_table: torch.Tensor,
    repr_key: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    device: torch.device,
    seed: int,
    precomputed_pool: list[tuple[list[Image.Image], list[int], dict]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_counter = 0
    num_rounds = (num_samples + batch_size - 1) // batch_size

    pool_n = len(precomputed_pool)
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


def default_candidate_suite_paths() -> tuple[str, str]:
    image_root = "/root/autodl-tmp/dit_candidates_50k_cfg1.0_deno10"
    repr_dir = "/root/autodl-tmp/repr_candidate_deno10_cfg1.0/mean"
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
    p.add_argument("--imagenet_val", type=str, default="/root/autodl-tmp/imagenet/val")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/eval_dit_random_search_ranker")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--precomputed_candidates_pt", type=str, default="")
    p.add_argument("--precomputed_candidates_image_dir", type=str, default="")
    p.add_argument("--precomputed_suite_image_root", type=str, default="")
    p.add_argument("--precomputed_suite_repr_dir", type=str, default="")
    p.add_argument("--precomputed_suite_num_candidates", type=int, default=16)
    p.add_argument("--precomputed_suite_batch_index", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if (
        not str(args.precomputed_candidates_pt).strip()
        and not str(args.precomputed_suite_image_root).strip()
        and not str(args.precomputed_suite_repr_dir).strip()
    ):
        ir, rd = default_candidate_suite_paths()
        args.precomputed_suite_image_root = ir
        args.precomputed_suite_repr_dir = rd
        print(
            "未指定 precomputed_*，使用默认离线路径：\n"
            f"  --precomputed_suite_image_root {ir}\n"
            f"  --precomputed_suite_repr_dir {rd}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()
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
    if not precomputed:
        raise SystemExit(
            "当前脚本固定离线模式：须指定预计算 suite/pt。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        precomputed_pool, pre_meta = load_precomputed_pool_from_rollout_suite(
            Path(args.precomputed_suite_repr_dir).expanduser().resolve(),
            Path(args.precomputed_suite_image_root).expanduser().resolve(),
            int(args.precomputed_suite_num_candidates),
            batch_index=int(args.precomputed_suite_batch_index),
            repr_keys=[args.repr_key],
            batch_size=int(args.batch_size),
        )
        print(f"预计算 suite: {len(precomputed_pool)} candidates  layout={pre_meta.get('layout')}")
    elif use_pt_single:
        precomputed_pool = load_precomputed_candidate_pool(
            Path(args.precomputed_candidates_pt).expanduser().resolve(),
            Path(args.precomputed_candidates_image_dir).expanduser().resolve(),
        )
        print(f"预计算单 pt: {len(precomputed_pool)} batches")

    if precomputed_pool is not None:
        verify_pool_repr_key(precomputed_pool, args.repr_key)

    print("init: 固定关闭（n_init=0）")
    run_random_search_ranker(
        ranker=ranker,
        prompt_table=prompt_table,
        repr_key=args.repr_key,
        output_dir=output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        device=device,
        seed=args.seed,
        precomputed_pool=precomputed_pool,
    )

    cfg = {
        "ranker_ckpt": str(ckpt_p),
        "prompt_repr_pt": str(prompt_p),
        "repr_key": args.repr_key,
        "output_dir": str(output_dir),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_candidates": args.num_candidates,
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
