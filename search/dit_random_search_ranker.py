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
import shutil
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


def _all_layer_ratios(n_blocks: int) -> tuple[float, ...]:
    """与 ``ranker/rollout_trianing_data.py`` 一致：每个 DiT block 一个 ratio。"""
    if n_blocks <= 1:
        return (1.0,)
    return tuple(i / (n_blocks - 1) for i in range(n_blocks))


def _all_step_ratio_ls(num_inference_steps: int) -> list[float]:
    """使 hook 步覆盖 ``0 .. T-1``（与 rollout 训练数据一致）。"""
    t = num_inference_steps
    if t <= 1:
        return [0.0]
    return [i / (t - 1) for i in range(t)]


def make_repr_capture(
    pipe: DiTPipeline,
    args: argparse.Namespace,
    *,
    ranker_meta: dict | None,
) -> DitRolloutReprCapture:
    """
    ``verifier=fid``：沿用 ``DitRolloutReprCapture`` 默认稀疏层/步（省算力）。
    ``verifier=ranker``：优先用 checkpoint ``meta``（与 ``repr.pt`` / ``selected_repr_keys`` 对齐）；
    无 ``meta`` 时与 rollout 脚本相同：全 block + 全去噪步。
    """
    if args.verifier != "ranker":
        store_dtype = torch.float16 if args.fp16 else torch.float32
        return DitRolloutReprCapture(
            pipe,
            num_inference_steps=args.deno_steps,
            guidance_scale=args.guidance_scale,
            store_dtype=store_dtype,
            rollout_base_seed=args.seed,
        )

    meta = ranker_meta or {}
    n_blocks = len(pipe.transformer.transformer_blocks)

    if meta.get("layer_ratios"):
        layer_ratios = tuple(float(x) for x in meta["layer_ratios"])
        T = int(meta.get("num_inference_steps") or args.deno_steps)
        if T != args.deno_steps:
            print(
                f"   警告: ranker ckpt meta 使用 num_inference_steps={T}，"
                f"与 --deno_steps={args.deno_steps} 不一致；已按 meta 对齐。"
            )
        step_ratio_ls = _all_step_ratio_ls(T)
        gs = float(meta.get("guidance_scale", args.guidance_scale))
        sd = meta.get("store_dtype")
        store_dtype = (
            sd
            if isinstance(sd, torch.dtype)
            else (torch.float16 if args.fp16 else torch.float32)
        )
    else:
        layer_ratios = _all_layer_ratios(n_blocks)
        T = args.deno_steps
        step_ratio_ls = _all_step_ratio_ls(T)
        gs = args.guidance_scale
        store_dtype = torch.float16 if args.fp16 else torch.float32
        print(
            "   警告: ranker checkpoint 无 meta.layer_ratios，"
            "已用当前 DiT 的「全层 + 全步」抓取（与 rollout_trianing_data 默认一致）。"
        )

    return DitRolloutReprCapture(
        pipe,
        num_inference_steps=T,
        guidance_scale=gs,
        layer_ratios=layer_ratios,
        step_ratio_ls=step_ratio_ls,
        store_dtype=store_dtype,
        rollout_base_seed=args.seed,
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
) -> tuple[torch.nn.Module, list[str], bool, dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    selected: list[str] = list(ckpt.get("selected_repr_keys") or [])
    if not selected:
        raise ValueError(f"{ckpt_path}: missing selected_repr_keys")
    meta_ckpt: dict = dict(ckpt.get("meta") or {})
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
    return model, selected, listwise, meta_ckpt


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


def verify_precomputed_pool_ranker_keys(
    pool: list[tuple[list[Image.Image], list[int], dict]],
    selected_keys: list[str],
    *,
    repr_dir_hint: Path | None = None,
) -> None:
    """启动时检查每个预计算 batch 是否含 ranker 所需键，避免跑到第一轮才报错。"""
    for i, (_, _, rd) in enumerate(pool):
        missing = [k for k in selected_keys if k not in rd]
        if not missing:
            continue
        extra = ""
        if repr_dir_hint is not None:
            extra = f"（suite 下多为 {repr_dir_hint}/idx{i}.pt 仍为旧 rollout 产物）"
        raise SystemExit(
            f"预计算候选 pool[{i}] 与 ranker ckpt 的 selected_repr_keys 不一致："
            f"缺 {len(missing)} 键，例如 {missing[:3]}{extra}。\n"
            "二选一：\n"
            "  A) 用 ``search/rollout.py --repr_full_training`` 重跑 suite，使 repr 键与「全层全步」训练的 ckpt 一致；\n"
            "  B) 保持稀疏层/步 rollout（默认），用同一套 ``--repr_layer_ratios``/``--repr_step_ratios`` "
            "重新训练 ranker（``ranker/train.py`` 的 ``--layer_ratios``/``--step_ratios`` 须与 rollout 一致）。"
        )


def copy_init_images_from_rollout_candidate(
    output_dir: Path,
    candidate_dir: Path,
    n_init: int,
) -> int:
    """
    从 ``rollout.py`` 的 ``candidate_{idx:04d}/`` 按全局图序号取前 ``n_init`` 张，
    复制到 ``output_dir`` 并重命名为 ``image_{0..n_init-1}_{label:05d}.png``
   （与 ``init_random_samples`` + ``init_pool_statistics`` 约定一致）。
    """
    if n_init <= 0:
        return 0
    candidate_dir = Path(candidate_dir).resolve()
    if not candidate_dir.is_dir():
        raise FileNotFoundError(f"init 源目录不存在: {candidate_dir}")
    paths = sorted(
        candidate_dir.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if len(paths) < n_init:
        raise RuntimeError(
            f"init 需要 {n_init} 张，{candidate_dir} 下仅 {len(paths)} 张 image_*_*.png"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(paths[:n_init]):
        parts = src.stem.split("_")
        if len(parts) < 3 or parts[0] != "image":
            raise ValueError(f"unexpected image filename: {src.name}")
        lab = int(parts[2])
        dst = output_dir / f"image_{i:05d}_{lab:05d}.png"
        shutil.copy2(src, dst)
    return n_init


def slice_repr_batch(repr_dict: dict, n: int) -> dict:
    """保留每个 key 的前 ``n`` 条样本（``mean``/``last`` 第 0 维为 batch）。"""
    if n <= 0:
        raise ValueError("n must be positive")
    out: dict = {}
    for k, v in repr_dict.items():
        out[k] = {
            "mean": v["mean"][:n].clone(),
            "last": v["last"][:n].clone(),
        }
    return out


def _batch_record_to_pool_entry(
    rec: dict,
    image_dir: Path,
    *,
    source_hint: str = "",
) -> tuple[list[Image.Image], list[int], dict]:
    """单条 ``batches[i]`` + 图像目录 -> ``(imgs, labels, repr_dict)``。"""
    names = rec.get("image_save_names") or []
    labels = list(rec.get("labels") or [])
    repr_dict = rec.get("repr")
    if not names or not labels or repr_dict is None:
        raise ValueError(f"{source_hint}: batch 缺少 image_save_names / labels / repr")
    if len(names) != len(labels):
        raise ValueError(
            f"{source_hint}: image_save_names 与 labels 长度不一致 "
            f"({len(names)} vs {len(labels)})"
        )
    image_dir = Path(image_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"预计算图像目录不存在: {image_dir}")
    imgs: list[Image.Image] = []
    for fn in names:
        p = image_dir / Path(fn).name
        if not p.is_file():
            raise FileNotFoundError(f"预计算候选图像未找到: {p}")
        imgs.append(Image.open(p).convert("RGB").copy())
    return imgs, labels, repr_dict


def load_precomputed_candidate_pool(
    repr_pt: Path,
    image_dir: Path,
) -> tuple[list[tuple[list[Image.Image], list[int], dict]], dict]:
    """
    读取与 ``DitRolloutReprCapture.save_pt`` / 训练 ``repr.pt`` 相同的 ``batches`` 结构。
    每个 batch：从 ``image_dir`` 加载 ``image_save_names`` 对应 PNG，组装 ``(imgs, labels, repr)``。
    """
    try:
        obj = torch.load(repr_pt, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(repr_pt, map_location="cpu")
    batches = obj.get("batches")
    if not batches:
        raise ValueError(f"{repr_pt}: missing non-empty 'batches'")
    meta = dict(obj.get("meta") or {})
    image_dir = Path(image_dir).resolve()

    pool: list[tuple[list[Image.Image], list[int], dict]] = []
    for rec in batches:
        pool.append(
            _batch_record_to_pool_entry(rec, image_dir, source_hint=str(repr_pt))
        )

    return pool, meta


def load_precomputed_pool_from_rollout_suite(
    repr_dir: Path,
    image_root: Path,
    num_candidates: int,
    batch_index: int = 0,
) -> tuple[list[tuple[list[Image.Image], list[int], dict]], dict]:
    """
    与 ``scripts/rollout_all.sh`` / ``search/rollout.py`` 输出一致：

    - ``{repr_dir}/idx{i}.pt``（``i = 0 .. num_candidates-1``）
    - 图像在 ``{image_root}/candidate_{i:04d}/``（与 ``rollout._output_dir`` 相同）

    每个 candidate 只取 ``batches[batch_index]`` 一条（默认 0，即首个 batch）作为该候选的一条「可比 batch」。
    """
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")
    if batch_index < 0:
        raise ValueError("batch_index must be >= 0")

    repr_dir = Path(repr_dir).resolve()
    image_root = Path(image_root).resolve()
    if not repr_dir.is_dir():
        raise FileNotFoundError(f"repr 目录不存在: {repr_dir}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"candidate 图像根目录不存在: {image_root}")

    pool: list[tuple[list[Image.Image], list[int], dict]] = []
    meta0: dict = {}
    for idx in range(num_candidates):
        repr_pt = repr_dir / f"idx{idx}.pt"
        cand_dir = image_root / f"candidate_{idx:04d}"
        if not repr_pt.is_file():
            raise FileNotFoundError(f"缺少 repr 文件: {repr_pt}")
        try:
            obj = torch.load(repr_pt, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(repr_pt, map_location="cpu")
        batches = obj.get("batches")
        if not batches:
            raise ValueError(f"{repr_pt}: missing non-empty 'batches'")
        if batch_index >= len(batches):
            raise IndexError(
                f"{repr_pt}: batches 长度 {len(batches)}，"
                f"--precomputed_suite_batch_index={batch_index} 越界"
            )
        if idx == 0:
            meta0 = dict(obj.get("meta") or {})
        rec = batches[batch_index]
        pool.append(
            _batch_record_to_pool_entry(
                rec, cand_dir, source_hint=f"{repr_pt} batches[{batch_index}]"
            )
        )

    return pool, meta0


@torch.inference_mode()
def run_random_search(
    pipe: DiTPipeline | None,
    repr_capture: DitRolloutReprCapture | None,
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
    *,
    precomputed_pool: list[tuple[list[Image.Image], list[int], dict]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    if precomputed_pool is not None:
        pool_n = len(precomputed_pool)
        if pool_n < num_candidates:
            raise ValueError(
                f"预计算候选数 {pool_n} 小于 --num_candidates={num_candidates}"
            )
        b0 = len(precomputed_pool[0][1])
        for i, (_, lab, _) in enumerate(precomputed_pool):
            if len(lab) != b0:
                raise ValueError(
                    f"预计算各 batch 大小须一致；batch 0 为 {b0}，batch {i} 为 {len(lab)}"
                )
        if b0 != batch_size:
            raise ValueError(
                f"预计算 batch 大小 {b0} 与 --batch_size={batch_size} 不一致"
            )

    for _ in tqdm(range(num_rounds), desc=f"random search ({verifier})"):
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
            candidate_seeds = rng.integers(
                0, CANDIDATE_SEED_HIGH, size=num_candidates
            ).tolist()
            candidate_iter = []
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
                candidate_iter.append((imgs, labels, repr_dict))

        for imgs, labels, repr_dict in candidate_iter:

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
    p.add_argument(
        "--precomputed_candidates_pt",
        type=str,
        default="",
        help=(
            "若设置：random search 阶段不再 DiT rollout，从该 ``repr.pt`` 的 ``batches`` 里"
            "每次随机抽 ``num_candidates`` 条 batch 参与比较（需与 ``--batch_size`` 一致）。"
        ),
    )
    p.add_argument(
        "--precomputed_candidates_image_dir",
        type=str,
        default="",
        help="预计算 batch 中 ``image_save_names`` 所在目录（与 rollout 数据目录一致）。",
    )
    p.add_argument(
        "--precomputed_suite_image_root",
        type=str,
        default="",
        help=(
            "``scripts/rollout_all.sh`` / ``search/rollout.py`` 布局："
            "``candidate_0000`` … 所在根目录。"
            "若 ``init_random_samples>0``，init 会从 ``--init_suite_candidate_idx`` 对应目录复制，"
            "不再跑 DiT。"
        ),
    )
    p.add_argument(
        "--precomputed_suite_repr_dir",
        type=str,
        default="",
        help=(
            "上述 layout 下各 ``idx{i}.pt`` 所在目录（如 "
            "``/root/autodl-tmp/repr_candidate_deno10_cfg1.0``）。"
        ),
    )
    p.add_argument(
        "--precomputed_suite_num_candidates",
        type=int,
        default=16,
        help="suite 模式下 candidate 个数（与 rollout 的 idx 0..N-1 一致，默认 16）。",
    )
    p.add_argument(
        "--precomputed_suite_batch_index",
        type=int,
        default=0,
        help="每个 ``idx{i}.pt`` 内取 ``batches[k]`` 的索引 k（默认 0）。",
    )
    p.add_argument(
        "--init_suite_candidate_idx",
        type=int,
        default=0,
        help=(
            "suite 模式且 ``init_random_samples>0`` 时：从 "
            "``candidate_{idx:04d}/`` 复制 init 图（默认 0，即与 ``rollout_all`` 首条 candidate 一致）。"
        ),
    )
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

    use_pt_single = bool(str(args.precomputed_candidates_pt).strip())
    use_suite = bool(str(args.precomputed_suite_image_root).strip())
    if use_pt_single and use_suite:
        raise SystemExit(
            "不要同时指定单文件（--precomputed_candidates_pt）与 "
            "suite（--precomputed_suite_image_root）"
        )
    use_precomputed = use_pt_single or use_suite
    if use_pt_single:
        if not str(args.precomputed_candidates_image_dir).strip():
            raise SystemExit(
                "使用 --precomputed_candidates_pt 时必须提供 --precomputed_candidates_image_dir"
            )
    if use_suite:
        if not str(args.precomputed_suite_repr_dir).strip():
            raise SystemExit(
                "使用 --precomputed_suite_image_root 时必须提供 --precomputed_suite_repr_dir"
            )
    init_from_suite_disk = use_suite and args.init_random_samples > 0
    need_dit = (not use_precomputed) or (
        args.init_random_samples > 0 and not init_from_suite_disk
    )
    if need_dit and not dit_path.is_dir():
        raise SystemExit(f"DiT 路径未找到: {dit_path}")

    assert 0 <= args.init_random_samples <= args.num_samples

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_s = "fp16" if args.fp16 else "fp32"

    ranker: torch.nn.Module | None = None
    selected_keys: list[str] | None = None
    listwise = False
    ranker_meta: dict | None = None
    if args.verifier == "ranker":
        ckpt_p = Path(args.ranker_ckpt).expanduser().resolve()
        ranker, selected_keys, listwise, ranker_meta = load_diffusion_ranker(
            ckpt_p, device
        )

    precomputed_pool: list[tuple[list[Image.Image], list[int], dict]] | None = None
    pre_meta: dict = {}
    suite_image_root: Path | None = None
    if use_suite:
        suite_image_root = Path(args.precomputed_suite_image_root).expanduser().resolve()
        repr_d = Path(args.precomputed_suite_repr_dir).expanduser().resolve()
        nc = int(args.precomputed_suite_num_candidates)
        bi = int(args.precomputed_suite_batch_index)
        print(
            f"1. 加载 rollout suite 候选池: repr={repr_d}  images={suite_image_root}  "
            f"candidates={nc}  batch_index={bi} …"
        )
        precomputed_pool, pre_meta = load_precomputed_pool_from_rollout_suite(
            repr_d,
            suite_image_root,
            nc,
            batch_index=bi,
        )
        print(f"   共 {len(precomputed_pool)} 个 candidate batch")
    elif use_pt_single:
        pt_p = Path(args.precomputed_candidates_pt).expanduser().resolve()
        img_d = Path(args.precomputed_candidates_image_dir).expanduser().resolve()
        print(f"1. 加载预计算候选池: {pt_p}（图像目录 {img_d}）…")
        precomputed_pool, pre_meta = load_precomputed_candidate_pool(pt_p, img_d)
        print(f"   共 {len(precomputed_pool)} 个 batch 候选")

    if precomputed_pool is not None and selected_keys:
        verify_precomputed_pool_ranker_keys(
            precomputed_pool,
            selected_keys,
            repr_dir_hint=Path(args.precomputed_suite_repr_dir).expanduser().resolve()
            if use_suite
            else None,
        )

    pipe: DiTPipeline | None = None
    repr_capture: DitRolloutReprCapture | None = None
    inference_steps = int(
        pre_meta.get("num_inference_steps") or args.deno_steps
    )
    guidance_scale_run = float(
        pre_meta.get("guidance_scale", args.guidance_scale)
    )

    if need_dit:
        print("1b. 加载 DiT …" if use_precomputed else "1. 加载 DiT …")
        pipe = dit_loader(dit_path, device, dtype=dtype_s)
        pipe.set_progress_bar_config(disable=True)
        repr_capture = make_repr_capture(pipe, args, ranker_meta=ranker_meta)
        inference_steps = repr_capture.T
        guidance_scale_run = repr_capture.guidance_scale
        repr_capture.install()
    elif use_precomputed and ranker_meta:
        # 无 DiT 时仍用 ranker ckpt 的 meta 写配置（与训练表征一致）
        inference_steps = int(
            ranker_meta.get("num_inference_steps") or inference_steps
        )
        guidance_scale_run = float(
            ranker_meta.get("guidance_scale", guidance_scale_run)
        )

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

        print(f"2a. 初始化 {args.init_random_samples} 张 …")
        if args.init_random_samples > 0:
            if init_from_suite_disk:
                assert suite_image_root is not None
                ic = int(args.init_suite_candidate_idx)
                cand_dir = suite_image_root / f"candidate_{ic:04d}"
                n_init = copy_init_images_from_rollout_candidate(
                    output_dir,
                    cand_dir,
                    args.init_random_samples,
                )
                print(
                    f"   已从 rollout 复制: {cand_dir} -> {output_dir}（共 {n_init} 张）"
                )
            else:
                n_init = init_random_samples(
                    pipe,
                    output_dir,
                    args.init_random_samples,
                    args.batch_size,
                    inference_steps,
                    device,
                    args.seed,
                    guidance_scale=guidance_scale_run,
                )
                print(f"   DiT rollout 已写入 {n_init} 张")
        else:
            n_init = 0
            print("   （跳过 init）")

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
            inference_steps,
            device,
            args.seed,
            guidance_scale_run,
            n_init,
            precomputed_pool=precomputed_pool,
        )
    finally:
        if repr_capture is not None:
            repr_capture.remove()

    log_path = output_dir / "config.json"
    cfg = {
        "verifier": args.verifier,
        "ranker_ckpt": str(Path(args.ranker_ckpt).resolve())
        if str(args.ranker_ckpt).strip()
        else None,
        "dit_path": str(dit_path) if dit_path.is_dir() else None,
        "precomputed_candidates_pt": str(Path(args.precomputed_candidates_pt).resolve())
        if str(args.precomputed_candidates_pt).strip()
        else None,
        "precomputed_candidates_image_dir": str(
            Path(args.precomputed_candidates_image_dir).resolve()
        )
        if str(args.precomputed_candidates_image_dir).strip()
        else None,
        "precomputed_suite_image_root": str(
            Path(args.precomputed_suite_image_root).resolve()
        )
        if str(args.precomputed_suite_image_root).strip()
        else None,
        "precomputed_suite_repr_dir": str(
            Path(args.precomputed_suite_repr_dir).resolve()
        )
        if str(args.precomputed_suite_repr_dir).strip()
        else None,
        "precomputed_suite_num_candidates": args.precomputed_suite_num_candidates,
        "precomputed_suite_batch_index": args.precomputed_suite_batch_index,
        "init_suite_candidate_idx": args.init_suite_candidate_idx,
        "imagenet_val": str(imagenet_val),
        "output_dir": str(output_dir),
        "init_random_samples": args.init_random_samples,
        "deno_steps": inference_steps,
        "deno_steps_cli": args.deno_steps,
        "guidance_scale": guidance_scale_run,
        "guidance_scale_cli": args.guidance_scale,
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
