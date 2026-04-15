#!/usr/bin/env python3
r"""
用 **训练好的 listwise DiffusionRanker_Transformer** 做 DiT random search（多候选比一条，取 ranker 均分最高）。

依赖：与 ``ranker/train.py`` 一致的单键 DiT 表征（``DitRolloutReprCapture`` 的 ``mean``）+
``prompt_repr`` 表 ``(1000, D_prompt)``。

**默认**：search 阶段**只**从预计算候选池无放回抽样。未写 ``precomputed_*`` 时，自动使用与
``search/dit_random_search_clipscore.py`` / ``search/rollout.py`` 相同的路径约定（由 ``--guidance_scale``
与 ``--deno_steps`` 拼出）::

  图：``/root/autodl-tmp/dit_candidates_50k_cfg{gs}_deno{T}/candidate_XXXX/``
  repr：``/root/autodl-tmp/repr_candidate_deno{T}_cfg{gs}/idx{i}.pt``（与 ``fk_search/dit_ranker.py`` 示例一致）

若要在每一轮 search 里临时用 DiT 采样候选，须显式加 ``--online_rollout_candidates``。

最简示例（依赖上述默认目录已存在；``--repr_key`` 须与 rollout 抓取一致）::

  CUDA_VISIBLE_DEVICES=0 python3 search/dit_random_search_ranker.py \
    --ranker_ckpt ... --prompt_repr_pt ... --repr_key s1_r0.6 \
    --output_dir /root/autodl-tmp/rs_ranker_out \
    --num_samples 5000 --batch_size 32 --num_candidates 8

显式覆盖 suite：``--precomputed_suite_image_root`` + ``--precomputed_suite_repr_dir``。

单 pt 模式：``--precomputed_candidates_pt`` + ``--precomputed_candidates_image_dir``。

在线采样：``--online_rollout_candidates``（可配合 ``--init_random_samples``）。
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

import torch.nn as nn

from ranker.model import DiffusionRanker_MLP, DiffusionRanker_Transformer
from utils import (
    DitRolloutReprCapture,
    compute_fid_is_from_dirs,
    dit_loader,
    rollout_dit_with_steps,
    set_seed,
)
from reawrd import load_ranker,ranker_reward,load_prompt_hstates_ls
from pathlib import Path

CANDIDATE_SEED_HIGH = 2**31 - 1


# ---------------------------------------------------------------------------
# repr capture：与训练导出单键（如 step_0 + layer 0.6）对齐
# ---------------------------------------------------------------------------



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
                raise ValueError(f"预计算各 batch 大小须一致: 0={b0} vs {i}={len(lab)}")
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiT random search with trained listwise ranker")
    p.add_argument("--ranker_ckpt", type=str, required=True, help="ranker/train.py 保存的 .pt（含 model_state_dict）")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/eval_dit_random_search_ranker")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--num_candidates", type=int, default=10)
    p.add_argument("--deno_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DiT pipeline dtype（fp16/fp32）",
    )
    p.add_argument(
        "--layer_ratios",
        type=str,
        default="0.6",
        help="逗号分隔；须含 --repr_key 对应层比（与 DitRolloutReprCapture 一致）",
    )
    p.add_argument(
        "--step_ratios",
        type=str,
        default="0.0",
        help="逗号分隔去噪步比例（0..1），须含 repr 对应步",
    )
    return p.parse_args()


def _parse_float_list(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def default_candidate_suite_paths(guidance_scale: float, deno_steps: int) -> tuple[str, str]:
    """与 ``dit_random_search_clipscore`` / ``rollout.py`` / ``fk_search/dit_ranker`` 惯例一致。"""
    gs, T = float(guidance_scale), int(deno_steps)
    image_root = f"/root/autodl-tmp/dit_candidates_50k_cfg{gs}_deno{T}"
    repr_dir = f"/root/autodl-tmp/repr_candidate_deno{T}_cfg{gs}"
    return image_root, repr_dir


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
            "未指定 precomputed_*，使用默认候选路径（与 clipscore/imagereward/rollout 一致）：\n"
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
        raise SystemExit("不要同时指定单文件 precomputed_candidates_pt 与 suite")
    if use_pt_single and not str(args.precomputed_candidates_image_dir).strip():
        raise SystemExit("precomputed_candidates_pt 需要 precomputed_candidates_image_dir")
    if use_suite and not str(args.precomputed_suite_repr_dir).strip():
        raise SystemExit("precomputed_suite_image_root 需要 precomputed_suite_repr_dir")

    precomputed = use_pt_single or use_suite
    if not args.online_rollout_candidates and not precomputed:
        raise SystemExit(
            "random search 默认从你事先 rollout 好的候选池里抽样（每轮无放回选 --num_candidates 条）。\n"
            "请指定其一：\n"
            "  --precomputed_candidates_pt ... --precomputed_candidates_image_dir ...\n"
            "  或 --precomputed_suite_repr_dir ... --precomputed_suite_image_root ...\n"
            "若要在 search 阶段每轮用 DiT 临时采样候选，请加：--online_rollout_candidates"
        )

    init_from_suite = use_suite and args.init_random_samples > 0
    need_dit = (args.online_rollout_candidates and not precomputed) or (
        args.init_random_samples > 0 and not init_from_suite
    )
    if need_dit and not dit_path.is_dir():
        raise SystemExit(f"DiT 路径未找到: {dit_path}")

    assert 0 <= args.init_random_samples <= args.num_samples

    # init 阶段会调用 rollout_dit_with_steps 写图，须早于 random search 内的 mkdir
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
        ip = ranker.linear_prompt.in_features if hasattr(ranker, "linear_prompt") else ranker.process_prompt[0].in_features
    print(
        f"   input_dim_candidate={ic} input_dim_prompt={ip} "
        f"hidden={ranker.linear_candidate.out_features}"
    )

    print(f"加载 prompt 表: {prompt_p}")
    prompt_table = load_prompt_table(prompt_p)
    if prompt_table.shape[1] != ranker.input_dim_prompt:
        raise SystemExit(
            f"prompt 维 {prompt_table.shape[1]} != ranker.input_dim_prompt={ranker.input_dim_prompt}"
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
                ic = int(args.init_suite_candidate_idx)
                cand_dir = root / f"candidate_{ic:04d}"
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
