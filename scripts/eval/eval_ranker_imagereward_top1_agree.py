#!/usr/bin/env python3
"""
固定生成 N 张 DiT 图（默认 N=1000：类 0..999 各 1 张；N>1000 时类 id = i % 1000、不同噪声种子），
再随机抽若干次、每次 k 张，用 ranker 与 ImageReward 分别打分，比较「当前 k 张里 argmax 下标」是否一致。

单 k 模式（未传 --ks）:
  python3 eval_ranker_imagereward_top1_agree.py \\
    --ranker_dir outputs/ranker_mse_kl --subset_size 8

多 k 扫描（同一池、每个 k 各 num_trials 次，并画 k–一致率曲线）:
  python3 eval_ranker_imagereward_top1_agree.py \\
    --ranker_dir outputs/ranker_mse_kl \\
    --pool_size 10000 --num_trials 1000 \\
    --ks 2,4,8,16,32,64,128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import train_ranker_imagenet as tr

from utils import model_loader

NUM_IMAGENET_CLASSES = getattr(tr, "NUM_IMAGENET_CLASSES", 1000)


def load_cfg_and_ranker(ranker_dir: Path, ckpt_name: str, device: torch.device):
    cfg_path = ranker_dir / "run_config.json"
    ckpt_path = ranker_dir / ckpt_name
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    fp16 = bool(cfg.get("fp16", True))
    pipe = model_loader(cfg["dit_path"], device, dtype="fp16" if fp16 else "fp32")
    n_blocks = len(pipe.transformer.transformer_blocks)
    hidden_layer_idx = tr.resolve_hidden_layer_idx(
        n_blocks,
        cfg.get("layer_ratio", 0.6),
        cfg.get("hidden_layer_idx"),
    )

    ranker = tr.CrossSampleDiffusionRanker(
        dit_hidden_dim=int(cfg["dit_hidden_dim"]),
        proj_dim=int(cfg["ranker_proj_dim"]),
        num_tf_layers=int(cfg["ranker_tf_layers"]),
        n_heads=int(cfg.get("ranker_heads", 4)),
        ff_mult=int(cfg.get("ranker_ff_mult", 4)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ranker.load_state_dict(ckpt["ranker_state_dict"])
    ranker.eval()
    return cfg, pipe, ranker, hidden_layer_idx, n_blocks


def build_fixed_pool(
    pipe,
    *,
    pool_size: int,
    deno_steps: int,
    guidance_scale: float,
    hidden_layer_idx: int,
    device: str,
    pool_seed: int,
) -> tuple[torch.Tensor, list, list[int]]:
    """
    第 i 张图：class_id = i % 1000，seed = pool_seed + i * 10007（N>1000 时同类多图用不同种子）。
    返回 pooled_h [N,D] CPU float32，PIL 列表，class_id 列表。
    """
    rows = []
    pils: list = []
    cids: list[int] = []
    desc = "DiT pool" if pool_size > NUM_IMAGENET_CLASSES else "DiT pool (1 img / class)"
    for i in tqdm(range(pool_size), desc=desc):
        cid = i % NUM_IMAGENET_CLASSES
        seed = pool_seed + i * 10007
        pil_list, pooled_h = tr.rollout_dit_same_class_pil_and_pooled_hidden(
            pipe,
            1,
            cid,
            deno_steps,
            device,
            seed,
            hidden_layer_idx,
            guidance_scale=guidance_scale,
        )
        rows.append(pooled_h[0].detach().cpu().float())
        pils.append(pil_list[0])
        cids.append(cid)
    h_all = torch.stack(rows, dim=0)
    return h_all, pils, cids


def run_trials_for_k(
    *,
    k: int,
    pool_size: int,
    num_trials: int,
    pool_h: torch.Tensor,
    pool_pils: list,
    pool_cids: list[int],
    labels_1k: list[str],
    template: str,
    teacher,
    ranker: torch.nn.Module,
    device: torch.device,
    rng: np.random.Generator,
    verbose: bool,
) -> tuple[int, list[dict]]:
    matches = 0
    details: list[dict] = []
    for t in tqdm(range(num_trials), desc=f"trials k={k}", leave=False):
        idx = rng.choice(pool_size, size=k, replace=False)
        h_sub = pool_h[idx].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            ranker_scores = ranker(h_sub).float().cpu().numpy()

        ir_scores = np.empty(k, dtype=np.float64)
        for j, global_i in enumerate(idx):
            prompt = tr.prompt_for_class(labels_1k, pool_cids[int(global_i)], template)
            with torch.inference_mode():
                ir_scores[j] = float(teacher.score(prompt, pool_pils[int(global_i)]))

        top_r = int(np.argmax(ranker_scores))
        top_i = int(np.argmax(ir_scores))
        ok = top_r == top_i
        matches += int(ok)
        row = {
            "trial": t,
            "k": k,
            "indices": idx.tolist(),
            "top1_ranker_local": top_r,
            "top1_imagereward_local": top_i,
            "agree": ok,
        }
        details.append(row)
        if verbose:
            g_r = int(idx[top_r])
            g_i = int(idx[top_i])
            print(
                f"  k={k} trial {t}: agree={ok} | ranker_top local={top_r} global={g_r} | "
                f"IR_top local={top_i} global={g_i}"
            )
    return matches, details


def _plot_k_vs_rate(ks: list[int], rates: list[float], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, rates, "o-", linewidth=2, markersize=7)
    ax.set_xlabel("k (subset size)")
    ax.set_ylabel("top-1 agreement rate")
    ax.set_title(title)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ranker_dir",
        type=str,
        default="/data/home/yxfeng/diffusion_ranker/outputs/ranker_mse_kl",
    )
    p.add_argument("--ckpt_name", type=str, default="ranker_diffusion.pt")
    p.add_argument(
        "--pool_size",
        type=int,
        default=1000,
        help="固定池大小；>1000 时 class_id = i %% 1000，不同 i 不同噪声种子",
    )
    p.add_argument("--num_trials", type=int, default=100, help="单 k 模式下的试验次数；多 k 模式下每个 k 各做这么多次")
    p.add_argument("--subset_size", type=int, default=8, help="单 k 模式：每次抽取张数（未传 --ks 时生效）")
    p.add_argument(
        "--ks",
        type=str,
        default=None,
        help="多 k 扫描，逗号分隔，如 2,4,8,16,32,64,128；指定后忽略 --subset_size，并对每个 k 各跑 --num_trials 次",
    )
    p.add_argument("--pool_seed", type=int, default=42, help="生成固定池时噪声基准种子")
    p.add_argument("--eval_seed", type=int, default=12345, help="子集抽样基准随机种子")
    p.add_argument("--deno_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--verbose", action="store_true", help="打印每次 trial 的 top1 是否一致")
    p.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="结果 JSON 路径；默认 ranker_dir/eval_top1_agreement.json 或 eval_top1_agreement_k_sweep.json",
    )
    p.add_argument(
        "--out_plot",
        type=str,
        default=None,
        help="多 k 模式曲线图路径；默认 ranker_dir/eval_top1_agreement_vs_k.png",
    )
    p.add_argument(
        "--save_trial_details",
        action="store_true",
        help="多 k 时把每次 trial 明细写入 JSON（文件会很大）",
    )
    args = p.parse_args()

    ranker_dir = Path(args.ranker_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ks_list: list[int] | None = None
    if args.ks is not None and str(args.ks).strip():
        ks_list = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
        if not ks_list:
            raise SystemExit("--ks 解析为空")
        for k in ks_list:
            if k < 2:
                raise SystemExit(f"k 必须 >= 2，收到 {k}")
            if k > args.pool_size:
                raise SystemExit(f"k={k} 不能大于 pool_size={args.pool_size}")

    cfg, pipe, ranker, hidden_layer_idx, n_blocks = load_cfg_and_ranker(
        ranker_dir, args.ckpt_name, device
    )
    deno_steps = args.deno_steps if args.deno_steps is not None else int(cfg["deno_steps"])
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else float(cfg["guidance_scale"])
    )
    ir_device = cfg.get("imagereward_device") or str(device)

    deno_steps = int(deno_steps)
    labels_1k = tr.load_imagenet_simple_labels()
    template = str(cfg.get("prompt_template", "a high-quality picture of a {name}"))

    tr._ensure_imagereward_path()
    import ImageReward as RM

    teacher = RM.load("ImageReward-v1.0", device=ir_device)
    teacher.eval()

    pool_msg = (
        f"{args.pool_size} 张 (class_id = i % {NUM_IMAGENET_CLASSES}, seed_i = pool_seed + i*10007)"
        if args.pool_size > NUM_IMAGENET_CLASSES
        else f"{args.pool_size} 张 (class 0..{args.pool_size - 1} 各 1)"
    )
    print(
        f"构建固定池: {pool_msg}, "
        f"deno_steps={deno_steps}, guidance={guidance_scale}, capture_block={hidden_layer_idx}/{n_blocks}"
    )
    with torch.no_grad():
        pool_h, pool_pils, pool_cids = build_fixed_pool(
            pipe,
            pool_size=args.pool_size,
            deno_steps=deno_steps,
            guidance_scale=guidance_scale,
            hidden_layer_idx=hidden_layer_idx,
            device=str(device),
            pool_seed=args.pool_seed,
        )

    if ks_list is not None:
        sweep_results: list[dict] = []
        rates: list[float] = []
        pbar = tqdm(ks_list, desc="k sweep (ranker vs IR top-1)")
        for k in pbar:
            rng = np.random.default_rng(int(args.eval_seed) + k * 1_000_003)
            matches, details = run_trials_for_k(
                k=k,
                pool_size=args.pool_size,
                num_trials=args.num_trials,
                pool_h=pool_h,
                pool_pils=pool_pils,
                pool_cids=pool_cids,
                labels_1k=labels_1k,
                template=template,
                teacher=teacher,
                ranker=ranker,
                device=device,
                rng=rng,
                verbose=args.verbose,
            )
            rate = matches / args.num_trials
            rates.append(rate)
            row = {
                "k": k,
                "matches": matches,
                "num_trials": args.num_trials,
                "agreement_rate": rate,
            }
            if args.save_trial_details:
                row["trials"] = details
            sweep_results.append(row)
            pbar.set_postfix(k=k, rate=f"{rate:.2%}")

        out_json = Path(args.out_json) if args.out_json else ranker_dir / "eval_top1_agreement_k_sweep.json"
        out_plot = Path(args.out_plot) if args.out_plot else ranker_dir / "eval_top1_agreement_vs_k.png"
        payload = {
            "mode": "k_sweep",
            "ranker_dir": str(ranker_dir.resolve()),
            "ckpt_name": args.ckpt_name,
            "pool_size": args.pool_size,
            "num_trials_per_k": args.num_trials,
            "ks": ks_list,
            "pool_seed": args.pool_seed,
            "eval_seed": args.eval_seed,
            "deno_steps": deno_steps,
            "guidance_scale": guidance_scale,
            "hidden_layer_idx": hidden_layer_idx,
            "results": sweep_results,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"多 k 结果已写入 {out_json}")

        title = (
            f"Ranker vs ImageReward local top-1 agree\n"
            f"pool={args.pool_size}, trials/k={args.num_trials}, block={hidden_layer_idx}"
        )
        try:
            _plot_k_vs_rate(ks_list, rates, out_plot, title)
            print(f"曲线已保存 {out_plot}")
        except ImportError:
            print("未安装 matplotlib，跳过作图；可 pip install matplotlib 后重跑仅绘图（需自行从 JSON 画）")

        print("-" * 60)
        for r in sweep_results:
            print(f"  k={r['k']:3d}: {r['matches']}/{r['num_trials']} = {r['agreement_rate']:.2%}")
        print("-" * 60)
        return

    # ---- 单 k（原逻辑）----
    if args.subset_size > args.pool_size:
        raise SystemExit(f"subset_size={args.subset_size} 不能大于 pool_size={args.pool_size}")
    if args.subset_size < 2:
        raise SystemExit("subset_size 须 >= 2")

    rng = np.random.default_rng(args.eval_seed)
    matches, details = run_trials_for_k(
        k=args.subset_size,
        pool_size=args.pool_size,
        num_trials=args.num_trials,
        pool_h=pool_h,
        pool_pils=pool_pils,
        pool_cids=pool_cids,
        labels_1k=labels_1k,
        template=template,
        teacher=teacher,
        ranker=ranker,
        device=device,
        rng=rng,
        verbose=args.verbose,
    )
    rate = matches / args.num_trials
    print("-" * 60)
    print(
        f"top-1 一致: {matches}/{args.num_trials} = {rate:.2%} "
        f"(pool={args.pool_size}, subset={args.subset_size}, eval_seed={args.eval_seed})"
    )
    print("-" * 60)

    out_json = Path(args.out_json) if args.out_json else ranker_dir / "eval_top1_agreement.json"
    payload = {
        "mode": "single_k",
        "matches": matches,
        "num_trials": args.num_trials,
        "agreement_rate": rate,
        "pool_size": args.pool_size,
        "subset_size": args.subset_size,
        "pool_seed": args.pool_seed,
        "eval_seed": args.eval_seed,
        "deno_steps": deno_steps,
        "guidance_scale": guidance_scale,
        "hidden_layer_idx": hidden_layer_idx,
        "trials": details,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"详细结果已写入 {out_json}")


if __name__ == "__main__":
    main()
