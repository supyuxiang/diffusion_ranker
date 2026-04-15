#!/usr/bin/env python3
r"""
``repr.pt`` + rollout 图训练 listwise ranker（教师 ImageReward 或 ``reward_ls.json``）。

**默认**：只选 **layer_ratio=0.6、step_ratio=0.0** 对应的一条 repr 键；数据路径指向
``dit_training_data_perclass32_cfg1.0_deno10``。最小启动::

    cd /root/diffusion_ranker && CUDA_VISIBLE_DEVICES=0 python3 -m ranker.train

改输出目录或数据::

    python3 -m ranker.train --output_dir /path/out \\
      --repr_pt /path/repr.pt --image_dir /path/images \\
      --cached_rewards_json /path/reward_ls.json

多键：``--layer_ratios 0.1,0.6`` ``--step_ratios 0,1``；不筛层/步：``--layer_ratios all`` ``--step_ratios all``。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LambdaLR, SequentialLR
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))



from ranker.eval_ranker import eval_epoch
from ranker.loss import mse_kl_combined_loss
from ranker.model import DiffusionRanker_Transformer
from ranker.repr_training_common import (
    RankerReprGroupDataset,
    build_flat_indices,
    bucket_by_class,
    forward_listwise_transformer_ranker,
    iter_training_groups,
    load_imagenet_labels,
    load_prompt_repr_table,
    load_reward_lookup_from_json,
    tokens_for_sample,
)
from utils import Constants_default, ensure_image_reward_on_path, set_device, set_seed


def _args_dict_for_save(ns: argparse.Namespace) -> dict:
    d = dict(vars(ns))
    d.pop("prompt_table", None)
    return d









def parse_repr_key(k: str) -> tuple[int, float]:
    a, b = k.split("_", 1)
    s = int(a[1:])
    r = float(b[1:])
    return s, r


def _parse_float_list(s: str) -> list[float] | None:
    s = (s or "").strip()
    if not s or s.lower() in ("all", "none", ""):
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def infer_num_steps(keys: list[str], meta: dict) -> int:
    if meta and meta.get("num_inference_steps") is not None:
        return int(meta["num_inference_steps"])
    if not keys:
        return 1
    return max(parse_repr_key(k)[0] for k in keys) + 1


def steps_from_ratios(ratios: list[float], num_steps: int) -> set[int]:
    out: set[int] = set()
    ns = max(num_steps, 1)
    for u in ratios:
        u = max(0.0, min(1.0, float(u)))
        s_idx = int(round(u * max(ns - 1, 0)))
        out.add(s_idx)
    return out


def layers_from_ratios(ratios: list[float], n_blocks: int) -> set[float]:
    """
    与 ``DitRolloutReprCapture`` / ``rollout_trianing_data`` 一致：
    ``idx = round(u * (n_blocks - 1))``，repr key 中的 r 为 ``idx / (n_blocks - 1)``。
    """
    n = max(int(n_blocks), 1)
    if n <= 1:
        return {1.0}
    out: set[float] = set()
    nb1 = n - 1
    for u in ratios:
        u = max(0.0, min(1.0, float(u)))
        idx = int(round(u * nb1))
        idx = max(0, min(n - 1, idx))
        out.add(idx / nb1)
    return out


def ratio_to_block_idx(ratio: float, n_blocks: int) -> int:
    """
    与 ``DitRolloutReprCapture`` 选 block 一致：``round(ratio * (n_blocks-1))`` 再 clamp。
    用于按 **block 下标** 匹配键名里解析出的 ``r``（避免因 ``{r:g}`` 短小数与分数不完全相等）。
    """
    n = max(int(n_blocks), 1)
    if n <= 1:
        return 0
    nb1 = n - 1
    u = max(0.0, min(1.0, float(ratio)))
    idx = int(round(u * nb1))
    return max(0, min(n - 1, idx))


def select_repr_keys(
    all_keys: list[str],
    *,
    meta: dict,
    layer_ratios: list[float] | None,
    step_ratios: list[float] | None,
) -> list[str]:
    num_steps = infer_num_steps(all_keys, meta)
    parsed = [(k, *parse_repr_key(k)) for k in all_keys]
    if layer_ratios is not None:
        n_bl = meta.get("n_transformer_blocks")
        if n_bl is None:
            n_bl = len({r for _, _, r in parsed})
        n_bl = int(n_bl)
        user_lr = [float(x) for x in layer_ratios]
        want_block = {ratio_to_block_idx(u, n_bl) for u in user_lr}
        parsed = [
            (k, s, r)
            for k, s, r in parsed
            if ratio_to_block_idx(r, n_bl) in want_block
        ]
    if step_ratios is not None:
        want = steps_from_ratios(step_ratios, num_steps)
        parsed = [(k, s, r) for k, s, r in parsed if s in want]
    parsed.sort(key=lambda t: (t[1], t[2]))
    return [k for k, _, _ in parsed]


def build_optimizer(
    model: torch.nn.Module,
    name: str,
    lr: float,
    weight_decay: float,
    sgd_momentum: float,
) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=sgd_momentum,
            nesterov=sgd_momentum > 0,
        )
    raise ValueError(f"unknown optimizer: {name}")


def build_lr_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_ratio: float,
    base_lr: float,
    lr_min_ratio: float,
):
    name = (name or "none").lower()
    warmup_epochs = max(0, int(round(num_epochs * float(warmup_ratio))))
    eta_min = max(0.0, float(base_lr) * float(lr_min_ratio))
    end_factor = eta_min / base_lr if base_lr > 0 else 1.0

    if name == "none":
        return None

    if warmup_epochs >= num_epochs:
        return LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(num_epochs, 1)
        )

    if name == "cosine":
        w = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=eta_min
        )
        return SequentialLR(optimizer, [w, cosine], milestones=[warmup_epochs])

    if name == "linear":

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return max(1e-8, float(epoch + 1) / float(max(warmup_epochs, 1)))
            e = epoch - warmup_epochs
            T = num_epochs - warmup_epochs
            if T <= 1:
                return float(end_factor)
            t = e / float(T - 1)
            return float(1.0 + (end_factor - 1.0) * t)

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"unknown lr_scheduler: {name}")


def parse_args() -> argparse.Namespace:
    cd = Constants_default
    _data = "/root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10"
    p = argparse.ArgumentParser(description="Diffusion ranker（repr.pt + 图；默认单键 layer=0.6 step=0）")
    p.add_argument(
        "--repr_pt",
        type=str,
        default=f"{_data}/repr.pt",
        help="repr.pt 路径",
    )
    p.add_argument(
        "--image_dir",
        type=str,
        default=_data,
        help="与 repr 中 image_save_names 一致的图目录",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/ranker_train_l06_s00",
        help="checkpoint 输出目录",
    )
    p.add_argument("--proj_dim", type=int, default=int(cd.get("proj_dim", 64)))
    p.add_argument(
        "--num_list_layers",
        type=int,
        default=2,
        help="DiffusionRanker_Transformer 内 TransformerBlock 层数",
    )
    p.add_argument(
        "--num_list_heads",
        type=int,
        default=8,
        help="MultiheadAttention 头数（须整除 --proj_dim 即内部 hidden_dim）",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="TransformerBlock 内 dropout",
    )
    p.add_argument(
        "--prompt_repr_pt",
        type=str,
        default="",
        help="可选：预计算类别 prompt 向量 ``(1000, D)``；空则 prompt 位置用全零（与 model 中 else 分支一致）",
    )
    p.add_argument(
        "--ranker_float32",
        action="store_true",
        help="ranker 不用 bfloat16（默认与 model.half=True 一致）",
    )
    p.add_argument(
        "--ranker_type",
        type=str,
        default="listwise",
        choices=("listwise", "pointwise"),
        help="仅实现 listwise（DiffusionRanker_Transformer）；pointwise 未接好 model",
    )
    p.add_argument("--group_size", type=int, default=int(cd.get("parallel_size", 8)))
    p.add_argument("--num_epochs", type=int, default=int(cd.get("num_epochs", 100)))
    p.add_argument("--lr", type=float, default=float(cd.get("lr", 1e-4)))
    p.add_argument("--weight_decay", type=float, default=float(cd.get("weight_decay", 1e-4)))
    p.add_argument(
        "--optimizer",
        type=str,
        default=str(cd.get("optimizer", "adamw")),
        choices=("adamw", "sgd"),
    )
    p.add_argument("--sgd_momentum", type=float, default=0.9)
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default=str(cd.get("scheduler", "cosine")),
        choices=("cosine", "linear", "none"),
    )
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.01,
        help="余弦/线性衰减末端 lr / 初始 lr",
    )
    p.add_argument("--listwise_tau", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.3, help="MSE 权重；KL 权重为 1-alpha")
    p.add_argument(
        "--layer_ratios",
        type=str,
        default="0.6",
        help="逗号分隔层比例，或 all=不筛层（默认 0.6）",
    )
    p.add_argument(
        "--step_ratios",
        type=str,
        default="0.0",
        help="逗号分隔步比例，或 all=不筛步（默认 0.0=第 0 个去噪步）",
    )
    p.add_argument(
        "--pool_mode",
        type=str,
        default="mean",
        choices=("mean", "last"),
        help="repr 拼接：mean / last",
    )
    p.add_argument("--seed", type=int, default=int(cd.get("seed", 42)))
    p.add_argument(
        "--ir_device",
        type=str,
        default="",
        help="ImageReward 所在设备；默认与训练 device 相同",
    )
    p.add_argument(
        "--cached_rewards_json",
        type=str,
        default=f"{_data}/reward_ls.json",
        help="预计算教师分 JSON；空字符串则在线 ImageReward",
    )
    p.add_argument(
        "--max_groups_per_epoch",
        type=int,
        default=0,
        help="每个 epoch 最多训练多少个类组；0 表示用全部类",
    )
    p.add_argument(
        "--prompt_template",
        type=str,
        default="a photo of a {class_name}",
        help="在线 ImageReward 时的文本模板（需含 {class_name}）",
    )
    p.add_argument("--save_freq", type=int, default=1, help="每 N 个 epoch 存 checkpoint")
    return p.parse_args()




def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pt_obj: dict,
    selected_keys: list[str],
    image_root: Path,
    device: torch.device,
    reward_by_name: dict[str, float] | None,
    teacher,
    labels: list[str],
    prompt_template: str,
    ir_device: str,
    dataset: RankerReprGroupDataset,
    args: argparse.Namespace,
    ranker_type: str,
    epoch: int,
) -> dict[str, float]:
    if ranker_type != "listwise":
        raise RuntimeError("当前仅支持 listwise：请使用 model.DiffusionRanker_Transformer（见 --ranker_type）")
    model.train()
    dataset.set_epoch(epoch)
    max_g = args.max_groups_per_epoch if args.max_groups_per_epoch > 0 else None
    total_it = max_g if max_g is not None else len(dataset)
    shuffle_rng = random.Random(args.seed + epoch * 100_003)

    sum_tot = sum_mse = sum_kl = 0.0
    n_steps = 0
    prompt_table: torch.Tensor | None = getattr(args, "prompt_table", None)
    for tok_cpu, t_scores_ir, _class_id in tqdm(
        iter_training_groups(
            dataset,
            pt_obj,
            selected_keys,
            image_root,
            reward_by_name,
            teacher,
            labels,
            prompt_template,
            ir_device,
            max_g,
            shuffle_rng=shuffle_rng,
            pool_mode=args.pool_mode,
        ),
        desc="train",
        leave=False,
        total=total_it,
    ):
        tok = tok_cpu.to(device, non_blocking=True)
        t_scores = t_scores_ir.to(device)
        logits = forward_listwise_transformer_ranker(
            model, tok, _class_id, prompt_table=prompt_table
        )
        loss, mse_v, kl_v = mse_kl_combined_loss(
            logits, t_scores, args.listwise_tau, args.alpha
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sum_tot += float(loss.detach())
        sum_mse += float(mse_v.detach())
        sum_kl += float(kl_v.detach())
        n_steps += 1

    if n_steps == 0:
        return {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    return {
        "loss": sum_tot / n_steps,
        "mse": sum_mse / n_steps,
        "kl": sum_kl / n_steps,
    }




def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = set_device()
    ir_device = args.ir_device.strip() or str(device)

    repr_path = Path(args.repr_pt).expanduser().resolve()
    image_root = Path(args.image_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reward_by_name: dict[str, float] | None = None
    cache_json = (args.cached_rewards_json or "").strip()
    if cache_json:
        reward_by_name = load_reward_lookup_from_json(Path(cache_json), image_root)
        print(f"使用预计算教师分: {cache_json} ({len(reward_by_name)} 条)")
    else:
        ensure_image_reward_on_path()

    try:
        pt_obj = torch.load(repr_path, map_location="cpu", weights_only=False)
    except TypeError:
        pt_obj = torch.load(repr_path, map_location="cpu")

    if reward_by_name is not None:
        flat_n = len(build_flat_indices(pt_obj))
        if len(reward_by_name) != flat_n:
            raise ValueError(
                f"cached rewards / images count {len(reward_by_name)} != repr.pt 样本数 {flat_n} "
                "(请确认 labeling 与 rollout 为同一 image_dir、且未缺图)"
            )
    meta = pt_obj.get("meta") or {}
    all_keys = sorted(pt_obj["batches"][0]["repr"].keys())
    layer_f = _parse_float_list(args.layer_ratios)
    step_f = _parse_float_list(args.step_ratios)
    selected_keys = select_repr_keys(
        all_keys, meta=meta, layer_ratios=layer_f, step_ratios=step_f
    )
    if not selected_keys:
        raise RuntimeError("筛选后 repr key 为空，请检查 --layer_ratios / --step_ratios")

    flat = build_flat_indices(pt_obj)
    pool_mode = args.pool_mode.strip().lower()
    tok0 = tokens_for_sample(
        pt_obj, flat[0], selected_keys, pool_mode=pool_mode
    )
    hidden_dim = int(tok0.shape[-1])

    ranker_type = args.ranker_type.lower()
    if ranker_type != "listwise":
        raise SystemExit(
            "当前仅支持 --ranker_type listwise（使用 model.DiffusionRanker_Transformer）；"
            "pointwise 未与你的 model.py 对齐。"
        )

    args.prompt_table = None
    pr_path = (args.prompt_repr_pt or "").strip()
    if pr_path:
        args.prompt_table = load_prompt_repr_table(Path(pr_path).expanduser().resolve())
        nrows = int(args.prompt_table.shape[0])
        if nrows < 1000:
            print(f"注意: prompt 表仅 {nrows} 行，请保证 class_id 不越界")

    prompt_dim = int(args.prompt_table.shape[1]) if args.prompt_table is not None else hidden_dim
    if args.proj_dim % args.num_list_heads != 0:
        raise SystemExit(
            f"--proj_dim ({args.proj_dim}) 必须整除 --num_list_heads ({args.num_list_heads})"
        )

    model = DiffusionRanker_Transformer(
        input_dim_candidate=hidden_dim,
        input_dim_prompt=prompt_dim,
        hidden_dim=args.proj_dim,
        num_heads=args.num_list_heads,
        dropout=args.dropout,
        num_layers=args.num_list_layers,
        half=not args.ranker_float32,
    ).to(device)

    labels = load_imagenet_labels()
    buckets = bucket_by_class(flat)
    dataset_tr = RankerReprGroupDataset(buckets, args.group_size, seed=args.seed)

    teacher = None
    if reward_by_name is None:
        import ImageReward as RM

        teacher = RM.load("ImageReward-v1.0", device=ir_device)
        teacher.eval()

    optimizer = build_optimizer(
        model,
        args.optimizer,
        args.lr,
        args.weight_decay,
        args.sgd_momentum,
    )
    scheduler = build_lr_scheduler(
        args.lr_scheduler,
        optimizer,
        args.num_epochs,
        args.warmup_ratio,
        args.lr,
        args.lr_min_ratio,
    )

    for epoch in range(args.num_epochs):
        stats = train_epoch(
            model,
            optimizer,
            pt_obj,
            selected_keys,
            image_root,
            device,
            reward_by_name,
            teacher,
            labels,
            args.prompt_template,
            ir_device,
            dataset_tr,
            args,
            ranker_type,
            epoch,
        )
        if scheduler is not None:
            scheduler.step()
        lr_s = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1}/{args.num_epochs}  lr={lr_s:.2e}  "
            f"loss={stats['loss']:.6f}  mse={stats['mse']:.6f}  kl={stats['kl']:.6f}"
        )
        if (epoch + 1) % max(args.save_freq, 1) == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": _args_dict_for_save(args),
                "selected_repr_keys": selected_keys,
                "pool_mode": pool_mode,
                "meta": meta,
            }
            torch.save(ckpt, out_dir / f"ranker_epoch{epoch + 1}.pt")

    ev = eval_epoch(
        model,
        pt_obj,
        selected_keys,
        image_root,
        device,
        reward_by_name,
        teacher,
        labels,
        args.prompt_template,
        ir_device,
        dataset_tr,
        args,
        ranker_type,
        max_groups=32,
        epoch=args.num_epochs,
    )
    print(f"eval(32 groups): loss={ev['loss']:.6f} mse={ev['mse']:.6f} kl={ev['kl']:.6f}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": _args_dict_for_save(args),
            "selected_repr_keys": selected_keys,
            "pool_mode": pool_mode,
            "meta": meta,
        },
        out_dir / "ranker_final.pt",
    )


if __name__ == "__main__":
    main()
