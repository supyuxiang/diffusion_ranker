#!/usr/bin/env python3
r"""
用预计算的 ``repr.pt`` + 对应 rollout 图像训练 **Diffusion ranker**，教师为 **ImageReward**。

cd /root/diffusion_ranker && CUDA_VISIBLE_DEVICES=0 python3 -m ranker.train \
  --repr_pt /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/repr.pt \
  --image_dir /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10 \
  --cached_rewards_json /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/reward_ls.json \
  --output_dir /root/autodl-tmp/ranker_train_perclass32_deno10 \
  --ranker_type listwise \
  --group_size 8 \
  --proj_dim 64 \
  --num_list_layers 2 \
  --num_epochs 50 \
  --lr 1e-4 \
  --optimizer adamw \
  --seed 42 \
  --listwise_tau 1.0 \
  --alpha 0.3  
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LambdaLR, SequentialLR
from torch.utils.data import Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))



from ranker.model import ListwiseDiffusionRanker, PointwiseDiffusionRanker
from ranker.utils import Constants_default, ensure_image_reward_on_path, set_device, set_seed, labels_1k
from ranker.loss import mse_kl_combined_loss
from ranker.dataset import DiffusionRankerDataset
from ranker.eval_ranker import eval_epoch
from utils import get_optimizer, get_scheduler












def load_reward_lookup_from_json(json_path: Path, image_dir: Path) -> dict[str, float]:
    """``labeling.py`` 写出的 ``reward_ls.json``：与按全局序号排序的 ``image_*_*.png`` 逐行对齐。"""
    json_path = json_path.expanduser().resolve()
    image_dir = image_dir.expanduser().resolve()
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    paths = sorted(
        image_dir.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if len(raw) != len(paths):
        raise ValueError(
            f"cached rewards len={len(raw)} != sorted rollout images len={len(paths)} "
            f"under {image_dir}"
        )
    return {p.name: float(s) for p, s in zip(paths, raw)}



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
        user_lr = [float(x) for x in layer_ratios]
        allowed_r = layers_from_ratios(user_lr, int(n_bl))
        canon_tol = 1e-7
        literal_tol = 1e-5
        parsed = [
            (k, s, r)
            for k, s, r in parsed
            if any(abs(r - ar) < canon_tol for ar in allowed_r)
            or any(abs(r - ulr) < literal_tol for ulr in user_lr)
        ]
    if step_ratios is not None:
        want = steps_from_ratios(step_ratios, num_steps)
        parsed = [(k, s, r) for k, s, r in parsed if s in want]
    parsed.sort(key=lambda t: (t[1], t[2]))
    return [k for k, _, _ in parsed]


@dataclass
class FlatIndex:
    batch_idx: int
    idx_in_batch: int
    label: int
    image_name: str


def build_flat_indices(pt_obj: dict) -> list[FlatIndex]:
    batches = pt_obj["batches"]
    out: list[FlatIndex] = []
    for b, rec in enumerate(batches):
        labels = rec["labels"]
        names = rec["image_save_names"]
        for i in range(len(labels)):
            out.append(
                FlatIndex(
                    batch_idx=b,
                    idx_in_batch=i,
                    label=int(labels[i]),
                    image_name=str(names[i]),
                )
            )
    return out


def tokens_for_sample(
    pt_obj: dict,
    fi: FlatIndex,
    selected_keys: list[str],
    *,
    pool_mode: str = "mean",
) -> torch.Tensor:
    """
    按 ``pool_mode`` 从 ``repr[k]`` 取向量，在 token 维拼接（顺序与 ``selected_keys`` 一致）。

    - ``mean``：每 key 仅 ``mean``，形状 ``(N_keys, H)``
    - ``last``：每 key 仅 ``last``
    - ``both``：每 key 先 ``mean`` 再 ``last``，形状 ``(2 * N_keys, H)``（与旧版一致）
    """
    mode = (pool_mode or "mean").strip().lower()
    if mode not in ("mean", "last", "both"):
        raise ValueError(f"pool_mode must be mean|last|both, got {pool_mode!r}")

    rec = pt_obj["batches"][fi.batch_idx]
    rd = rec["repr"]
    parts: list[torch.Tensor] = []
    for k in selected_keys:
        if mode in ("mean", "both"):
            parts.append(rd[k]["mean"][fi.idx_in_batch].float())
        if mode in ("last", "both"):
            parts.append(rd[k]["last"][fi.idx_in_batch].float())
    if not parts:
        raise RuntimeError("tokens_for_sample: empty parts (check pool_mode / selected_keys)")
    return torch.stack(parts, dim=0)


def bucket_by_class(indices: list[FlatIndex]) -> dict[int, list[FlatIndex]]:
    buckets: dict[int, list[FlatIndex]] = {}
    for fi in indices:
        buckets.setdefault(fi.label, []).append(fi)
    return buckets


class RankerReprGroupDataset(Dataset):
    """按 ImageNet 类分组；``__getitem__`` 返回同类 ``group_size`` 条的 ``FlatIndex``。"""

    def __init__(
        self,
        buckets: dict[int, list[FlatIndex]],
        group_size: int,
        *,
        classes: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        self.group_size = group_size
        self.classes = sorted(
            c
            for c, lst in buckets.items()
            if len(lst) >= group_size and (classes is None or c in set(classes))
        )
        self.buckets = buckets
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.classes)

    def __getitem__(self, idx: int) -> list[FlatIndex]:
        c = self.classes[idx]
        rng = random.Random(self.seed + self._epoch * 100_003 + idx)
        lst = list(self.buckets[c])
        rng.shuffle(lst)
        return lst[: self.group_size]


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


def teacher_scores_for_group(
    teacher,
    labels: list[str],
    image_paths: list[Path],
    class_ids: list[int],
    ir_device: str,
) -> torch.Tensor:
    scores: list[float] = []
    for p, lab in zip(image_paths, class_ids):
        img = Image.open(p).convert("RGB")
        prompt = prompt_for_class(lab, labels)
        scores.append(float(teacher.score(prompt, img)))
    return torch.tensor(scores, dtype=torch.float32, device=ir_device)


def cached_scores_for_group(
    group_fi: list[FlatIndex],
    reward_by_name: dict[str, float],
) -> torch.Tensor:
    out: list[float] = []
    for fi in group_fi:
        name = fi.image_name
        if name not in reward_by_name:
            raise KeyError(f"no cached score for image_save_names entry {name!r}")
        out.append(reward_by_name[name])
    return torch.tensor(out, dtype=torch.float32, device="cpu")


def iter_training_groups(
    dataset: RankerReprGroupDataset,
    pt_obj: dict,
    selected_keys: list[str],
    image_root: Path,
    reward_by_name: dict[str, float] | None,
    teacher,
    labels: list[str],
    prompt_template: str,
    ir_device: str,
    max_groups: int | None,
    shuffle_rng: random.Random | None = None,
    *,
    pool_mode: str = "mean",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    n = len(dataset)
    order = list(range(n))
    rng = shuffle_rng if shuffle_rng is not None else random.Random()
    rng.shuffle(order)
    for j, idx in enumerate(order):
        if max_groups is not None and j >= max_groups:
            break
        group_fi: list[FlatIndex] = dataset[idx]
        paths = [image_root / fi.image_name for fi in group_fi]
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f"missing image: {p}")
        tok = torch.stack(
            [
                tokens_for_sample(pt_obj, fi, selected_keys, pool_mode=pool_mode)
                for fi in group_fi
            ],
            dim=0,
        )
        class_ids = [fi.label for fi in group_fi]
        with torch.inference_mode():
            if reward_by_name is not None:
                t_scores = cached_scores_for_group(group_fi, reward_by_name)
            else:
                assert teacher is not None
                t_scores = teacher_scores_for_group(
                    teacher, labels, prompt_template, paths, class_ids, ir_device
                )
        yield tok, t_scores




def parse_args() -> argparse.Namespace:
    cd = Constants_default
    p = argparse.ArgumentParser(description="Diffusion ranker + ImageReward (repr.pt + images)")
    p.add_argument("--repr_pt", type=str, required=True, help="rollout 保存的 repr.pt")
    p.add_argument("--image_dir", type=str, required=True, help="与 repr 中 image_save_names 对应的目录")
    p.add_argument(
        "--output_dir", type=str, default=str(_ROOT / "outputs" / "diffusion_ranker_train")
    )
    p.add_argument("--proj_dim", type=int, default=int(cd.get("proj_dim", 64)))
    p.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="ListwiseDiffusionRanker 内 TransformerEncoder 层数",
    )
    p.add_argument(
        "--ranker_type",
        type=str,
        default="listwise",
        choices=("listwise", "pointwise"),
        help="默认 listwise",
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
        help=(
            "逗号分隔 0~1 层比例；按 ``round(u*(n_blocks-1))`` 映射到 block，与步的 round 规则一致。"
            "``all`` 表示不筛选"
        ),
    )
    p.add_argument(
        "--step_ratios",
        type=str,
        default="1.0",
        help="逗号分隔 0~1，映射到步索引 round(u*(T-1))；默认 1.0 为最后一步；``all`` 表示不筛选",
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
    model.train()
    dataset.set_epoch(epoch)
    max_g = args.max_groups_per_epoch if args.max_groups_per_epoch > 0 else None
    total_it = max_g if max_g is not None else len(dataset)
    shuffle_rng = random.Random(args.seed + epoch * 100_003)

    sum_tot = sum_mse = sum_kl = 0.0
    n_steps = 0
    for tok_cpu, t_scores_ir in tqdm(
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
        if ranker_type == "listwise":
            logits = model(tok.unsqueeze(0)).squeeze(0)
        else:
            logits = model(tok)
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
    if ranker_type == "listwise":
        model = ListwiseDiffusionRanker(
            hidden_dim,
            proj_dim=args.proj_dim,
            num_list_layers=args.num_list_layers,
        ).to(device)
    else:
        model = PointwiseDiffusionRanker(hidden_dim, proj_dim=args.proj_dim).to(device)

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
        if (epoch + 1) % max(args.save_every, 1) == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
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
            "args": vars(args),
            "selected_repr_keys": selected_keys,
            "pool_mode": pool_mode,
            "meta": meta,
        },
        out_dir / "ranker_final.pt",
    )


if __name__ == "__main__":
    main()
