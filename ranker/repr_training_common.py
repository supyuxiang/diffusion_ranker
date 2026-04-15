"""
``repr.pt`` 分组训练 / 评估共用：按类组 batch、教师分、迭代器。

由 ``train.py`` 与 ``eval_ranker.py`` 引用，避免与 ``train`` 循环 import。
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

_RP_ROOT = Path(__file__).resolve().parent.parent
if str(_RP_ROOT) not in sys.path:
    sys.path.insert(0, str(_RP_ROOT))

from utils import labels_1k


def load_imagenet_labels() -> list[str]:
    return list(labels_1k)


def load_prompt_repr_table(path: Path) -> torch.Tensor:
    """``(N, D)`` 常用 ``N=1000``；或 ``torch.save`` 的 dict 含 ``tensor`` / ``prompt_repr``。"""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        t = obj.float()
    elif isinstance(obj, dict):
        raw = obj.get("tensor") or obj.get("prompt_repr") or obj.get("data")
        if not isinstance(raw, torch.Tensor):
            raise ValueError(f"{path}: expected Tensor or dict with tensor key")
        t = raw.float()
    else:
        raise ValueError(f"{path}: unsupported type {type(obj)}")
    if t.dim() != 2:
        raise ValueError(f"{path}: expected 2D prompt table, got shape {tuple(t.shape)}")
    return t


def forward_listwise_transformer_ranker(
    model: nn.Module,
    tok: torch.Tensor,
    class_id: int,
    *,
    prompt_table: torch.Tensor | None,
) -> torch.Tensor:
    """
    ``tok``: ``(K, N, H)``；在 token 维 mean 后送入 ``DiffusionRanker_Transformer``。
    返回 ``(K,)`` logits。
    """
    w = next(model.parameters())
    cand = tok.float().mean(dim=1).unsqueeze(0).to(device=w.device, dtype=w.dtype)
    if prompt_table is not None:
        pr = (
            prompt_table[int(class_id)]
            .to(device=w.device, dtype=w.dtype)
            .unsqueeze(0)
        )
    else:
        pr = None
    logits = model(cand, pr)
    return logits.squeeze(0).float()


DEFAULT_IMAGE_REWARD_PROMPT = "a photo of a {class_name}"


def prompt_for_class(
    class_id: int,
    labels: list[str],
    template: str = DEFAULT_IMAGE_REWARD_PROMPT,
) -> str:
    name = labels[int(class_id)]
    return template.format(class_name=name)


def load_reward_lookup_from_json(json_path: Path, image_dir: Path) -> dict[str, float]:
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
        labels_ls = rec["labels"]
        names = rec["image_save_names"]
        for i in range(len(labels_ls)):
            out.append(
                FlatIndex(
                    batch_idx=b,
                    idx_in_batch=i,
                    label=int(labels_ls[i]),
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


def teacher_scores_for_group(
    teacher,
    labels: list[str],
    image_paths: list[Path],
    class_ids: list[int],
    ir_device: str,
    *,
    prompt_template: str = DEFAULT_IMAGE_REWARD_PROMPT,
) -> torch.Tensor:
    scores: list[float] = []
    for p, lab in zip(image_paths, class_ids):
        img = Image.open(p).convert("RGB")
        pr = prompt_for_class(lab, labels, template=prompt_template)
        scores.append(float(teacher.score(pr, img)))
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
) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
    order = list(range(len(dataset)))
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
                    teacher,
                    labels,
                    paths,
                    class_ids,
                    ir_device,
                    prompt_template=prompt_template,
                )
        yield tok, t_scores, int(class_ids[0])
