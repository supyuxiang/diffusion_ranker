"""FK steering 公共：累积 log-belief + 重采样 + 选 candidate。"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch


def fk_update_state(
    log_belief: torch.Tensor,
    scores: torch.Tensor,
    *,
    tau: float,
    do_resample: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tau <= 0:
        raise ValueError("tau must be positive")
    log_belief = log_belief + scores / float(tau)
    w = torch.softmax(log_belief, dim=0)
    if not do_resample:
        return log_belief, w
    k = int(w.shape[0])
    idx = torch.multinomial(w, num_samples=k, replacement=True)
    counts = torch.bincount(idx, minlength=k).float()
    w = counts / counts.sum().clamp(min=1e-12)
    log_belief = torch.log(w.clamp(min=1e-20))
    return log_belief, w


def pick_candidate(w: torch.Tensor, *, pick: str) -> int:
    if pick == "argmax":
        return int(torch.argmax(w).item())
    if pick == "sample":
        return int(torch.multinomial(w, num_samples=1).item())
    raise ValueError(f"pick must be argmax|sample, got {pick!r}")


def save_best_images_preserve_names(best_path_ls: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in best_path_ls:
        shutil.copy2(path, output_dir / path.name)
    print(f"saved {len(best_path_ls)} best images to {output_dir}")


def save_best_images_fid_layout(best_path_ls: list[Path], output_dir: Path) -> None:
    """与 ``dit_random_search_fid_reward.save_best_images`` 一致：``image_{i:05d}_{class:05d}.png``。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(best_path_ls):
        class_id = int(path.stem.split("_")[-1])
        dest = output_dir / f"image_{i:05d}_{class_id:05d}.png"
        shutil.copy2(path, dest)
