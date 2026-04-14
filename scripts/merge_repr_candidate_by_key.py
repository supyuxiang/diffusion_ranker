#!/usr/bin/env python3
"""
将 ``repr_candidate_deno10_cfg1.0`` 风格目录下多个 ``idx*.pt`` 按 key 合并为按图顺序的表征文件。

- 在**同目录**下创建 ``mean/``、``last/`` 子目录。
- 每个 repr key（如 ``s0_r0.1``）对应两个文件，文件名：
  ``step_ratio{step_ratio}_layer_ratio{layer_ratio}.pt``
  其中 ``step_ratio = step_index / max(T-1,1)``，``layer_ratio`` 为 key 中的 ``r``（与 rollout meta 一致）。
- 默认每个 ``.pt`` 为字典，含 ``tensor (N,H)`` 及 ``repr_key``、``step_ratio``、``layer_ratio`` 等。
- ``--as_list`` 时额外写入 ``repr_list``：每张图一个 ``(H,)`` tensor，顺序与 ``tensor`` 行一致。

内存：对每个 repr key 分两次扫 ``idx*.pt``（先拼 mean 再拼 last），并用 ``torch.load(..., mmap=True)``（失败则回退）与按文件合并 batch，降低峰值内存。

example:

python3 /root/diffusion_ranker/scripts/merge_repr_candidate_by_key.py \
 --repr_dir /root/autodl-tmp/repr_candidate_deno10_cfg1.0
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import torch


def _load_repr_pt(path: Path) -> dict:
    """Load idx*.pt; mmap tensor storages when possible to lower peak RSS."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except (RuntimeError, OSError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _to_float_cpu(t: torch.Tensor) -> torch.Tensor:
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if t.dtype != torch.float32:
        t = t.float()
    return t


def _concat_repr_for_key_from_obj(obj: dict, key: str, field: str) -> torch.Tensor:
    """All batches in one file for one key and one of mean/last -> (N_file, H)."""
    parts: list[torch.Tensor] = []
    for rec in obj["batches"]:
        parts.append(_to_float_cpu(rec["repr"][key][field]))
    if not parts:
        raise RuntimeError(f"empty batches for key {key!r}")
    return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]


def _parse_repr_key(k: str) -> tuple[int, float]:
    a, b = k.split("_", 1)
    s = int(a[1:])
    r = float(b[1:])
    return s, r


def _step_ratio(step_idx: int, num_inference_steps: int) -> float:
    t = max(int(num_inference_steps), 1)
    if t <= 1:
        return 0.0
    return float(step_idx) / float(t - 1)


def _filename_for_key(meta: dict, repr_key: str) -> str:
    s, r = _parse_repr_key(repr_key)
    t = int(meta.get("num_inference_steps") or 1)
    sr = _step_ratio(s, t)
    # 文件名避免空格；g 格式足够区分
    return f"step_ratio{sr:g}_layer_ratio{r:g}.pt"


def _natural_idx_pt_files(repr_dir: Path) -> list[Path]:
    def sort_key(p: Path) -> tuple[int, str]:
        m = re.match(r"^idx(\d+)\.pt$", p.name)
        if m:
            return (int(m.group(1)), p.name)
        return (10**9, p.name)

    files = sorted(repr_dir.glob("idx*.pt"), key=sort_key)
    return [p for p in files if p.is_file()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repr_dir",
        type=Path,
        default=Path("/root/autodl-tmp/repr_candidate_deno10_cfg1.0"),
    )
    p.add_argument(
        "--as_list",
        action="store_true",
        help="在 .pt 中增加 repr_list（每张图一个 1D tensor，占空间更大）",
    )
    args = p.parse_args()
    repr_dir: Path = args.repr_dir.expanduser().resolve()
    if not repr_dir.is_dir():
        raise SystemExit(f"not a directory: {repr_dir}")

    pt_files = _natural_idx_pt_files(repr_dir)
    if not pt_files:
        raise SystemExit(f"no idx*.pt under {repr_dir}")

    mean_dir = repr_dir / "mean"
    last_dir = repr_dir / "last"
    mean_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    meta0: dict = {}
    all_keys: list[str] = []

    first = _load_repr_pt(pt_files[0])
    meta0 = dict(first.get("meta") or {})
    b0 = first["batches"][0]
    all_keys = sorted(b0["repr"].keys(), key=lambda k: (_parse_repr_key(k)[0], _parse_repr_key(k)[1]))
    del first
    gc.collect()

    print(f"repr_dir={repr_dir}")
    print(f"idx pt files: {len(pt_files)}  keys per batch: {len(all_keys)}")

    for key in all_keys:
        # Two passes over idx*.pt: peak RAM ~= one checkpoint + one branch (mean or last)
        # of per-file chunks, instead of holding both mean and last chunk lists at once.
        mean_chunks: list[torch.Tensor] = []
        for fp in pt_files:
            obj = _load_repr_pt(fp)
            try:
                mean_chunks.append(_concat_repr_for_key_from_obj(obj, key, "mean"))
            finally:
                del obj

        mean_cat = torch.cat(mean_chunks, dim=0)
        del mean_chunks

        last_chunks: list[torch.Tensor] = []
        for fp in pt_files:
            obj = _load_repr_pt(fp)
            try:
                last_chunks.append(_concat_repr_for_key_from_obj(obj, key, "last"))
            finally:
                del obj

        last_cat = torch.cat(last_chunks, dim=0)
        del last_chunks
        gc.collect()

        assert mean_cat.shape == last_cat.shape

        fname = _filename_for_key(meta0, key)
        payload_mean = {
            "tensor": mean_cat,
            "n_images": int(mean_cat.shape[0]),
            "hidden_dim": int(mean_cat.shape[1]),
            "repr_key": key,
            "step_ratio": _step_ratio(_parse_repr_key(key)[0], int(meta0.get("num_inference_steps") or 1)),
            "layer_ratio": _parse_repr_key(key)[1],
            "source_idx_pt": [f.name for f in pt_files],
        }
        if args.as_list:
            payload_mean["repr_list"] = [row.contiguous() for row in mean_cat.unbind(0)]
        payload_last = {**payload_mean, "tensor": last_cat}
        if args.as_list:
            payload_last["repr_list"] = [row.contiguous() for row in last_cat.unbind(0)]

        torch.save(payload_mean, mean_dir / fname)
        torch.save(payload_last, last_dir / fname)

        del mean_cat, last_cat, payload_mean, payload_last
        gc.collect()

    print(f"wrote {len(all_keys)} files each under {mean_dir} and {last_dir}")


if __name__ == "__main__":
    main()
