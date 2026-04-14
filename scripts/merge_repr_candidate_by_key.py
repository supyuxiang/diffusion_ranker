#!/usr/bin/env python3
"""
将 ``repr_candidate_deno10_cfg1.0`` 风格目录下多个 ``idx*.pt`` 按 key 合并为按图顺序的表征文件。

- 在**同目录**下创建 ``mean/``、``last/`` 子目录。
- 每个 repr key（如 ``s0_r0.1``）对应两个文件，文件名：
  ``step_ratio{step_ratio}_layer_ratio{layer_ratio}.pt``
  其中 ``step_ratio = step_index / max(T-1,1)``，``layer_ratio`` 为 key 中的 ``r``（与 rollout meta 一致）。
- 默认每个 ``.pt`` 为字典，含 ``tensor (N,H)`` 及 ``repr_key``、``step_ratio``、``layer_ratio`` 等。
- ``--as_list`` 时额外写入 ``repr_list``：每张图一个 ``(H,)`` tensor，顺序与 ``tensor`` 行一致。

example:

python3 /root/diffusion_ranker/scripts/merge_repr_candidate_by_key.py \
 --repr_dir /root/autodl-tmp/repr_candidate_deno10_cfg1.0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


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

    first = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    meta0 = dict(first.get("meta") or {})
    b0 = first["batches"][0]
    all_keys = sorted(b0["repr"].keys(), key=lambda k: (_parse_repr_key(k)[0], _parse_repr_key(k)[1]))

    print(f"repr_dir={repr_dir}")
    print(f"idx pt files: {len(pt_files)}  keys per batch: {len(all_keys)}")

    for key in all_keys:
        mean_chunks: list[torch.Tensor] = []
        last_chunks: list[torch.Tensor] = []
        for fp in pt_files:
            obj = torch.load(fp, map_location="cpu", weights_only=False)
            for rec in obj["batches"]:
                rd = rec["repr"][key]
                mean_chunks.append(rd["mean"].float().cpu())
                last_chunks.append(rd["last"].float().cpu())

        mean_cat = torch.cat(mean_chunks, dim=0)
        last_cat = torch.cat(last_chunks, dim=0)
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

    print(f"wrote {len(all_keys)} files each under {mean_dir} and {last_dir}")


if __name__ == "__main__":
    main()
