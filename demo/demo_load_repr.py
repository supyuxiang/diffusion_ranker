#!/usr/bin/env python3
"""
加载 ``DitRolloutReprCapture`` / ``DitSimpleRolloutReprCapture`` 保存的 ``.pt``，打印结构与张量形状。

用法::

    python3 demo/demo_load_repr.py /path/to/repr.pt
    python3 demo/demo_load_repr.py   # 使用下面默认路径
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

_DEFAULT_PT = Path("/root/autodl-tmp/repr_candidate_deno10_cfg1.0/idx0.pt")


def summarize_repr_dict(repr_d: dict, *, max_keys: int = 12) -> None:
    keys = sorted(repr_d.keys())
    print(f"    repr: {len(keys)} keys")
    for k in keys[:max_keys]:
        v = repr_d[k]
        if isinstance(v, dict):
            parts = []
            for ik, t in v.items():
                if torch.is_tensor(t):
                    parts.append(f"{ik}={tuple(t.shape)}/{t.dtype}")
                else:
                    parts.append(f"{ik}={type(t).__name__}")
            print(f"      {k!r}: {{{', '.join(parts)}}}")
        elif torch.is_tensor(v):
            print(f"      {k!r}: tensor {tuple(v.shape)} {v.dtype}")
        else:
            print(f"      {k!r}: {type(v).__name__}")
    if len(keys) > max_keys:
        print(f"      ... ({len(keys) - max_keys} more keys)")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect a rollout repr .pt file")
    p.add_argument(
        "pt_path",
        nargs="?",
        type=Path,
        default=_DEFAULT_PT,
        help=f"path to .pt (default: {_DEFAULT_PT})",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=0,
        help="which batches[batch] to summarize in detail",
    )
    args = p.parse_args()

    path = args.pt_path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")

    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"loaded: {path}")
    print(f"top-level type: {type(obj).__name__}")

    if not isinstance(obj, dict):
        print(obj)
        return

    print("top-level keys:", list(obj.keys()))
    meta = obj.get("meta")
    if isinstance(meta, dict):
        print("--- meta ---")
        for k, v in meta.items():
            print(f"  {k}: {v!r}")

    batches = obj.get("batches")
    if not isinstance(batches, list):
        return
    print(f"--- batches: len={len(batches)} ---")
    bi = args.batch
    if bi < 0 or bi >= len(batches):
        raise SystemExit(f"--batch {bi} out of range [0, {len(batches) - 1}]")
    b = batches[bi]
    if not isinstance(b, dict):
        print(f"batches[{bi}]: {type(b)}")
        return
    print(f"batches[{bi}] keys: {list(b.keys())}")
    for k, v in b.items():
        if k == "repr":
            summarize_repr_dict(v)
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)} (e.g. {v[:2]!r} ...)")
        else:
            print(f"  {k}: {v!r}")


if __name__ == "__main__":
    main()
