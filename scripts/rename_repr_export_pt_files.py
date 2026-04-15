#!/usr/bin/env python3
r"""
将旧版导出文件名 ``step_ratio{sr}_layer_ratio{r}.pt`` 改为与 ``repr.pt`` 键一致：``s{step_index}_r{r}.pt``。

- ``step_index = round(sr * (T - 1))``，``T`` = ``--deno_steps``（与导出时 ``num_inference_steps`` 一致）。
- ``r`` 部分保留原文件名中的字符串（避免 float 再格式化丢精度）。

默认对每个 ``--root`` 下的 ``mean/``、``last/`` 分别处理（若目录存在）。

example::

    python3 scripts/rename_repr_export_pt_files.py \\
        --root /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10 \\
        --root /root/autodl-tmp/repr_candidate_deno10_cfg1.0 \\
        --deno_steps 10
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_OLD_RE = re.compile(r"^step_ratio([^_]+)_layer_ratio(.+)\.pt$")


def old_name_to_new(name: str, *, deno_steps: int) -> str | None:
    m = _OLD_RE.match(name)
    if not m:
        return None
    sr = float(m.group(1))
    lr_str = m.group(2).strip()
    float(lr_str)  # validate
    t = max(int(deno_steps), 1)
    if t <= 1:
        s_idx = 0
    else:
        s_idx = int(round(sr * (t - 1)))
        s_idx = max(0, min(t - 1, s_idx))
    return f"s{s_idx}_r{lr_str}.pt"


def rename_in_dir(d: Path, *, deno_steps: int, dry_run: bool) -> int:
    if not d.is_dir():
        return 0
    n = 0
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.suffix != ".pt":
            continue
        new_name = old_name_to_new(p.name, deno_steps=deno_steps)
        if new_name is None:
            continue
        dest = d / new_name
        if p.name == new_name:
            continue
        if dest.exists():
            raise SystemExit(
                f"目标已存在，拒绝覆盖: {dest}\n"
                f"  源: {p}\n"
                f"  请先检查是否重复映射。"
            )
        if dry_run:
            print(f"DRY  {p.name}  ->  {new_name}")
        else:
            p.rename(dest)
            print(f"OK   {p.name}  ->  {new_name}")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        action="append",
        default=[],
        help="数据根目录（其下处理 mean/、last/）；可多次指定",
    )
    ap.add_argument(
        "--dir",
        type=Path,
        action="append",
        default=[],
        help="直接处理某一目录内的旧文件名（可多次指定）",
    )
    ap.add_argument("--deno_steps", type=int, default=10)
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将要执行的重命名，不写盘",
    )
    args = ap.parse_args()

    total = 0
    dirs: list[Path] = []
    for r in args.root:
        r = r.expanduser().resolve()
        for sub in ("mean", "last"):
            dirs.append(r / sub)
    for d in args.dir:
        dirs.append(d.expanduser().resolve())

    if not dirs:
        raise SystemExit("请至少指定 --root 或 --dir")

    for d in dirs:
        k = rename_in_dir(d, deno_steps=args.deno_steps, dry_run=args.dry_run)
        if k:
            print(f"# {d}: {k} file(s)")
        total += k

    print(f"done: {total} renames" + (" (dry_run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
