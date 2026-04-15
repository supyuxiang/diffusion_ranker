#!/usr/bin/env python3
"""
将 ``rollout_trianing_data`` 生成的 ``repr.pt`` 按 **与 ``labeling.py`` / ``reward_ls.json`` 相同** 的
图顺序导出为 ``mean/``、``last/`` 下的大张量（每行一张图，无 list）。

顺序定义（与 ``ranker/labeling.py`` 一致）::

    sorted(Path(image_dir).glob("*.png"))

``reward_ls.json`` 须为长度 ``N`` 的 float 列表，且 ``N == len(sorted_png)``（仅校验）。

输出（在 ``--output_root`` 下）::

    mean/{repr_key}.pt
    last/{repr_key}.pt

其中 ``repr_key`` 与 ``repr.pt`` 内键一致，形如 ``s4_r0.037037``（``s``=去噪步索引，``r``=层比例），
**不再**用 ``step_index/(T-1)`` 拼进文件名，以免与稀疏 ``step_ratio_ls`` 混淆。

每个文件 ``torch.load`` 后为 dict，至少含 ``tensor`` shape ``(N, H)``、``repr_key``、``step_index``、
``layer_ratio``、``step_ratio``（仅作派生元数据）、``n_images``、``hidden_dim``、``row_order`` 等。

性能 / 内存：对 ``repr.pt`` 优先 ``mmap=True`` 加载；对 ``batches`` 只预计算一次每张图对应的行号，
每个 key 内用整段 ``tensor`` 索引写入，避免 Python 逐行循环与重复的 ``.float().cpu()`` 路径判断。

example:

python3 /root/diffusion_ranker/scripts/export_training_repr_tensors.py \
  --image_dir /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10 \
  --repr_pt /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/repr.pt \
  --reward_json /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/reward_ls.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_repr_pt(path: Path) -> dict:
    """Load repr checkpoint; mmap storages when possible to lower peak RSS."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        try:
            return torch.load(path, map_location="cpu", mmap=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except (RuntimeError, OSError):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _to_float_cpu(t: torch.Tensor) -> torch.Tensor:
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if t.dtype != torch.float32:
        t = t.float()
    return t


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


def _filename_for_key(repr_key: str) -> str:
    """文件名 = ``repr.pt`` 中的键名 + ``.pt``，与 ``s{步索引}_r{层比例}`` 一一对应。"""
    if "/" in repr_key or repr_key.startswith("."):
        raise ValueError(f"unsafe repr_key for filename: {repr_key!r}")
    return f"{repr_key}.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image_dir",
        type=Path,
        default=Path("/root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10"),
    )
    p.add_argument(
        "--repr_pt",
        type=Path,
        default=None,
        help="默认: <image_dir>/repr.pt",
    )
    p.add_argument(
        "--reward_json",
        type=Path,
        default=None,
        help="默认: <image_dir>/reward_ls.json",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="默认与 --image_dir 相同（在其下创建 mean/、last/）",
    )
    args = p.parse_args()

    image_dir: Path = args.image_dir.expanduser().resolve()
    repr_pt = (
        args.repr_pt.expanduser().resolve()
        if args.repr_pt
        else image_dir / "repr.pt"
    )
    reward_json = (
        args.reward_json.expanduser().resolve()
        if args.reward_json
        else image_dir / "reward_ls.json"
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else image_dir
    )

    if not image_dir.is_dir():
        raise SystemExit(f"missing image_dir: {image_dir}")
    if not repr_pt.is_file():
        raise SystemExit(f"missing repr_pt: {repr_pt}")
    if not reward_json.is_file():
        raise SystemExit(f"missing reward_json: {reward_json}")

    sorted_png = sorted(image_dir.glob("*.png"))
    n = len(sorted_png)
    name_to_row = {path.name: i for i, path in enumerate(sorted_png)}

    with open(reward_json, encoding="utf-8") as f:
        reward_ls = json.load(f)
    if not isinstance(reward_ls, list):
        raise SystemExit("reward_ls.json must be a JSON list of floats (labeling.py format)")
    if len(reward_ls) != n:
        raise SystemExit(
            f"reward_ls length {len(reward_ls)} != sorted *.png count {n} under {image_dir}"
        )

    obj = _load_repr_pt(repr_pt)

    meta = dict(obj.get("meta") or {})
    batches = obj["batches"]
    b0 = batches[0]
    all_keys = sorted(b0["repr"].keys(), key=lambda k: (_parse_repr_key(k)[0], _parse_repr_key(k)[1]))

    h0 = int(b0["repr"][all_keys[0]]["mean"].shape[1])
    total_in_repr = sum(len(rec["image_save_names"]) for rec in batches)
    if total_in_repr != n:
        raise SystemExit(
            f"repr.pt 中图像条数 {total_in_repr} 与 sorted *.png 数量 {n} 不一致"
        )

    batch_row_idx: list[torch.Tensor] = []
    for rec in batches:
        names = rec["image_save_names"]
        rows: list[int] = []
        for fn in names:
            row = name_to_row.get(Path(fn).name)
            if row is None:
                raise KeyError(
                    f"repr image {fn!r} not found in sorted *.png listing for {image_dir}"
                )
            rows.append(row)
        batch_row_idx.append(torch.tensor(rows, dtype=torch.long))

    mean_dir = output_root / "mean"
    last_dir = output_root / "last"
    mean_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    for key in all_keys:
        mean_buf = torch.full((n, h0), float("nan"), dtype=torch.float32)
        last_buf = torch.full((n, h0), float("nan"), dtype=torch.float32)
        for rec, row_idx in zip(batches, batch_row_idx):
            rd = rec["repr"][key]
            m = _to_float_cpu(rd["mean"])
            lt = _to_float_cpu(rd["last"])
            mean_buf[row_idx] = m
            last_buf[row_idx] = lt

        bad_rows = torch.isnan(mean_buf).any(dim=1) | torch.isnan(last_buf).any(dim=1)
        if bad_rows.any():
            missing = int(bad_rows.sum().item())
            raise RuntimeError(
                f"key {key!r}: {missing} rows never filled (repr batches missing some png names?)"
            )

        fname = _filename_for_key(key)
        s_idx, layer_r = _parse_repr_key(key)
        payload_base = {
            "tensor": None,
            "n_images": n,
            "hidden_dim": h0,
            "repr_key": key,
            "step_index": s_idx,
            "layer_ratio": layer_r,
            "step_ratio": _step_ratio(s_idx, int(meta.get("num_inference_steps") or 1)),
            "row_order": "sorted(image_dir.glob('*.png')) same as ranker/labeling.py and reward_ls.json",
            "image_dir": str(image_dir),
            "repr_pt": str(repr_pt),
            "reward_json": str(reward_json),
        }
        torch.save({**payload_base, "tensor": mean_buf}, mean_dir / fname)
        torch.save({**payload_base, "tensor": last_buf}, last_dir / fname)

    del obj
    print(
        f"done: {len(all_keys)} keys × (mean, last) -> {mean_dir} , {last_dir} ; N={n}, H={h0}"
    )


if __name__ == "__main__":
    main()
