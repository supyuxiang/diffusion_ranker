#!/usr/bin/env python3
"""
将 ``rollout_trianing_data`` 生成的 ``repr.pt`` 按 **与 ``labeling.py`` / ``reward_ls.json`` 相同** 的
图顺序导出为 ``mean/``、``last/`` 下的大张量（每行一张图，无 list）。

顺序定义（与 ``ranker/labeling.py`` 一致）::

    sorted(Path(image_dir).glob("*.png"))

``reward_ls.json`` 须为长度 ``N`` 的 float 列表，且 ``N == len(sorted_png)``（仅校验）。

输出（在 ``--output_root`` 下）::

- **默认（``--name_style auto``）**：若同时按层/步筛选（如 ``--preset sparse``），文件名为约定比例：  
  ``step_{步比}_layer_{层比}.pt``，例如 ``step_0.25_layer_0.3.pt``；dict 内仍含 ``repr_key`` 指向真实键。  
- 全量导出或 ``--name_style repr``：文件名为 ``{repr_key}.pt``，与 ``repr.pt`` 键一致。

每个文件 ``torch.load`` 后为 dict，至少含 ``tensor`` shape ``(N, H)``、``repr_key``、``step_index``、
``layer_ratio``、``step_ratio``（仅作派生元数据）、``n_images``、``hidden_dim``、``row_order`` 等。

性能 / 内存：对 ``repr.pt`` 优先 ``mmap=True`` 加载；对 ``batches`` 只预计算一次每张图对应的行号，
每个 key 内用整段 ``tensor`` 索引写入，避免 Python 逐行循环与重复的 ``.float().cpu()`` 路径判断。

example:

python3 /root/diffusion_ranker/scripts/export_training_repr_tensors.py \
  --image_dir /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10 \
  --repr_pt /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/repr.pt \
  --reward_json /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/reward_ls.json

只导出与 ``ranker/train.py`` 相同的「稀疏」层/步约定（dense ``repr.pt`` 内会映射到对应 ``s*_r*`` 键）::

python3 scripts/export_training_repr_tensors.py \\
  --image_dir /path/to/dit_training_data_... \\
  --preset sparse --purge_output_dirs

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ranker.train import infer_num_steps, ratio_to_block_idx, select_repr_keys

# 与 ``rollout_trianing_data.py --repr_layout sparse`` / ``DitRolloutReprCapture`` 默认网格一致
_SPARSE_LAYER_RATIOS: list[float] = [i / 10.0 for i in range(1, 11)]
_SPARSE_STEP_RATIOS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]


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
    """文件名 = ``repr.pt`` 中的键名 + ``.pt``。"""
    if "/" in repr_key or repr_key.startswith("."):
        raise ValueError(f"unsafe repr_key for filename: {repr_key!r}")
    return f"{repr_key}.pt"


def _fmt_ratio_label(x: float) -> str:
    """稳定短字符串：``0.25``、``1`` 等，避免科学计数。"""
    t = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return t if t else "0"


def _user_layer_ratio_for_repr_r(
    r_key: float, layer_list: list[float], n_bl: int
) -> float:
    bi = ratio_to_block_idx(r_key, n_bl)
    for u in layer_list:
        if ratio_to_block_idx(u, n_bl) == bi:
            return float(u)
    raise RuntimeError(
        f"无法在 layer_list 中为 r={r_key!r} (block_idx={bi}) 找到约定层比"
    )


def _user_step_ratio_for_index(
    s_idx: int, step_list: list[float], num_steps: int
) -> float:
    t = max(int(num_steps), 1)
    nb = max(t - 1, 0)
    for v in step_list:
        vv = max(0.0, min(1.0, float(v)))
        si = int(round(vv * nb)) if nb else 0
        if si == s_idx:
            return float(v)
    raise RuntimeError(
        f"无法在 step_list 中为 step_index={s_idx} (T={t}) 找到约定步比"
    )


def _semantic_export_filename(
    repr_key: str,
    *,
    layer_list: list[float],
    step_list: list[float],
    meta: dict,
    num_steps: int,
) -> str:
    s_idx, r_key = _parse_repr_key(repr_key)
    n_bl = meta.get("n_transformer_blocks")
    if n_bl is None:
        raise SystemExit("semantic 命名需要 repr.pt meta 含 n_transformer_blocks")
    n_bl = int(n_bl)
    u_layer = _user_layer_ratio_for_repr_r(r_key, layer_list, n_bl)
    u_step = _user_step_ratio_for_index(s_idx, step_list, num_steps)
    stem = f"step_{_fmt_ratio_label(u_step)}_layer_{_fmt_ratio_label(u_layer)}"
    if "/" in stem or stem.startswith("."):
        raise ValueError(f"unsafe semantic stem: {stem!r}")
    return f"{stem}.pt"


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
    p.add_argument(
        "--preset",
        choices=("", "sparse"),
        default="",
        help="sparse：层 0.1…1.0、步 0/0.25/0.5/0.75/1；可用 --layer_ratios/--step_ratios 覆盖其中一部分",
    )
    p.add_argument(
        "--layer_ratios",
        default="",
        help="逗号分隔，语义同 train.py；非空则按层过滤（覆盖 preset 的层列表）",
    )
    p.add_argument(
        "--step_ratios",
        default="",
        help="逗号分隔，语义同 train.py；非空则按步过滤（覆盖 preset 的步列表）",
    )
    p.add_argument(
        "--purge_output_dirs",
        action="store_true",
        help="写入前删除 mean/、last/ 下全部 .pt，避免残留未筛选键",
    )
    p.add_argument(
        "--name_style",
        choices=("auto", "repr", "semantic"),
        default="auto",
        help="auto：同时有层/步筛选时用约定比例文件名 step_X_layer_Y.pt；否则用 repr 键名",
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
    all_keys_raw = sorted(
        b0["repr"].keys(),
        key=lambda k: (_parse_repr_key(k)[0], _parse_repr_key(k)[1]),
    )

    layer_f: list[float] | None = None
    step_f: list[float] | None = None
    if args.preset == "sparse":
        layer_f = list(_SPARSE_LAYER_RATIOS)
        step_f = list(_SPARSE_STEP_RATIOS)
    if args.layer_ratios.strip():
        layer_f = [
            float(x.strip()) for x in args.layer_ratios.split(",") if x.strip()
        ]
    if args.step_ratios.strip():
        step_f = [float(x.strip()) for x in args.step_ratios.split(",") if x.strip()]

    if layer_f is not None or step_f is not None:
        all_keys = select_repr_keys(
            all_keys_raw,
            meta=meta,
            layer_ratios=layer_f,
            step_ratios=step_f,
        )
        print(
            f"filter: preset={args.preset!r} layer_ratios={layer_f} step_ratios={step_f} "
            f"-> {len(all_keys)} keys (of {len(all_keys_raw)})"
        )
        if not all_keys:
            raise SystemExit(
                "筛选后没有键：请检查 repr.pt 是否与 layer/step 约定一致，"
                "或去掉 --preset/--layer_ratios/--step_ratios 导出全部键"
            )
    else:
        all_keys = all_keys_raw

    if args.name_style == "repr":
        use_semantic_names = False
    elif args.name_style == "semantic":
        if layer_f is None or step_f is None:
            raise SystemExit(
                "--name_style semantic 需要同时指定层与步（例如 --preset sparse）"
            )
        use_semantic_names = True
    else:
        use_semantic_names = layer_f is not None and step_f is not None

    num_steps_infer = infer_num_steps(all_keys_raw, meta)

    h0 = int(b0["repr"][all_keys_raw[0]]["mean"].shape[1])
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

    if args.purge_output_dirs:
        for d in (mean_dir, last_dir):
            for fp in d.glob("*.pt"):
                fp.unlink()

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

        if use_semantic_names:
            assert layer_f is not None and step_f is not None
            fname = _semantic_export_filename(
                key,
                layer_list=layer_f,
                step_list=step_f,
                meta=meta,
                num_steps=num_steps_infer,
            )
        else:
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
            "export_filename": fname,
            "name_style": "semantic" if use_semantic_names else "repr",
        }
        if use_semantic_names:
            payload_base["user_layer_ratio"] = _user_layer_ratio_for_repr_r(
                layer_r, layer_f, int(meta["n_transformer_blocks"])
            )
            payload_base["user_step_ratio"] = _user_step_ratio_for_index(
                s_idx, step_f, num_steps_infer
            )
        torch.save({**payload_base, "tensor": mean_buf}, mean_dir / fname)
        torch.save({**payload_base, "tensor": last_buf}, last_dir / fname)

    del obj
    print(
        f"done: {len(all_keys)} keys × (mean, last) -> {mean_dir} , {last_dir} ; N={n}, H={h0}"
    )


if __name__ == "__main__":
    main()
