#!/usr/bin/env python3
"""
从 autodl 公共盘上的 ILSVRC2012 devkit（无需解压整包）读取 ``meta.mat``，生成
``imagenet-simple-labels.json``（1000 个字符串，下标与 PyTorch/DiT ``class_labels`` 的 WNID 字典序一致）。

完全离线；依赖 ``scipy``。生成后 ``utils.imagenet_idx2text`` 只读该 json 路径。

用法::

    python3 scripts/build_imagenet_simple_labels.py
    python3 scripts/build_imagenet_simple_labels.py \
      --devkit-tar /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz \
      --out /root/diffusion_ranker/imagenet-simple-labels.json
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import scipy.io as sio

_META_MEMBER = "ILSVRC2012_devkit_t12/data/meta.mat"


def _short_label(words: str) -> str:
    s = str(words).strip()
    return s.split(",", 1)[0].strip() if "," in s else s


def labels_from_devkit_t12(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path, "r:gz") as tf:
        m = tf.getmember(_META_MEMBER)
        bio = tf.extractfile(m)
        if bio is None:
            raise RuntimeError(f"cannot read {_META_MEMBER} from {tar_path}")
        raw = bio.read()
    mat = sio.loadmat(io.BytesIO(raw))
    syn = mat["synsets"]
    rows: list[tuple[str, str]] = []
    for i in range(syn.shape[0]):
        s = syn[i, 0]
        iid = int(s["ILSVRC2012_ID"][0, 0])
        if not (1 <= iid <= 1000):
            continue
        wnid = str(s["WNID"][0])
        w = s["words"][0]
        w = str(w.item()) if isinstance(w, np.ndarray) else str(w)
        rows.append((wnid, _short_label(w)))
    rows.sort(key=lambda x: x[0])
    if len(rows) != 1000:
        raise RuntimeError(f"expected 1000 classes, got {len(rows)}")
    return [lbl for _, lbl in rows]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--devkit-tar",
        type=Path,
        default=Path("/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/root/diffusion_ranker/imagenet-simple-labels.json"),
    )
    args = p.parse_args()
    tar_path = args.devkit_tar.expanduser().resolve()
    if not tar_path.is_file():
        raise SystemExit(f"devkit tar not found: {tar_path}")
    labels = labels_from_devkit_t12(tar_path)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(labels)} labels to {out}")


if __name__ == "__main__":
    main()
