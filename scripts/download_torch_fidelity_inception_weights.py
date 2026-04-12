#!/usr/bin/env python3
"""预下载 torch-fidelity / torchmetrics FID 用的 Inception V3 权重到固定目录。

默认写入 ``<repo>/third_party/torch_fidelity_weights/``，与 ``utils.resolve_fidelity_inception_weights_path``
搜索顺序一致，无需再设环境变量。

用法::

    python3 scripts/download_torch_fidelity_inception_weights.py
    python3 scripts/download_torch_fidelity_inception_weights.py --out-dir /root/autodl-tmp/torch_fidelity_weights

下载失败会依次尝试多个 URL（GitHub release、常见镜像前缀）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# 与 ``utils.FIDELITY_INCEPTION_WEIGHTS_FILENAME`` 保持一致
FIDELITY_INCEPTION_WEIGHTS_FILENAME = "weights-inception-2015-12-05-6726825d.pth"

_DEFAULT_URLS = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
    + FIDELITY_INCEPTION_WEIGHTS_FILENAME,
    "https://ghproxy.net/https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
    + FIDELITY_INCEPTION_WEIGHTS_FILENAME,
    "https://mirror.ghproxy.com/https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
    + FIDELITY_INCEPTION_WEIGHTS_FILENAME,
)


def _download(url: str, dest: Path, chunk: int = 8 * 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "diffusion_ranker-download-script/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = resp.headers.get("Content-Length")
            n_total = int(total) if total and total.isdigit() else None
            n_read = 0
            with open(tmp, "wb") as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    n_read += len(block)
                    if n_total:
                        pct = 100.0 * n_read / n_total
                        print(f"\r  {n_read // (1024 * 1024)} MiB / {n_total // (1024 * 1024)} MiB ({pct:.1f}%)", end="", flush=True)
        print()
        tmp.replace(dest)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "third_party" / "torch_fidelity_weights",
        help="权重保存目录（文件名固定为 utils 中的常量）",
    )
    p.add_argument(
        "--url",
        action="append",
        default=[],
        help="额外优先尝试的下载 URL（可多次指定；先于内置镜像列表）",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="已存在同名文件时仍重新下载",
    )
    args = p.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    dest = out_dir / FIDELITY_INCEPTION_WEIGHTS_FILENAME

    if dest.is_file() and not args.force:
        print(f"已存在，跳过: {dest}")
        print(f"export TORCH_FIDELITY_INCEPTION_WEIGHTS={dest}")
        return

    urls = list(args.url) + list(_DEFAULT_URLS)
    last_err: Exception | None = None
    for url in urls:
        print(f"try: {url[:80]}{'...' if len(url) > 80 else ''}")
        try:
            _download(url, dest)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last_err = e
            print(f"  failed: {e}")
    else:
        print("所有 URL 均失败。", file=sys.stderr)
        if last_err:
            raise last_err
        raise RuntimeError("no download URLs")

    # 与 torch.hub 缓存布局一致时，可选再拷一份（便于其它库默认路径命中）
    hub_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / FIDELITY_INCEPTION_WEIGHTS_FILENAME
    try:
        hub_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, hub_ckpt)
        print(f"also copied to: {hub_ckpt}")
    except OSError as e:
        print(f"note: could not copy to hub cache: {e}")

    print(f"done: {dest}")
    print(f"export TORCH_FIDELITY_INCEPTION_WEIGHTS={dest}")


if __name__ == "__main__":
    main()
