#!/usr/bin/env python3
"""
用本机已下载的 ImageReward 权重给**指定图片**打分（不触发 Hub 下载）。

默认权重目录（与 ``demo_imagereward_download_and_score.py download`` 一致）::

    ~/.cache/ImageReward/ImageReward.pt
    ~/.cache/ImageReward/med_config.json

也可用 ``--ckpt`` / ``--med_config`` 指向例如 ``/root/autodl-tmp/ImageReward/``。

示例::

    cd /root/diffusion_ranker
    python3 demo/demo_imagereward_score_image.py \\
      --image /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/image_00000_00000.png \\
      --prompt "a photo of a tench"

    # 多张图、同一 prompt
    python3 demo/demo_imagereward_score_image.py \\
      --image img1.png img2.png \\
      --prompt "a high-quality picture of a dog"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_CACHE = Path.home() / ".cache" / "ImageReward"


def main() -> None:
    p = argparse.ArgumentParser(description="ImageReward：对指定图片打分")
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(_DEFAULT_CACHE / "ImageReward.pt"),
        help="ImageReward.pt",
    )
    p.add_argument(
        "--med_config",
        type=str,
        default="",
        help="med_config.json；默认同 ckpt 所在目录",
    )
    p.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="一张或多张图片路径",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="a photo of a golden retriever",
        help="与图片对应的文本（训练时常用 a photo of a {class}）",
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    args = p.parse_args()

    import torch
    from PIL import Image

    from utils import ensure_image_reward_on_path

    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(
            f"找不到权重: {ckpt}\n请先下载: python3 demo/demo_imagereward_download_and_score.py download"
        )
    med = args.med_config.strip()
    if med:
        med_p = Path(med).expanduser().resolve()
        if not med_p.is_file():
            raise SystemExit(f"找不到 med_config: {med_p}")
        med = str(med_p)
    else:
        med_p = ckpt.parent / "med_config.json"
        if not med_p.is_file():
            raise SystemExit(
                f"找不到 {med_p}，请指定 --med_config 或把 med_config.json 与 .pt 放同目录"
            )
        med = str(med_p)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda 不可用，使用 cpu")
        device = "cpu"

    ensure_image_reward_on_path()
    import ImageReward as RM

    print(f"load: {ckpt}")
    model = RM.load(str(ckpt), device=device, med_config=med)
    model.eval()

    for path_str in args.image:
        ip = Path(path_str).expanduser().resolve()
        if not ip.is_file():
            print(f"[skip] 不是文件: {ip}")
            continue
        img = Image.open(ip).convert("RGB")
        s = float(model.score(args.prompt, img))
        print(f"{s:10.6f}  {ip}")


if __name__ == "__main__":
    main()
