#!/usr/bin/env python3
"""
ImageReward：先下载权重，再对单张图打分。

**下载前**（AutoDL 等）建议在 shell 里先::

    source /etc/network_turbo

**子命令**

1. ``download`` — 从 HuggingFace 拉 ``ImageReward.pt`` 与 ``med_config.json``（仅用 ``huggingface_hub``，不加载模型）。
2. ``score`` — 加载本地权重并 ``model.score(prompt, pil_image)``。

示例::

    cd /root/diffusion_ranker

    # 1) 下载到默认目录 ~/.cache/ImageReward（或 --out）
    python3 demo/demo_imagereward_download_and_score.py download
    python3 demo/demo_imagereward_download_and_score.py download --out /root/autodl-tmp/ImageReward

    # 2) 打分（默认用 ~/.cache/ImageReward 下的文件）
    python3 demo/demo_imagereward_download_and_score.py score \\
      --image /path/to/image.png \\
      --prompt "a photo of a dog"

    # 显式指定权重目录
    python3 demo/demo_imagereward_download_and_score.py score \\
      --ckpt /root/autodl-tmp/ImageReward/ImageReward.pt \\
      --med_config /root/autodl-tmp/ImageReward/med_config.json \\
      --image /path/to/image.png \\
      --prompt "a photo of a dog"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_OUT = Path.home() / ".cache" / "ImageReward"
REPO_ID = "THUDM/ImageReward"
FILES = ("ImageReward.pt", "med_config.json")


def cmd_download(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for fn in FILES:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=fn,
            local_dir=str(out),
        )
        print(f"ok  {fn} -> {path}")
    print(f"\n完成。之后打分可用: --ckpt {out / 'ImageReward.pt'}")


def cmd_score(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image

    from utils import ensure_image_reward_on_path

    ensure_image_reward_on_path()
    import ImageReward as RM

    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"找不到权重: {ckpt}\n请先运行: python3 demo/demo_imagereward_download_and_score.py download")

    med = args.med_config
    if med:
        med = str(Path(med).expanduser().resolve())
        if not Path(med).is_file():
            raise SystemExit(f"找不到 med_config: {med}")
    else:
        cand = ckpt.parent / "med_config.json"
        med = str(cand if cand.is_file() else (DEFAULT_OUT / "med_config.json"))
        if not Path(med).is_file():
            raise SystemExit(
                f"找不到 med_config.json，请指定 --med_config 或先 download 到同目录\n尝试路径: {med}"
            )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: cuda 不可用，改用 cpu")
        device = "cpu"

    print(f"加载 ImageReward: {ckpt}")
    model = RM.load(str(ckpt), device=device, med_config=med)
    model.eval()

    img_path = Path(args.image).expanduser().resolve()
    if not img_path.is_file():
        raise SystemExit(f"找不到图片: {img_path}")

    img = Image.open(img_path).convert("RGB")
    score = float(model.score(args.prompt, img))
    print(f"prompt: {args.prompt!r}")
    print(f"image:  {img_path}")
    print(f"score:  {score:.6f}")


def main() -> None:
    p = argparse.ArgumentParser(description="ImageReward 下载 + 打分 demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("download", help="下载 ImageReward.pt 与 med_config.json")
    pd.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help=f"保存目录（默认 {DEFAULT_OUT}）",
    )
    pd.set_defaults(func=cmd_download)

    ps = sub.add_parser("score", help="加载本地权重并打分")
    ps.add_argument(
        "--ckpt",
        type=str,
        default=str(DEFAULT_OUT / "ImageReward.pt"),
        help="ImageReward.pt 路径",
    )
    ps.add_argument(
        "--med_config",
        type=str,
        default="",
        help="med_config.json；默认与 ckpt 同目录或 ~/.cache/ImageReward",
    )
    ps.add_argument("--image", type=str, required=True, help="图片路径")
    ps.add_argument(
        "--prompt",
        type=str,
        default="a photo of a golden retriever",
        help="文本 prompt",
    )
    ps.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="无 GPU 时会自动改 cpu",
    )
    ps.set_defaults(func=cmd_score)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
