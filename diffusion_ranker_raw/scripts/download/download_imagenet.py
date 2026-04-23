#!/usr/bin/env python3
"""
下载 ImageNet 到指定目录（解决 Permission denied 问题）

Ultralytics 自带的 get_imagenet.sh 使用相对路径 ../datasets/imagenet，
在无写权限的目录下会失败。本脚本将数据下载到可写目录。

用法:
    python scripts/download_imagenet.py
    python scripts/download_imagenet.py --save_dir /data/yxfeng/datasets
    python scripts/download_imagenet.py --val_only  # 仅下载验证集（6.3G，用于快速测试）

下载完成后，设置 Ultralytics：
    yolo settings datasets_dir=/data/yxfeng/datasets
"""

import os
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="/data/yxfeng/datasets",
        help="数据集根目录，ImageNet 将保存到 save_dir/imagenet/",
    )
    parser.add_argument(
        "--val_only",
        action="store_true",
        help="仅下载验证集 (6.3G)，用于快速测试",
    )
    parser.add_argument(
        "--skip_valprep",
        action="store_true",
        help="跳过 valprep（解压不完整时使用，验证集将保持扁平结构）",
    )
    args = parser.parse_args()

    save_path = Path(args.save_dir) / "imagenet"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"ImageNet 将保存到: {save_path}")

    script_dir = Path(__file__).resolve().parent
    import ultralytics
    ul_pkg = Path(ultralytics.__file__).parent
    get_script = ul_pkg / "data" / "scripts" / "get_imagenet.sh"
    if not get_script.exists():
        print(f"未找到 {get_script}，请检查 ultralytics 安装")
        return 1

    # 在 save_path 的父目录下运行，使 ../datasets/imagenet 指向正确位置
    # get_imagenet.sh: d='../datasets/imagenet', mkdir -p $d && cd $d
    # 若 cwd=save_dir (datasets/), 则 ../datasets/imagenet = datasets/imagenet (错误)
    # 若 cwd=datasets/imagenet 的父级... 脚本会 cd 到 d，所以 d 必须存在
    # 创建 datasets 目录结构：save_dir = datasets, imagenet 在 datasets/imagenet
    parent = save_path.parent  # e.g. /data/yxfeng/datasets
    # 脚本期望 parent/datasets/imagenet 或 parent/imagenet
    # 原脚本: d='../datasets/imagenet'，从 scripts/ 的父级算，即 parent/datasets/imagenet
    # 所以需要 cwd 使得 ../datasets = parent. 即 cwd 的父级的 datasets = parent.
    # 简单做法：复制并修改脚本，使用绝对路径
    custom_script = script_dir / "get_imagenet_custom.sh"
    with open(get_script) as f:
        content = f.read()
    # 替换 d='../datasets/imagenet' 为绝对路径
    content = content.replace(
        "d='../datasets/imagenet' # unzip directory",
        f"d='{save_path}' # unzip directory",
    )
    # 若跳过 valprep 或解压不完整，跳过 valprep 避免 "cannot stat" 错误
    if args.skip_valprep:
        content = content.replace(
            "wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash # move into subdirs",
            "# valprep skipped (--skip_valprep); val images remain in flat structure",
        )
    else:
        # 添加检查：解压后若文件数不足 50000 则跳过 valprep，避免 cannot stat
        old_val_block = """if [ "$val" == "true" ]; then
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar # download 6.3G, 50000 images
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash # move into subdirs
fi"""
        new_val_block = """if [ "$val" == "true" ]; then
  if [ -f ILSVRC2012_img_val.tar.1 ] && [ $(stat -c%s ILSVRC2012_img_val.tar.1 2>/dev/null) -gt 6000000000 ]; then
    mv ILSVRC2012_img_val.tar.1 ILSVRC2012_img_val.tar
  else
    rm -f ILSVRC2012_img_val.tar ILSVRC2012_img_val.tar.1
    wget -O ILSVRC2012_img_val.tar https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
  fi
  mkdir -p val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xf ILSVRC2012_img_val.tar
  n=$(ls -1 *.JPEG 2>/dev/null | wc -l)
  if [ "$n" -ge 49900 ]; then
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  else
    echo "WARNING: Only $n val images extracted (expected 50000). Skipping valprep. Re-download if tar was corrupted."
  fi
fi"""
        content = content.replace(old_val_block, new_val_block)
    with open(custom_script, "w") as f:
        f.write(content)
    custom_script.chmod(0o755)

    opts = ["--val"] if args.val_only else ["--train", "--val"]
    try:
        subprocess.run(
            ["bash", str(custom_script)] + opts,
            check=True,
            cwd=str(script_dir),
        )
        print(f"\n✓ 下载完成！路径: {save_path}")
        print("\n设置 Ultralytics 数据集目录:")
        print(f"  yolo settings datasets_dir={parent}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 下载失败: {e}")
        return 1
    finally:
        custom_script.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    exit(main())
