# 用 ImageReward 给 rollout 图打分；文件名需为 ``image_*_{class_id:05d}.png``（与 rollout 一致）。

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import _imagenet_simple_labels_list, ensure_image_reward_on_path

ensure_image_reward_on_path()
from ImageReward import RM  # noqa: E402

_TRAINING_IMAGE_DIR = Path("/root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10")
# 与下载脚本一致；也可指向 ~/.cache/ImageReward
_IMAGE_REWARD_CKPT = Path.home() / ".cache" / "ImageReward" / "ImageReward.pt"
_IMAGE_REWARD_MED_CONFIG = Path.home() / ".cache" / "ImageReward" / "med_config.json"

PROMPT_TEMPLATE = "a photo of a {class_name}"

_IMAGENET_NAMES = _imagenet_simple_labels_list()


def class_id_from_rollout_png(path: Path) -> int:
    """``image_01234_00042.png`` → ``42``（ImageNet 索引 0..999）。"""
    return int(path.stem.split("_")[-1])


def class_id_to_name(class_id: int) -> str:
    if not 0 <= class_id < len(_IMAGENET_NAMES):
        raise ValueError(f"class_id out of range: {class_id}")
    return _IMAGENET_NAMES[class_id]


teacher = RM.load(
    str(_IMAGE_REWARD_CKPT),
    device="cuda",
    med_config=str(_IMAGE_REWARD_MED_CONFIG),
)
teacher.to("cuda")

img_path_ls = sorted(_TRAINING_IMAGE_DIR.glob("*.png"))
reward_ls: list[float] = []

for img_path in img_path_ls:
    cid = class_id_from_rollout_png(img_path)
    class_name = class_id_to_name(cid)
    prompt = PROMPT_TEMPLATE.replace("{class_name}", class_name)
    img = Image.open(img_path).convert("RGB")
    reward = float(teacher.score(prompt, img))
    reward_ls.append(reward)

print(reward_ls)
print(len(reward_ls))

save_path = Path(
    '/root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/reward_ls.json'
)

with open(save_path,"w") as f:
    json.dump(reward_ls,f)
