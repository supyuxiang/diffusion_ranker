"""供 random search 脚本共用的 prompt 工具与 DiT 在线 rollout。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from diffusers import DiTPipeline

from utils import (
    DitRolloutReprCapture,
    _prepare_pil_for_save,
    rollout_dit_with_steps,
)
from utils import _imagenet_simple_labels_list

CANDIDATE_SEED_HIGH = 2**31 - 1


def load_imagenet_simple_labels() -> list[str]:
    """与 ``utils.imagenet_label2text_for_clipscore`` 同源标签表（本地 json）。"""
    return list(_imagenet_simple_labels_list())


def prompt_for_class(labels: list[str], class_id: int, template: str) -> str:
    name = labels[class_id] if 0 <= class_id < len(labels) else f"class {class_id}"
    return template.format(name=name, id=class_id)


def init_random_samples(
    pipe: DiTPipeline,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    device: str | torch.device,
    seed: int,
    *,
    guidance_scale: float = 1.0,
) -> int:
    """在 ``output_dir`` 写入 ``image_{idx:05d}_{label:05d}.png``，返回实际写入张数。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device) if isinstance(device, str) else device
    return rollout_dit_with_steps(
        pipe,
        output_dir,
        num_steps,
        num_samples,
        batch_size,
        dev,
        seed=seed,
        guidance_scale=guidance_scale,
        repr_capture=None,
    )


@torch.inference_mode()
def rollout_dit_batch_with_repr(
    pipe: DiTPipeline,
    repr_capture: DitRolloutReprCapture,
    num_steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    *,
    guidance_scale: float = 1.0,
    image_size: int = 256,
) -> tuple[list[Image.Image], list[int], dict]:
    """
    单次条件采样 batch；返回 PIL 列表、ImageNet 类 id 列表、``repr`` 字典（与 ``DitRolloutReprCapture`` 一致）。

    调用前须已 ``repr_capture.install()``；本函数会清空 ``repr_capture.records`` 中本 batch 写入的一条以省内存。
    """
    labels = torch.randint(0, 1000, (batch_size,), device=device).tolist()
    repr_capture.set_capturing(True)
    repr_capture.begin_batch(labels)
    gen = torch.Generator(device=device).manual_seed(int(seed) % CANDIDATE_SEED_HIGH)
    out = pipe(
        class_labels=labels,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=gen,
    )
    repr_capture.end_batch(0, 0, generator_seed=int(seed) % CANDIDATE_SEED_HIGH)
    if not repr_capture.records:
        raise RuntimeError("repr_capture.records empty after end_batch")
    repr_dict = repr_capture.records[-1]["repr"]
    repr_capture.records.clear()

    imgs: list[Image.Image] = []
    for img in out.images:
        imgs.append(_prepare_pil_for_save(img.convert("RGB"), image_size))
    return imgs, labels, repr_dict
