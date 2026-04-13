"""供 ``dit_random_search_clipscore`` / ``dit_random_search_imagereward`` 共用的 ImageNet 标签与 prompt。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import _imagenet_simple_labels_list


def load_imagenet_simple_labels() -> list[str]:
    """与 ``utils.imagenet_label2text_for_clipscore`` 同源标签表（本地 json）。"""
    return list(_imagenet_simple_labels_list())


def prompt_for_class(labels: list[str], class_id: int, template: str) -> str:
    name = labels[class_id] if 0 <= class_id < len(labels) else f"class {class_id}"
    return template.format(name=name, id=class_id)
