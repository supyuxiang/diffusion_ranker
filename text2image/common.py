from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PromptItem:
    prompt_id: str
    prompt_text: str
    benchmark: str = "custom"
    tags: list[str] | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str | Path) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_prompts_jsonl(path: str | Path) -> list[PromptItem]:
    p = Path(path).expanduser().resolve()
    items: list[PromptItem] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt_id" not in obj or "prompt_text" not in obj:
                raise ValueError(f"{p}:{i} missing prompt_id/prompt_text")
            items.append(
                PromptItem(
                    prompt_id=str(obj["prompt_id"]),
                    prompt_text=str(obj["prompt_text"]),
                    benchmark=str(obj.get("benchmark", "custom")),
                    tags=list(obj.get("tags", [])),
                )
            )
    if not items:
        raise ValueError(f"no prompts found in {p}")
    return items


def list_prompt_dirs(pool_root: str | Path, benchmark: str | None = None) -> list[Path]:
    root = Path(pool_root).expanduser().resolve()
    if benchmark:
        bdir = root / benchmark
        if not bdir.is_dir():
            return []
        return sorted([d for d in bdir.iterdir() if d.is_dir()])
    out: list[Path] = []
    for b in sorted([d for d in root.iterdir() if d.is_dir()]):
        out.extend(sorted([d for d in b.iterdir() if d.is_dir()]))
    return out


def safe_image_name(idx: int) -> str:
    return f"image_{idx:05d}.png"


def torch_dtype_from_str(dtype: str) -> torch.dtype:
    d = dtype.lower().strip()
    if d in ("fp16", "float16"):
        return torch.float16
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype {dtype}")
