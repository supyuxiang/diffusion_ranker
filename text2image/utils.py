import torch
import os
import sys
import logging
from pathlib import Path



def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32