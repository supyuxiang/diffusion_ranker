#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_cifar10_resnet18(num_classes: int = 10) -> torch.nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    # CIFAR-10 standard stem
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def load_cifar10_classifier(
    ckpt_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    p = Path(ckpt_path).expanduser().resolve()
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    model = build_cifar10_resnet18()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


_to_tensor = transforms.ToTensor()
_normalize = transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)


@torch.inference_mode()
def reward_neg_entropy_cifar10(
    classifier: torch.nn.Module,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """
    Reward = negative entropy of CIFAR-10 classifier output.
    Higher reward => lower entropy => sharper class confidence.
    """
    x = torch.stack([_normalize(_to_tensor(img)) for img in images], dim=0).to(device)
    logits = classifier(x).float()
    p = F.softmax(logits, dim=-1)
    entropy = -(p * (p + 1e-10).log()).sum(dim=-1)
    return (-entropy).detach().cpu()

