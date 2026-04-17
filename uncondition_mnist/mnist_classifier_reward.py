#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_RESIZE_HW = 224


def build_mnist_resnet18(
    num_classes: int = 10,
    *,
    pretrained: bool = True,
) -> torch.nn.Module:
    """ImageNet-pretrained ResNet18 finetuned for MNIST (10 classes)."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_mnist_classifier(
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
    model = build_mnist_resnet18(pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


_to_tensor = transforms.ToTensor()
_resize = transforms.Resize((_RESIZE_HW, _RESIZE_HW), interpolation=transforms.InterpolationMode.BILINEAR)
_normalize = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)


@torch.inference_mode()
def reward_neg_entropy_mnist(
    classifier: torch.nn.Module,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """
    Reward = negative entropy of MNIST classifier output.
    Higher reward => lower entropy => sharper class confidence.
    """
    x = torch.stack([_normalize(_to_tensor(_resize(img.convert("RGB")))) for img in images], dim=0).to(device)
    logits = classifier(x).float()
    p = F.softmax(logits, dim=-1)
    entropy = -(p * (p + 1e-10).log()).sum(dim=-1)
    return (-entropy).detach().cpu()
