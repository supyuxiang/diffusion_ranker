#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel
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


def _images_to_gray_tensor(images: list[Image.Image]) -> torch.Tensor:
    arr = np.stack([np.array(im.convert("L"), dtype=np.float32) / 255.0 for im in images], axis=0)
    t = torch.from_numpy(arr)  # [B,H,W], float32 in [0,1]
    return t


@torch.inference_mode()
def reward_brightness_mnist(images: list[Image.Image]) -> torch.Tensor:
    """
    Reward = mean luminance in grayscale space.
    For MNIST-like images, brighter digits receive higher score.
    """
    g = _images_to_gray_tensor(images)
    return g.mean(dim=(1, 2)).detach().cpu()


@torch.inference_mode()
def reward_sharpness_mnist(images: list[Image.Image]) -> torch.Tensor:
    """
    Reward = Laplacian variance of grayscale image.
    Higher value indicates sharper edges / strokes.
    """
    g = _images_to_gray_tensor(images).unsqueeze(1)  # [B,1,H,W]
    k = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=g.dtype,
        device=g.device,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(g, k, padding=1)
    var = lap.flatten(1).var(dim=1, unbiased=False)
    return var.detach().cpu()


@torch.inference_mode()
def reward_image_entropy_mnist(images: list[Image.Image], bins: int = 256) -> torch.Tensor:
    """
    Reward = Shannon entropy of grayscale histogram.
    """
    if bins <= 1:
        raise ValueError(f"bins must be > 1, got {bins}")
    g = _images_to_gray_tensor(images)  # [B,H,W]
    eps = 1e-12
    vals: list[torch.Tensor] = []
    for i in range(g.shape[0]):
        hist = torch.histc(g[i], bins=bins, min=0.0, max=1.0)
        p = hist / hist.sum().clamp_min(eps)
        ent = -(p * (p + eps).log()).sum()
        vals.append(ent)
    return torch.stack(vals, dim=0).detach().cpu()


@dataclass
class MnistAlignReward:
    clip_model: CLIPModel
    clip_processor: AutoImageProcessor
    dino_model: AutoModel
    dino_processor: AutoImageProcessor
    device: torch.device


@torch.inference_mode()
def load_mnist_align_reward(
    clip_model_path: str | Path,
    dino_model_path: str | Path,
    device: torch.device,
) -> MnistAlignReward:
    clip_path = str(Path(clip_model_path).expanduser().resolve())
    dino_path = str(Path(dino_model_path).expanduser().resolve())

    clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device).eval()
    clip_processor = AutoImageProcessor.from_pretrained(clip_path, local_files_only=True)

    dino_model = AutoModel.from_pretrained(dino_path, local_files_only=True).to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)

    return MnistAlignReward(
        clip_model=clip_model,
        clip_processor=clip_processor,
        dino_model=dino_model,
        dino_processor=dino_processor,
        device=device,
    )


@torch.inference_mode()
def _clip_features(
    model: CLIPModel,
    processor: AutoImageProcessor,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    proc = processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)
    # Be robust to transformers versions/checkpoints where get_image_features
    # may not directly return a tensor.
    clip_out = model.vision_model(pixel_values=pixel_values)
    pooled = clip_out.pooler_output
    if hasattr(model, "visual_projection") and model.visual_projection is not None:
        feat = model.visual_projection(pooled).float()
    else:
        feat = pooled.float()
    feat = F.normalize(feat, dim=-1)
    return feat


@torch.inference_mode()
def _dino_features(
    model: AutoModel,
    processor: AutoImageProcessor,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    proc = processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)
    out = model(pixel_values=pixel_values)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output.float()
    else:
        feat = out.last_hidden_state[:, 0, :].float()
    feat = F.normalize(feat, dim=-1)
    return feat


@torch.inference_mode()
def reward_clip_dino_align_mnist(
    align_reward: MnistAlignReward,
    mid_images: list[Image.Image],
    final_images: list[Image.Image],
    *,
    clip_weight: float = 0.5,
) -> torch.Tensor:
    """
    Reward = weighted cosine similarity between middle-step and final-image features.
      reward = clip_weight * cos(CLIP(mid), CLIP(final))
             + (1-clip_weight) * cos(DINO(mid), DINO(final))
    """
    if len(mid_images) != len(final_images):
        raise ValueError(f"mid/final image count mismatch: {len(mid_images)} vs {len(final_images)}")
    if not (0.0 <= clip_weight <= 1.0):
        raise ValueError(f"clip_weight must be in [0,1], got {clip_weight}")

    device = align_reward.device
    clip_mid = _clip_features(align_reward.clip_model, align_reward.clip_processor, mid_images, device)
    clip_fin = _clip_features(align_reward.clip_model, align_reward.clip_processor, final_images, device)
    dino_mid = _dino_features(align_reward.dino_model, align_reward.dino_processor, mid_images, device)
    dino_fin = _dino_features(align_reward.dino_model, align_reward.dino_processor, final_images, device)

    clip_sim = (clip_mid * clip_fin).sum(dim=-1)
    dino_sim = (dino_mid * dino_fin).sum(dim=-1)
    reward = clip_weight * clip_sim + (1.0 - clip_weight) * dino_sim
    return reward.detach().cpu()
