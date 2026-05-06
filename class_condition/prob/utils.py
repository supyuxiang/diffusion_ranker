from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from PIL import Image

from utils import ensure_image_reward_on_path

_ROOT = Path(__file__).resolve().parents[2]
_LABELS_JSON = _ROOT / "imagenet-simple-labels.json"
_DEFAULT_IMAGEREWARD_DIR = Path("/root/autodl-tmp/ImageReward")
_DEFAULT_IMAGEREWARD_CKPT = _DEFAULT_IMAGEREWARD_DIR / "ImageReward.pt"
_DEFAULT_IMAGEREWARD_MED = _DEFAULT_IMAGEREWARD_DIR / "med_config.json"


def load_imagenet_class_names() -> list[str]:
    labels = json.loads(_LABELS_JSON.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or len(labels) != 1000:
        raise ValueError(f"Invalid ImageNet labels: {_LABELS_JSON}")
    return [str(x) for x in labels]


def load_imagereward(device: torch.device):
    ensure_image_reward_on_path()
    import ImageReward as RM
    from transformers import BertTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _ = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    _orig = BertTokenizer.from_pretrained

    def _offline_from_pretrained(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        return _orig(*args, **kwargs)

    BertTokenizer.from_pretrained = _offline_from_pretrained
    model = RM.load(
        str(_DEFAULT_IMAGEREWARD_CKPT),
        device=device,
        med_config=str(_DEFAULT_IMAGEREWARD_MED),
    )
    model.eval()
    return model


def broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        dt = torch.float64 if isinstance(t, float) else torch.int64
        timesteps = torch.tensor([t], dtype=dt, device=latent_model_input.device)
    elif len(t.shape) == 0:
        timesteps = t[None].to(latent_model_input.device)
    else:
        timesteps = t
    return timesteps.expand(latent_model_input.shape[0])


def decode_dit_latents_to_pil(pipe, x: torch.Tensor, guidance_scale: float) -> list[Image.Image]:
    if guidance_scale > 1:
        latents, _ = x.chunk(2, dim=0)
    else:
        latents = x
    latents = 1 / pipe.vae.config.scaling_factor * latents
    samples = pipe.vae.decode(latents).sample
    samples = (samples / 2 + 0.5).clamp(0, 1)
    images: list[Image.Image] = []
    for bi in range(samples.shape[0]):
        arr = samples[bi].detach().cpu().permute(1, 2, 0).float().numpy()
        images.append(Image.fromarray((arr * 255.0).round().clip(0, 255).astype("uint8")))
    return images


def pearsonr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.view(-1).float()
    y = y.view(-1).float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx.pow(2).sum() + eps) * (vy.pow(2).sum() + eps))
    return float((vx * vy).sum().item() / denom.item())

