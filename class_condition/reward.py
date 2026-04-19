import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from typing import Optional

_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from utils import *

import numpy as np
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop
import torch.nn.functional as F


############# fid reward ####################

def extract_features_from_images(
    images: list[Image.Image],
    feat_extractor,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    from torch_fidelity.datasets import TransformPILtoRGBTensor
    import torchvision.transforms as T

    transform = T.Compose(
        [T.Resize(256), T.CenterCrop(256), TransformPILtoRGBTensor()]
    )
    feat_extractor.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            tensors = torch.stack(
                [transform(img) for img in images[i : i + batch_size]], dim=0
            )
            tensors = tensors.to(device, non_blocking=device.type == "cuda")
            out = feat_extractor(tensors)
            if isinstance(out, tuple): # tuple
                out = out[0]
            chunks.append(out.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)  # (N, 2048)

def fid_features_to_statistics(features: np.ndarray) -> dict:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {"mu": mu, "sigma": sigma}


def merge_fid_statistics(stats_a: dict, n_a: int, stats_b: dict, n_b: int) -> dict:
    if n_a == 0:
        return stats_b
    if n_b == 0:
        return stats_a

    mu_a, sigma_a = stats_a["mu"], stats_a["sigma"]
    mu_b, sigma_b = stats_b["mu"], stats_b["sigma"]
    n = n_a + n_b

    mu_merged = (n_a * mu_a + n_b * mu_b) / n
    sum_xx_a = ((n_a - 1) * sigma_a if n_a > 1 else 0) + n_a * np.outer(mu_a, mu_a)
    sum_xx_b = ((n_b - 1) * sigma_b if n_b > 1 else 0) + n_b * np.outer(mu_b, mu_b)
    sum_xx = sum_xx_a + sum_xx_b
    sigma_merged = (sum_xx - n * np.outer(mu_merged, mu_merged)) / (n - 1) if n > 1 else np.zeros_like(sigma_a)

    return {"mu": mu_merged, "sigma": sigma_merged}


def fid_statistics_to_distance(stat_gen: dict, stat_ref: dict) -> float:
    mu1, sigma1 = stat_gen["mu"], stat_gen["sigma"]
    mu2, sigma2 = stat_ref["mu"], stat_ref["sigma"]
    diff = mu1 - mu2
    evals = np.linalg.eigvals(sigma1 @ sigma2)
    evals = np.maximum(evals.real, 0.0)
    tr_covmean = np.sum(np.sqrt(evals.astype("float64") + 1e-14))
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fid)

    

def fid_reward(batch_stats:dict,n_batch:int,selected_stats:dict,n_selected:int,ref_stats:dict) -> float:
    merged_stats = merge_fid_statistics(selected_stats,n_selected,batch_stats,n_batch)
    return -fid_statistics_to_distance(merged_stats, ref_stats) # 越大越好


########################     end fid reward     ##############################


######################## clipscore reward ##############################
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


with open('/root/diffusion_ranker/imagenet-simple-labels.json', 'r') as f:
    labels_1k = json.load(f)


def _path_to_clip_prompt(path: Path, labels_1k: list[str]) -> str:
    """``image_14007_00512.png`` → 取 stem 最后一段为 class_id。"""
    stem = Path(path).stem
    parts = stem.split("_")
    # check
    if len(parts) < 3:
        raise ValueError(f"expected image_*_*_{{class}}.png, got {path.name}")
    class_id = int(parts[-1])
    if not (0 <= class_id < len(labels_1k)):
        raise IndexError(f"class_id={class_id} out of range for labels ({len(labels_1k)})")

    return f"A photo of a {labels_1k[class_id]}."


def load_clip(device: torch.device):

    _DEFAULT_CLIP_DIR = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")

    clip = CLIPModel.from_pretrained(str(_DEFAULT_CLIP_DIR), local_files_only=True)
    clip.to(device)
    clip.eval()
    processor = CLIPProcessor.from_pretrained(str(_DEFAULT_CLIP_DIR), local_files_only=True)
    return clip, processor


def clipscore_reward(
    batch_paths: list[Path],
    clip,
    processor,
    labels_1k: list[str],
) -> float:
    batch_images = [
        Image.open(path).convert("RGB").copy() for path in batch_paths
    ]
    batch_prompts = [_path_to_clip_prompt(path, labels_1k) for path in batch_paths]

    inputs = processor(
        text=batch_prompts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k:v.to(clip.device) for k,v in inputs.items()}
    # inference
    with torch.no_grad():
        out = clip(**inputs)
        image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_features = out.text_embeds / out.text_embeds.norm(dim=-1,keepdim=True)
        sim = (image_features * text_features).sum(dim=-1)
    return float(sim.mean().item())

########################     end clipscore reward     ##############################

########################     imagereward reward     ##############################
from pathlib import Path
from PIL import Image
import numpy as np

_DEFAULT_IMAGEREWARD_DIR = Path("/root/autodl-tmp/ImageReward")
_DEFAULT_IMAGEREWARD_CKPT = _DEFAULT_IMAGEREWARD_DIR / "ImageReward.pt"
med = _DEFAULT_IMAGEREWARD_DIR / "med_config.json"
ensure_image_reward_on_path()
import ImageReward as RM

def load_imagereward(device: torch.device):
    import os
    import ImageReward as RM
    from transformers import BertTokenizer

    # 避免在无网/慢网环境下卡在 tokenizer 在线探测
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    print("[imagereward] warmup bert-base-uncased tokenizer (local_files_only=True)")
    try:
        _ = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            "BertTokenizer 本地缓存不可用。请先在可联网环境下载 bert-base-uncased，"
            "或把缓存放到当前机器后重试。"
        ) from e

    # ImageReward 内部仍会再次调用 from_pretrained，这里强制补 local_files_only=True
    _orig_from_pretrained = BertTokenizer.from_pretrained

    def _offline_from_pretrained(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        return _orig_from_pretrained(*args, **kwargs)

    BertTokenizer.from_pretrained = _offline_from_pretrained
    print("[imagereward] loading ImageReward checkpoint ...")
    model = RM.load(str(_DEFAULT_IMAGEREWARD_CKPT), device=device, med_config=str(med))
    model.eval()
    print("[imagereward] checkpoint loaded")
    return model

def imagereward_reward(batch_paths:list[Path],model) -> float:
    import numpy as np
    from PIL import Image
    batch_images = [
        Image.open(path).convert('RGB').copy() for path in batch_paths
    ]
    batch_prompts = [
        _path_to_clip_prompt(path, labels_1k) for path in batch_paths
    ]

    reward_ls = []
    for prompt,image in zip(batch_prompts,batch_images):
        reward = float(model.score(prompt,image))
        reward_ls.append(reward)
    return np.mean(reward_ls)

########################     end imagereward reward     ##############################



########################     ranker reward（tensor 路径）##############################


@torch.inference_mode()
def ranker_mean_logit_score(
    ranker: nn.Module,
    repr_dict: dict,
    repr_key: str,
    labels: list[int],
    prompt_table: torch.Tensor,
    device: torch.device,
) -> float:
    """单键 ``repr_dict[repr_key]['mean']`` (B,D) + ImageNet 类 prompt；K=1 时 ranker 输出 logits 的 batch 均值。"""
    if repr_key not in repr_dict:
        raise KeyError(
            f"repr 中无键 {repr_key!r}；示例键: {list(repr_dict.keys())[:8]} …"
        )
    cand = repr_dict[repr_key]["mean"].float()
    if cand.dim() != 2:
        raise ValueError(f"{repr_key}['mean'] 期望 (B, D)，得到 {tuple(cand.shape)}")
    b = cand.shape[0]
    if len(labels) != b:
        raise ValueError(f"labels 长度 {len(labels)} != batch {b}")
    lab = torch.tensor(labels, device=device, dtype=torch.long)
    if (lab < 0).any() or (lab >= prompt_table.shape[0]).any():
        raise IndexError(f"class id 越界，prompt 表行数={prompt_table.shape[0]}")

    p = prompt_table[lab.cpu()].to(device=device, dtype=torch.float32)
    w = next(ranker.parameters())
    cand_b = cand.to(device=device, dtype=w.dtype).unsqueeze(1)  # (B,1,D_c)
    p_b = p.to(device=device, dtype=w.dtype)
    logits = ranker(cand_b, p_b)
    return float(logits.float().mean().item())


############ end ranker reward     ##############################
    


    
    
##########     is reward     ##############################

@torch.inference_mode()
def _batch_is_reward(paths: list[Path], feat_extractor, device: torch.device) -> float:
    from torch_fidelity.datasets import TransformPILtoRGBTensor

    pil_images = [Image.open(p).convert("RGB").copy() for p in paths]
    transform = Compose([Resize(256), CenterCrop(256), TransformPILtoRGBTensor()])
    tensors = torch.stack([transform(img) for img in pil_images], dim=0).to(
        device, non_blocking=(device.type == "cuda")
    )
    logits = feat_extractor(tensors)
    # breakpoint()
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = F.softmax(logits.float(), dim=-1)
    return float(probs.max(dim=-1).values.mean().item()) # 考虑对每个样本计算概率，然后取平均

########################     end is reward     ##############################



    











