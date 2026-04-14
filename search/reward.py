import torch
import torch.nn as nn

import sys
from pathlib import Path
from typing import Optional

_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from utils import *

import numpy as np

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

def load_clip(model_dir:Path,device):
    _default_clip_dir = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained(model_dir, local_files_only=True)
    clip.to(device)
    clip.eval()
    return clip

def clipscore_reward(batch_paths:list[Path],clip:CLIPModel,processor:CLIPProcessor,labels_1k:list[str]) -> float:
    batch_images = [
        Image.open(path).convert('RGB').copy()
        for path in batch_paths
    ]
    batch_prompts = [
        path2prompt(path,labels_1k)
        for path in batch_paths
    ]

    def path2prompt(path:Path,labels_1k:list[str]) -> str:
        # 类似   image_14007_00512.png, 其中512是classid, 00000 到 00999
        path = Path(path)
        parts = path.stem.split('_')[-1] # 00512, str
        class_id = int(parts) # 512, int

        return f"a photo of a {labels_1k[class_id]}"
        











