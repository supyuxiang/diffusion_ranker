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

    return f"a photo of a {labels_1k[class_id]}"


def load_clip(device: torch.device):

    _DEFAULT_CLIP_DIR = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")

    clip = CLIPModel.from_pretrained(str(_DEFAULT_CLIP_DIR), local_files_only=True)
    clip.to(device)
    clip.eval()
    processor = CLIPProcessor.from_pretrained(str(_DEFAULT_CLIP_DIR))
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
    import ImageReward as RM
    model = RM.load(str(_DEFAULT_IMAGEREWARD_CKPT), device=device, med_config=str(med))
    model.eval()
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



########################     ranker reward     ##############################

def load_ranker(ranker_type:str,
                input_dim_candidate:int=1152,
                input_dim_prompt:int=3584,
                hidden_dim:int=128,
                num_heads:int=4,
                dropout:float=0.1,
                num_layers:int=4,
                half:bool=True
                ):
    assert ranker_type in ['pointwise','listwise','cross_attention']
    if ranker_type == 'pointwise':
        from ranker.model import DiffusionRanker_MLP
        ranker = DiffusionRanker_MLP(
            input_dim_candidate=input_dim_candidate,
            input_dim_prompt=input_dim_prompt,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            half=half
        )
    elif ranker_type == 'listwise':
        pass
    else:
        pass

def load_repr(layer_ratio:float,step_idx:int,pooling_mode:str='mean'):
    assert pooling_mode in ['mean','last']
    import torch
    base_dir = Path('/root/autodl-tmp/repr_candidate_deno10_cfg1.0') / str(pooling_mode)
    repr_path = base_dir / f's_{step_idx}_r{layer_idx}.pt'
    
    raw_repr = torch.load(repr_path,map_location='cpu',weights_only=False)
    # split. [80000,1152] -> 16 x [5000,1152]
    repr_ls = []
    for i in range(16):
        start_idx = i*5000
        end_idx = start_idx + 5000
        repr_i = raw_repr[start_idx:end_idx]
        repr_ls.append(repr_i)
        del repr_i
    return repr_ls

def path2idx(path:Path) -> int:
    cand_idx = int(path.split('/')[-2].split('_')[-1])
    img_idx = int(path.stem.split('_')[-2])
    return cand_idx,img_idx
    

def get_prompt_hstates(path:Path,prompt_hstates_ls:list[torch.Tensor]) -> torch.Tensor:
    label_idx = int(path.stem.split('_')[-1])
    prompt_hstates = prompt_hstates_ls[label_idx]
    return prompt_hstates
    


def ranker_reward(batch_paths:list[Path],ranker,repr_ls:list[torch.Tensor],prompt_hstates_ls:list[torch.Tensor]) -> float:
    assert len(batch_paths) % 10 == 0
    reward_batch_size = len(batch_paths) // 10
    total_reward = 0

    cand_idx_ls , img_idx_ls = [],[]
    for path in batch_paths:
        cand_idx,img_idx = path2idx(path)
        cand_idx_ls.append(cand_idx)
        img_idx_ls.append(img_idx)
    
    batch_prompt_hstates = [
        get_prompt_hstates(path,prompt_hstates_ls) for path in batch_paths
    ]
    
    batch_repr_ls = [
        repr_ls[cand_idx][img_idx] for cand_idx,img_idx in zip(cand_idx_ls,img_idx_ls)
    ] 
    batch_repr = torch.stack(batch_repr_ls,dim=0) # b,1152

    for i in range(0,len(batch_paths),reward_batch_size):
        mini_batch = batch_repr[i:i+reward_batch_size]
        # [b,1152] x 10 -> [b,10,1152]
        candidate_hstates = torch.stack(mini_batch,dim=1) # b,10,1152
        prompt_hstates = batch_prompt_hstates[i:i+reward_batch_size]
        prompt_hstates = torch.stack(prompt_hstates,dim=1) # b,10,hidden_dim

        reward = ranker(candidate_hstates,prompt_hstates)
        total_reward += reward.mean().item()

    return total_reward / len(batch_paths)
        
def load_prompt_hstates_ls() -> list[torch.Tensor]:
    path = Path('/root/autodl-tmp/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt')
    return torch.load(path,map_location='cpu',weights_only=False)

############ end ranker reward     ##############################
    


    
    





    











