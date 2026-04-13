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

    

def fid_reward(batch_stats:dict,n_batch:int,ref_stats:dict,n_ref:int,imagenet_val_stats:dict) -> float:
    merged_stats = merge_fid_statistics(batch_stats,n_batch,ref_stats,n_ref)
    return -fid_statistics_to_distance(merged_stats, imagenet_val_stats) # 越大越好


########################     end fid reward     ##############################


######################## clipscore reward ##############################










