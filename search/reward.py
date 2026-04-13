import torch
import torch.nn as nn

import sys
from pathlib import Path
from typing import Optional

_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from utils import *



def fid_reward(batch_stats:dict,n_batch:int,ref_stats:dict,n_ref:int,imagenet_val_stats:dict) -> float:
    merged_stats = merge_fid_statistics(batch_stats,n_batch,ref_stats,n_ref)
    return -fid_statistics_to_distance(merged_stats, imagenet_val_stats) # 越大越好












