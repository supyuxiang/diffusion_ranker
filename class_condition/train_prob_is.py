import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
import wandb
import argparse
import swanlab
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
from pathlib import Path
import json
import time 
import yaml


import os
import sys
from typing import Optional, Dict, Any, List, Tuple, Union








class ProbIS(nn.Module):
    def __init__(
        self,
        input_dim:int,
        hidden_dim:int,
        num_layers:int,
        dropout_prob:float,
        half:bool=True,
    ):
        super(ProbIS, self).__init__()
        self.half = half
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_proj = nn.Linear(input_dim,hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim,4*hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.LayerNorm(4*hidden_dim),
                nn.Linear(4*hidden_dim,hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
            ]
            for _ in range(num_layers)
        )
        self.reward_head = nn.Linear(hidden_dim,1)
        self._init_weights()
        if half:
            self.to(torch.float16)
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        if self.reward_head.bias is not None:
            nn.init.zeros_(self.reward_head.bias)
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.reward_head(x)
    
    def count_param(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HiddenRewardDataset(Dataset):
    def __init__(
        self,
        hiddens:torch.Tensor,
        rewards:torch.Tensor,
    ):
        self.hiddens = hiddens.float()
        self.rewards = rewards.view(-1).float()

        # check
        # check end

    def __len__(self) -> int:
        return int(self.hiddens.shape[0])
    
    def __getitem__(self, idx:int):
        return self.hiddens[idx], self.rewards[idx]
    



def collate(batch:List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    hiddens = torch.stack([item[0] for item in batch])
    rewards = torch.stack([item[1] for item in batch])
    return hiddens, rewards



def set_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    return p.parse_args()


class Trainer:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.build_dataloader()
        self.build_model()
        self.build_optimzer()
        self.build_scheduler()
    
    def build_dataloader(self,):
        pass
    

    pass


@torch.inference_mode()
def reward_batch(
    inception_model,
    batch_images:torch.Tensor,
):
    pass




def main():
    pass




if __name__ == "__main__":
    main()
    













