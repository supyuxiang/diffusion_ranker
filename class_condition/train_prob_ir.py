import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from tqdm import tqdm
import swanlab
from pprint import pprint


import sys
import os


config = {
    "train_batch_size": 128,
    "eval_batch_size": 128,
    "num_workers": 8,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "half": True,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout_prob": 0.1,
    "input_dim": 1024,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "half": True,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout_prob": 0.1,
    "input_dim": 1024,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "gradient_accumulation_steps": 1,
}


class Prob_ImageReward(nn.Module):
    def __init__(
        self,
        input_dim:int,
        hidden_dim:int=256,
        num_layers:int=2,
        dropout_prob:float=0.1,
        half:bool=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.half = half

        self.proj = nn.Linear(input_dim,hidden_dim)
    
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim,4*hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.LayerNorm(4*hidden_dim),
                nn.Linear(4*hidden_dim,hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob)
            ]
            for _ in range(num_layers)
        )

        self.reward_head = nn.Linear(hidden_dim,1)
        
        self._init_weights()
        if self.half:
            self.to(torch.float16)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.reward_head.weight)
        if self.reward_head.bias is not None:
            nn.init.zeros_(self.reward_head.bias)
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m,nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m,nn.GELU):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m,nn.Dropout):
                    nn.init.zeros_(m.weight)
                else:
                    raise ValueError(f"Unsupported module: {type(m)}")
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for layer in self.layers:
            x = x+layer(x)
        return self.reward_head(x)



def collect_train_data(
    save_dir:Path=Path("data/train"),
    num_samples:int=5000,
    class_idx:int=256,
    mc_samples:int=8,
    num_workers:int=8,
    batch_size:int=64,
):
    save_dir.mkdir(parents=True,exist_ok=True)
    pass

    


class Trainer:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.build_dataloaders()
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
    
    def build_dataloaders(self):
        

















