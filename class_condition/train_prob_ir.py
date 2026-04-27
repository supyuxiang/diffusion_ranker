import torch
import torch.nn as nn
from pathlib import Path


import sys
import os




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
        

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass
















