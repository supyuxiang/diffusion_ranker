import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import os
import sys

from torch.utils.data import random_split
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import set_device, set_seed


training_samples_per_class = 128


class RankerDataset(Dataset):
    def __init__(
        self,
        step_idx:int, # choice of target step index
        layer_idx:int=16, # ratio=0.6
        prompt_type:str | None = None,
        shuffle_class:bool=False,
    ):
        super().__init__()
        self.prompt_type = prompt_type
        self.shuffle_class = shuffle_class
        self.step_idx = step_idx
        self.layer_idx = layer_idx
        self.load_prompt_encoding()
        self.load_hs_reward()
        self.load_reward()
        if shuffle_class:
            self.shuffle_class()

    
    def load_prompt_encoding(self):
        self.num_samples = int(training_samples_per_class * 1000)
        if not self.prompt_type:
            full_prompt_encoding = torch.zeros(self.num_samples, 1000) # 1000 classes, 128 samples per class
        else:
            if self.prompt_type == 'one_hot':
                path = ROOT / 'class_condition' / 'prompt_encoding' / 'one_hot.pt'
                self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
            elif self.prompt_type == 'clip':
                path = ROOT / 'class_condition' / 'prompt_encoding' / 'clip.pt'
                self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
            elif self.prompt_type == 'bert':
                path = ROOT / 'class_condition' / 'prompt_encoding' / 'bert.pt'
                self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
            elif self.prompt_type == 'llm_tokenizer':
                path = ROOT / 'class_condition' / 'prompt_encoding' / 'llm_tokenizer.pt'
                self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
            elif self.prompt_type == 'llm_hidden_state':
                path = ROOT / 'class_condition' / 'prompt_encoding' / 'llm_hidden_state.pt'
                self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
                if self.prompt_encoding.dim() != 3:
                    raise ValueError(
                        f"llm_hidden_state prompt encoding expected 3D (1000, L, H), got {tuple(self.prompt_encoding.shape)}"
                    )
                # Use last-token hidden so prompt encoding reduces to 2D: (1000, H).
                self.prompt_encoding = self.prompt_encoding[:, -1, :] # last token的hidden state
                
            else:
                raise ValueError(f"Invalid prompt type: {self.prompt_type}")
            # prompt_encoding shape is 2D: (1000, D)
            full_prompt_encoding = torch.zeros(
                                    self.num_samples, # 1000*128
                                    self.prompt_encoding.shape[1], # prompt_encoding_hidden_dim
                                    dtype=self.prompt_encoding.dtype, # float32
                                    ) # 1000*128, prompt_encoding_hidden_dim

            for class_idx in range(1000):
                start_idx = class_idx * training_samples_per_class
                end_idx = start_idx + training_samples_per_class
                full_prompt_encoding[start_idx:end_idx] = self.prompt_encoding[class_idx]
        self.prompt_hstate = full_prompt_encoding # 1000*128, prompt_encoding_hidden_dim
    
    def load_hs_reward(self):
        step_idx = self.step_idx
        layer_idx = self.layer_idx
        path = Path(self.hs_reward_path).expanduser().resolve()
        hs_reward = torch.load(path,map_location='cpu',weights_only=False)
        self.hs = hs_reward['hiddens'] # 1000*128, hidden_dim
        self.reward = hs_reward['reward'] # 1000*128, 1
        self.labels = hs_reward['labels'] # 1000*128, 1

        assert self.hs.shape[0] == self.num_samples
        assert self.reward.shape[0] == self.num_samples
        assert self.labels.shape[0] == self.num_samples

    
    def __len__(self):
        return self.num_samples
    
    def shuffle_class(self):
        indices = torch.randperm(self.num_samples)
        self.hs = self.hs[indices]
        self.reward = self.reward[indices]
        self.labels = self.labels[indices]
        self.prompt_hstate = self.prompt_hstate[indices]


    def __getitem__(self, idx):
        prompt_hs = self.prompt_hstate[idx]
        candidate_hs = self.hs[idx]
        reward = self.reward[idx]
        label = self.labels[idx]
        return prompt_hs, candidate_hs, reward, label


    

def collate_fn(batch):
    # batch: list of (prompt_hs, candidate_hs, reward, label)
    # stack into tensors
    prompt_hs, candidate_hs, reward, label = torch.stack(
        [item[0] for item in batch]
    ),
    torch.stack(
        [item[1] for item in batch]
    ),
    torch.stack(
        [item[2] for item in batch]
    ),
    torch.stack(
        [item[3] for item in batch]
    )

    return prompt_hs, candidate_hs, reward, label




    

    








