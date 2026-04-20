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


class RankerTrainingDataset(Dataset):
    def __init__(
        self,
        hs_reward_path:Path,
        step_idx:int, # choice of target step index
        layer_idx:int=16, # ratio=0.6
        prompt_type:str | None = None,
        shuffle_class:bool=False,
    ):
        super().__init__()
        self.hs_reward_path = hs_reward_path
        self.prompt_type = prompt_type
        self.shuffle_class = shuffle_class
        self.step_idx = step_idx
        self.layer_idx = layer_idx
        self.load_prompt_encoding()
        self.load_hs_reward()
        self.load_reward()
    
    def load_prompt_encoding(self):
        num_samples = int(training_samples_per_class * 1000)
        if not self.prompt_type:
            full_prompt_encoding = torch.zeros(num_samples, 1000) # 1000 classes, 128 samples per class
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
                num_samples,
                *self.prompt_encoding.shape[1:],
                dtype=self.prompt_encoding.dtype,
            )
            for class_idx in range(1000):
                start_idx = class_idx * training_samples_per_class
                end_idx = start_idx + training_samples_per_class
                full_prompt_encoding[start_idx:end_idx] = self.prompt_encoding[class_idx]
        self.prompt_hstate = full_prompt_encoding
    
    def load_hs_reward(self):
        # 128 * 1000, hidden_dim



        pass
    
    def load_reward(self):
        # 128 * 1000, 1
        pass

    
    # prompt_hstate: 1000*training_samples_per_class, dim
    # candidate_hstates: 1000*training_samples_per_class, hidden_dim
    # reward: 1000*training_samples_per_class, 1

    # next step: 拼接起来 ()
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        start_idx = idx * training_samples_per_class
        end_idx = start_idx + training_samples_per_class
        prompt_hstate = torch.stack(self.prompt_hstate[start_idx:end_idx],dim=0) # (128, prompt_dim)
        candidate_hstates = torch.stack(self.candidate_hstates[start_idx:end_idx],dim=0) # (128, hidden_dim)
        return prompt_hstate, candidate_hstates


# class RankerTrainingDataset_shuffle_class(Dataset):
#     def __init__(
#         self,
#         candidate_dir:Path,
#         prompt_type:str | None = None,
#     ):
#         super().__init__()
#         self.candidate_dir = candidate_dir
#         self.prompt_type = prompt_type
#         self.load_prompt_encoding()
#         self.load_candidate_hiddens()
    
#     def load_prompt_encoding(self):
#         if not self.prompt_type:
#             self.prompt_hstate = None
#         else:
#             if self.prompt_type == 'one_hot':
#                 path = ROOT / 'class_condition' / 'prompt_encoding' / 'one_hot.pt'
#                 self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
#             elif self.prompt_type == 'clip':
#                 path = ROOT / 'class_condition' / 'prompt_encoding' / 'clip.pt'
#                 self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
#             elif self.prompt_type == 'bert':
#                 path = ROOT / 'class_condition' / 'prompt_encoding' / 'bert.pt'
#                 self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
#             elif self.prompt_type == 'llm_tokenizer':
#                 path = ROOT / 'class_condition' / 'prompt_encoding' / 'llm_tokenizer.pt'
#                 self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
#             elif self.prompt_type == 'llm_hidden_state':
#                 path = ROOT / 'class_condition' / 'prompt_encoding' / 'llm_hidden_state.pt'
#                 self.prompt_encoding = torch.load(path,map_location='cpu',weights_only=False)#
#             else:
#                 raise ValueError(f"Invalid prompt type: {self.prompt_type}")
#             # shape: 1000, dim
#             num_samples = int(training_samples_per_class * 1000)
#             full_prompt_encoding = torch.zeros(num_samples, self.prompt_encoding.shape[1])
#             for class_idx in range(1000):
#                 start_idx = class_idx * training_samples_per_class
#                 end_idx = start_idx + training_samples_per_class
#                 full_prompt_encoding[start_idx:end_idx] = self.prompt_encoding[class_idx]
#         self.prompt_hstate = full_prompt_encoding
    
#     def load_candidate_hiddens(self):
#         pass
    
#     def __len__(self):
#         return len(self.candidate_hiddens)
    
#     def __getitem__(self, idx):
#         pass
    
#     def shuffle

            

    
    def load_candidate_hiddens(self):
        if not self.candidate_dir:
            raise ValueError(f"Candidate directory is not set")
        pass
    

    def __len__(self):
        return len(self.candidate_hiddens)
    
    def __getitem__(self, idx):
        return self.candidate_hiddens[idx], self.prompt_encoding[idx]
    

def collate_fn(batch):
    return torch.stack([item[0] for item in batch]), torch.stack([item[1] for item in batch])




    

    








