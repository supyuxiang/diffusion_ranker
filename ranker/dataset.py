import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
import json



def load_prompt_repr(prompt_repr_pt:Path=Path('/root/diffusion_ranker/ranker/prompt_repr.pt')):
    prompt_repr = torch.load(prompt_repr_pt,map_location='cpu',weights_only=False)
    return prompt_repr # 1000,repr_dim


class DiffusionRankerDataset(Dataset):
    def __init__(self,repr_pt:Path,reward_path:Path,prompt_repr:Tensor):
        self.repr_pt = repr_pt
        self.reward_path = reward_path
        self.load_repr_and_reward()
    
    def load_repr_and_reward(self):
        with open(self.reward_path,'r',encoding='utf-8') as f:
            self.reward_ls = json.load(f)
        self.hiddens = torch.load(self.repr_pt,map_location='cpu')['tensor'] 
        self.hidden_dim = self.hiddens.shape[1]
        assert len(self.reward_ls) == len(self.hiddens)

    
    def __len__(self):
        return len(self.reward_ls)
    
    def __getitem__(self,idx):
        return self.hiddens[idx],self.reward_ls[idx]

def collect(batch:list[tuple[torch.Tensor,float]]):
    batch_hiddens,batch_rewards = [],[]
    for item in batch:
        hidden,reward = item
        batch_hiddens.append(hidden)
        batch_rewards.append(reward)
    batch_hiddens = torch.cat(batch_hiddens,dim=0)
    batch_rewards = torch.tensor(batch_rewards)
    return batch_hiddens,batch_rewards

def dataloader(repr_pt:Path,reward_path:Path,batch_size:int,shuffle:bool=True):
    train_dataset = DiffusionRankerDataset(repr_pt,reward_path)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collect)
    return train_loader

def test():
    pass

if __name__ == "__main__":
    test()