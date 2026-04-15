import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
import json



def load_prompt_repr(prompt_repr_pt:Path=Path('/root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_logits.pt')):
    prompt_repr = torch.load(prompt_repr_pt,map_location='cpu',weights_only=False)
    return prompt_repr # 1000,repr_dim


class DiffusionRankerDataset(Dataset): # output: candidate_hidden, prompt_hidden, reward
    def __init__(self,candidate_repr_path:Path,reward_path:Path,prompt_repr_path:Path):
        self.candidate_repr_path = candidate_repr_path
        self.reward_path = reward_path
        self.prompt_repr_path = prompt_repr_path
        self.load_repr_and_reward()
    
    def load_repr_and_reward(self):
        with open(self.reward_path,'r',encoding='utf-8') as f:
            self.reward_ls = json.load(f) # 1000,1

        self.candidate_repr = torch.load(self.candidate_repr_path,map_location='cpu')['tensor'] 
        self.candidate_hidden_dim = self.candidate_repr.shape[1]
        assert len(self.reward_ls) == len(self.candidate_repr)
        
        self.prompt_repr = load_prompt_repr(self.prompt_repr_path)
        self.prompt_hidden_dim = self.prompt_repr.shape[1]
        assert len(self.reward_ls) == len(self.prompt_repr)

    
    def __len__(self):
        return len(self.reward_ls)
    
    def __getitem__(self,idx):
        return self.candidate_repr[idx],self.prompt_repr[idx],self.reward_ls[idx]

def collect(batch:list[tuple[torch.Tensor,torch.Tensor,float]]):
    batch_candidate_hstates, batch_prompt_hstates, batch_rewards = [],[],[]
    for item in batch:
        candidate_hstates,prompt_hstates,reward = item
        batch_candidate_hstates.append(candidate_hstates)
        batch_prompt_hstates.append(prompt_hstates)
        batch_rewards.append(reward)
    batch_candidate_hstates = torch.cat(batch_candidate_hstates,dim=0) # B,K,hidden_dim
    batch_prompt_hstates = torch.cat(batch_prompt_hstates,dim=0) # B,hidden_dim
    batch_rewards = torch.tensor(batch_rewards).view(-1,1) # B,1
    return batch_candidate_hstates,batch_prompt_hstates,batch_rewards


def diffusion_ranker_dataloader(candidate_repr_path:Path,reward_path:Path,prompt_repr_path:Path,batch_size:int,shuffle:bool=True):
    train_dataset = DiffusionRankerDataset(candidate_repr_path,reward_path,prompt_repr_path)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collect)
    return train_loader

def test():
    candidate_repr_path = Path('/root/diffusion_ranker/ranker/candidate_repr_mean_pooling/from_logits.pt')
    reward_path = Path('/root/diffusion_ranker/ranker/reward.json')
    prompt_repr_path = Path('/root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_logits.pt')
    batch_size = 4
    shuffle = True
    train_loader = dataloader(candidate_repr_path,reward_path,prompt_repr_path,batch_size,shuffle)
    for batch in train_loader:
        print(batch)

if __name__ == "__main__":
    test()