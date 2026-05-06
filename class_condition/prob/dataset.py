from __future__ import annotations

import torch
from torch.utils.data import Dataset


class HiddenRewardDataset(Dataset):
    def __init__(self, hiddens: torch.Tensor, rewards: torch.Tensor):
        self.hiddens = hiddens.float()
        self.rewards = rewards.view(-1).float()
        if self.hiddens.dim() != 2:
            raise ValueError(f"hiddens must be 2D, got {tuple(self.hiddens.shape)}")
        if self.hiddens.shape[0] != self.rewards.shape[0]:
            raise ValueError(f"row mismatch: {self.hiddens.shape[0]} vs {self.rewards.shape[0]}")

    def __len__(self) -> int:
        return int(self.hiddens.shape[0])

    def __getitem__(self, idx: int):
        return self.hiddens[idx], self.rewards[idx]

