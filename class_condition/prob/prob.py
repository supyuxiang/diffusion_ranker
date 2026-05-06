from __future__ import annotations

import torch
import torch.nn as nn


class RewardProb(nn.Module):
    """Predict scalar reward from pooled DiT hidden states only."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                )
                for _ in range(num_layers)
            ]
        )
        self.reward_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(hidden)
        for block in self.layers:
            x = x + block(x)
        return self.reward_head(x)

