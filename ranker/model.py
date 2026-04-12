import torch
import torch.nn as nn
import torch.nn.functional as F




class ListwiseRanker(nn.Module):
    def __init__(self, hidden_dim:int, cfg:float, num_layers:int, proj_dim:int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        self.num_layers = num_layers
        self.proj_dim = proj_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def count_params(self) -> int:
        return sum(
            p.numel() for p in self.parameters()
        ) #  (return / 1e6) M

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_layers, hidden_dim), where hidden_dim is from the DiT model
        # return: (batch_size, num_layers)
        pass


class _TransformerBlock(nn.Module):
    def __init__(self, hidden_dim:int, num_heads:int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass




class PointwiseRanker(nn.Module):
    def __init__(self, hidden_dim:int, proj_dim:int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def count_params(self) -> int:
        return sum(
            p.numel() for p in self.parameters()
        ) #  (return / 1e6) M
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    pass











