import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from pathlib import Path



def kl_reward_ranker_loss(
    pred_scores: torch.Tensor, teacher_scores: torch.Tensor, tau: float
) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    # shape: (batch_size, group_size, 1)
    q = F.softmax(teacher_scores / tau, dim=0)
    log_p = F.log_softmax(pred_scores / tau, dim=0)
    log_q = torch.log(q + 1e-8)
    return (q * (log_q - log_p)).sum()



class mse_kl_combined_loss(nn.Module):
    def __init__(self, alpha:float,tau:float):
        super(mse_kl_combined_loss,self).__init__()
        self.alpha = alpha
        self.tau = tau

    def forward(self, pred_scores:torch.Tensor, teacher_scores:torch.Tensor):
        mse = F.mse_loss(pred_scores, teacher_scores)
        kl = kl_reward_ranker_loss(pred_scores, teacher_scores, self.tau)
        alpha = float(self.alpha)
        total = alpha * mse + (1.0 - alpha) * kl
        return total, mse, kl