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


def mse_kl_combined_loss(
    pred_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    tau: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(pred_scores, teacher_scores)
    kl = kl_reward_ranker_loss(pred_scores, teacher_scores, tau)
    a = float(alpha)
    total = a * mse + (1.0 - a) * kl
    return total, mse, kl