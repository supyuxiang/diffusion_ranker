import shutil
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn




_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))






def _sorted_candidate_images(candidate_dir: Path) -> list[Path]:
    return sorted(
        candidate_dir.glob('*png'),
        key=lambda p: int(p.stem.split('_')[0])
    )


def _reset_best_dir(best_dir: Path) -> None:
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)


def save_best_images(best_path_ls: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in best_path_ls:
        copy_path = output_dir / path.name
        shutil.copy2(path,copy_path)
    print(f"saved {len(best_path_ls)} best images to {output_dir}")










#############    train utils    #############

def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    KL(q || p)
    q is teacher_logits, p is softmax(student_logits)
    """
    log_p = F.log_softmax(student_logits, dim=-1)
    q = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


def mse_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_logits, teacher_logits, reduction="mean")

def js_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon divergence between student and teacher distributions.
    Input is logits; internally converts with softmax.
    """
    p = F.softmax(student_logits, dim=-1)
    q = F.softmax(teacher_logits, dim=-1)
    m = 0.5 * (p + q)
    eps = 1e-8

    # JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
    kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=-1)
    return 0.5 * (kl_pm + kl_qm).mean()


def pairwise_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Differentiable pairwise ranking loss.
    Uses teacher ordering to supervise student ordering:
    if teacher_i > teacher_j, encourage student_i > student_j.
    """
    s = student_logits.reshape(-1)
    t = teacher_logits.reshape(-1)
    if s.numel() != t.numel():
        raise ValueError(f"shape mismatch: student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}")

    teacher_diff = t[:, None] - t[None, :]  # (N, N)
    mask = teacher_diff > 0  # pairs where i should rank above j
    if not bool(mask.any()):
        return s.new_zeros(())

    student_diff = s[:, None] - s[None, :]  # (N, N)
    # logistic pairwise loss: log(1 + exp(-(s_i - s_j)))
    return F.softplus(-student_diff[mask]).mean()


def combined_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, alpha:float=0.8, kl_type:str='kl') -> torch.Tensor:
    assert kl_type in ['kl', 'js'], f"Invalid KL type: {kl_type}"
    
    if kl_type == 'kl':
        kl_term = kl_loss(student_logits, teacher_logits)
    else:
        kl_term = js_loss(student_logits, teacher_logits)

    return alpha * mse_loss(student_logits, teacher_logits) + (1 - alpha) * kl_term


class RankerLoss(nn.Module):
    def __init__(self, alpha:float=0.8, kl_type:str='kl'):
        super(RankerLoss,self).__init__()
        self.alpha = alpha
        self.kl_type = kl_type
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        return combined_loss(student_logits, teacher_logits, self.alpha, self.kl_type)

                

