"""
Diffusion ranker：吃 **单个** 已选定的 (step ratio, layer ratio, pooling) 对应的 hidden，
输出 reward / 排序 logits。

**输入约定**（单条候选一张图）：

- ``x`` shape ``(B, 1, hidden_dim)`` 或 ``(B, hidden_dim)``：数据侧只保留一种 (步, 层, pooling) 组合的一条向量。
- ``hidden_dim`` 与 DiT 一致（如 XL 为 1152）。

**Listwise**：同一上下文下 ``K`` 个候选，``x`` shape ``(B, K, 1, hidden_dim)`` 或 ``(B, K, hidden_dim)``，
先对每个候选做与 pointwise 相同的单向量编码，再在候选维上做 ``TransformerEncoder``（与
``language_ranker-main`` 里在候选序列上堆叠 TransformerBlock 的思路一致，此处用 batch_first API）。
"""


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiffusionSingleTokenEncoder(nn.Module):
    """
    单 token：``(B, 1, H)`` / ``(B, H)`` → ``(B, proj_dim)``。
    Linear + LayerNorm + 残差 FFN（对齐 ``RankModel_Transformer`` 中 linear + FFN 支路，省去 seq=1 时的 MHA）。
    """

    def __init__(
        self,
        hidden_dim: int,
        proj_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.norm_in = nn.LayerNorm(proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, 4 * proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * proj_dim, proj_dim),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(proj_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: ``(B, 1, hidden_dim)`` 或 ``(B, hidden_dim)``
        Returns:
            ``(B, proj_dim)``
        """
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError(
                    "expected (B, 1, hidden_dim) for a single (step, layer, pooling) token; "
                    f"got {tuple(x.shape)}"
                )
            x = x.squeeze(1)
        elif x.dim() != 2:
            raise ValueError(f"expected (B, 1, H) or (B, H), got {tuple(x.shape)}")
        h = self.proj(x)
        h = self.norm_in(h)
        h = h + self.mlp(h)
        return self.norm_out(h)


class PointwiseDiffusionRanker(nn.Module):
    """Pointwise reward：每个样本一个标量分数 ``(B,)``。"""

    def __init__(
        self,
        hidden_dim: int,
        proj_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        *,
        use_half: bool = False,
    ) -> None:
        super().__init__()
        # num_heads 保留签名以兼容旧调用；单 token 编码不使用 attention。
        self.encoder = DiffusionSingleTokenEncoder(hidden_dim, proj_dim, dropout=dropout)
        self.score_head = nn.Linear(proj_dim, 1)
        self._init_linear_weights()
        if use_half:
            self.to(torch.bfloat16)

    def _init_linear_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: ``(B, 1, hidden_dim)`` 或 ``(B, hidden_dim)``
        Returns:
            ``(B,)`` 标量 reward（未约束正负，由损失定义偏好）
        """
        h = self.encoder(x)
        return self.score_head(h).squeeze(-1)


class ListwiseDiffusionRanker(nn.Module):
    """Listwise：``K`` 个候选互相交互后各输出一个 logit ``(B, K)``。"""

    def __init__(
        self,
        hidden_dim: int,
        proj_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_list_layers: int = 2,
        *,
        use_half: bool = False,
    ) -> None:
        super().__init__()
        if proj_dim % num_heads != 0:
            raise ValueError(f"proj_dim ({proj_dim}) must be divisible by num_heads ({num_heads})")
        self.candidate_encoder = DiffusionSingleTokenEncoder(
            hidden_dim, proj_dim, dropout=dropout
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=4 * proj_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.list_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_list_layers)
        self.score_head = nn.Linear(proj_dim, 1)
        self._init_linear_weights()
        if use_half:
            self.to(torch.bfloat16)

    def _init_linear_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Tensor, *, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: ``(B, K, 1, hidden_dim)`` 或 ``(B, K, hidden_dim)``
            key_padding_mask: optional ``(B, K)``，``True`` 表示该候选为 padding
        Returns:
            ``(B, K)`` logits
        """
        if x.dim() == 4:
            if x.shape[2] != 1:
                raise ValueError(
                    "expected (B, K, 1, hidden_dim) for one token per candidate; "
                    f"got {tuple(x.shape)}"
                )
            b, k, _, hdim = x.shape
            flat = x.reshape(b * k, 1, hdim)
        elif x.dim() == 3:
            b, k, hdim = x.shape
            flat = x.reshape(b * k, hdim)
        else:
            raise ValueError(f"expected 3D or 4D input, got {tuple(x.shape)}")

        cand_emb = self.candidate_encoder(flat)
        cand_emb = cand_emb.view(b, k, -1)
        list_out = self.list_encoder(cand_emb, src_key_padding_mask=key_padding_mask)
        return self.score_head(list_out).squeeze(-1)


# --- 别名 ---
PointwiseRanker = PointwiseDiffusionRanker
ListwiseRanker = ListwiseDiffusionRanker
# 旧代码可能 import DiffusionHiddenEncoder；映射到新编码器
DiffusionHiddenEncoder = DiffusionSingleTokenEncoder


def listwise_logits_to_probs(logits: Tensor, dim: int = -1) -> Tensor:
    return F.softmax(logits, dim=dim)


if __name__ == "__main__":
    torch.manual_seed(0)
    b, h = 4, 1152
    x_pw = torch.randn(b, 1, h)
    pw = PointwiseDiffusionRanker(h, proj_dim=128, num_heads=8)
    s = pw(x_pw)
    assert s.shape == (b,), s.shape

    k = 8
    x_ls = torch.randn(b, k, 1, h)
    ls = ListwiseDiffusionRanker(h, proj_dim=128, num_heads=8, num_list_layers=2)
    logits = ls(x_ls)
    assert logits.shape == (b, k), logits.shape
    print("pointwise", s.shape, "listwise", logits.shape, "ok")
    print(f"pointwise: {pw.count_params()}, listwise: {ls.count_params()}")
