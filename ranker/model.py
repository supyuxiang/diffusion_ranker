"""
Diffusion ranker：吃 DiT rollout 抓取的 **中间步 × 层 × pooling** 的 hidden states（如 ``mean`` / ``last``），
输出 reward / 排序 logits。与 ``language_ranker-main/ranker/ranker_structure.py`` 类似，分 **pointwise** 与 **listwise**。

**输入约定**（单条候选一张图对应的表征）：

- ``x`` shape ``(B, N_tokens, hidden_dim)``，其中 ``N_tokens = n_steps × n_layers × n_poolings``（你在数据侧把
  若干 ``s*``、``r*``、``mean``/``last`` 向量按固定顺序拼成序列即可）。
- ``hidden_dim`` 与 DiT 一致（如 XL 为 1152）。

**Listwise**：同一上下文下 ``K`` 个候选，``x`` shape ``(B, K, N_tokens, hidden_dim)``，输出每个候选一个标量 logit
``(B, K)``，供 softmax / ListNet / ListMLE 等使用。
"""


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiffusionHiddenEncoder(nn.Module):
    """将单候选的多 token 表征 ``(B, N, H)`` 压成一条向量 ``(B, proj_dim)``（learned query attention pool）。"""

    def __init__(
        self,
        hidden_dim: int,
        proj_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if proj_dim % num_heads != 0:
            raise ValueError(f"proj_dim ({proj_dim}) must be divisible by num_heads ({num_heads})")
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.norm_in = nn.LayerNorm(proj_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.empty(1, 1, proj_dim))
        nn.init.normal_(self.query, std=0.02)
        self.norm_mid = nn.LayerNorm(proj_dim)
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
            x: ``(B, N_tokens, hidden_dim)``
        Returns:
            ``(B, proj_dim)``
        """
        x = self.proj(x)
        x = self.norm_in(x)
        b = x.size(0)
        q = self.query.expand(b, -1, -1)
        attn_out, _ = self.attn(q, x, x, need_weights=False)
        h = attn_out.squeeze(1)
        h = self.norm_mid(h + self.mlp(h))
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
        self.encoder = DiffusionHiddenEncoder(
            hidden_dim, proj_dim, num_heads=num_heads, dropout=dropout
        )
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
            x: ``(B, N_tokens, hidden_dim)``
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
        self.candidate_encoder = DiffusionHiddenEncoder(
            hidden_dim, proj_dim, num_heads=num_heads, dropout=dropout
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
            x: ``(B, K, N_tokens, hidden_dim)``
            key_padding_mask: optional ``(B, K)``，``True`` 表示该候选为 padding（与 ``TransformerEncoder`` 约定一致）
        Returns:
            ``(B, K)`` logits
        """
        b, k, n, hdim = x.shape
        flat = x.reshape(b * k, n, hdim)
        cand_emb = self.candidate_encoder(flat)
        cand_emb = cand_emb.view(b, k, -1)
        list_out = self.list_encoder(cand_emb, src_key_padding_mask=key_padding_mask)
        return self.score_head(list_out).squeeze(-1)


# --- 与旧占位类名对齐的别名（便于已有 import） ---
PointwiseRanker = PointwiseDiffusionRanker
ListwiseRanker = ListwiseDiffusionRanker


def listwise_logits_to_probs(logits: Tensor, dim: int = -1) -> Tensor:
    return F.softmax(logits, dim=dim)


if __name__ == "__main__":
    torch.manual_seed(0)
    b, n_tok, h = 4, 50, 1152
    x_pw = torch.randn(b, n_tok, h)
    pw = PointwiseDiffusionRanker(h, proj_dim=128, num_heads=8)
    s = pw(x_pw)
    assert s.shape == (b,), s.shape

    k = 8
    x_ls = torch.randn(b, k, n_tok, h)
    ls = ListwiseDiffusionRanker(h, proj_dim=128, num_heads=8, num_list_layers=2)
    logits = ls(x_ls)
    assert logits.shape == (b, k), logits.shape
    print("pointwise", s.shape, "listwise", logits.shape, "ok")
