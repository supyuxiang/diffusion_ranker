#!/usr/bin/env python3
r"""
极简演示：**scaled dot-product attention**、**self-attention**、**cross-attention**。

与 ``ranker/model.py`` 里 ``TransformerBlock`` 的关系：

- 那里写的是 ``MultiheadAttention(x, x, x)``：Q、K、V 都来自**同一条序列** ``x``，
  所以是 **self-attention**（序列里每个位置看全体位置，含 prompt 与各 candidate）。
- 若写成 ``MultiheadAttention(query=candidates, key=prompt, value=prompt)``，
  才是典型的 **cross-attention**：query 只来自候选，K/V 只来自 prompt（或反过来）。

运行::

    python3 demo/demo_self_vs_cross_attention.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def scaled_dot_product_attention_multihead(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    多头版（最后一维是 **每个 head 的 head_dim**，不是总 embed_dim）::

        q: (L_q, num_heads, head_dim)
        k: (L_k, num_heads, head_dim)
        v: (L_k, num_heads, head_dim)

    每个 head：``scores = Q K^T / sqrt(head_dim)``，**不能**写成 ``Q @ K``（维度对不上）。

    返回 ``out`` (L_q, num_heads * head_dim)，``weights`` (num_heads, L_q, L_k)。
    """
    l_q, num_heads, head_dim = q.shape
    qh = q.transpose(0, 1)  # (H, L_q, D)
    kh = k.transpose(0, 1)
    vh = v.transpose(0, 1)
    scores = qh @ kh.transpose(-2, -1) / math.sqrt(head_dim)
    weights = F.softmax(scores, dim=-1)
    out = weights @ vh
    out = out.transpose(0, 1).contiguous().view(l_q, num_heads * head_dim)
    return out, weights


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q: (Lq, d_k)  —— 每个 query 位置一个向量
    k: (Lk, d_k)  —— 每个 key 位置一个向量（与 q 的最后一维相同）
    v: (Lk, d_v)  —— 每个 key 位置对应一个 value

    返回:
      out: (Lq, d_v)  —— 每个 query 位置对全体 key 加权后的向量
      weights: (Lq, Lk) —— 注意力权重（softmax 后），便于打印理解
    """
    d_k = q.size(-1)
    # 相似度：query 与每个 key 的点积，越大越「相关」
    scores = q @ k.T / math.sqrt(d_k)  # (Lq, Lk)
    weights = F.softmax(scores, dim=-1)
    out = weights @ v  # (Lq, d_v)
    return out, weights


def mask_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    **带 mask 的 self-attention**（单头、无 ``W_q/W_k/W_v``，与 ``scaled_dot_product_attention`` 同一形状约定）。

    - ``q, k, v`` 形状与 ``scaled_dot_product_attention`` 相同：``(Lq, d_k)``、``(Lk, d_k)``、``(Lk, d_v)``。
      典型 **self-attention** 下 ``Lq == Lk`` 且 ``q,k,v`` 同源。

    - ``is_causal=True``（decoder 常用）：位置 ``i`` 只能 attend ``j <= i``，在 scores 上对 **上三角**
     （不含对角）加 ``-inf`` 后再 softmax。

    - ``attn_mask``：与 ``scores`` 同形状 ``(Lq, Lk)`` 的 **加性** mask（``0`` 表示不挡，
      ``-inf`` 表示禁止该 (query, key) 对）。与 ``is_causal`` 不要同时为 True（会 assert）。

    返回 ``(out, weights)``，``weights`` 为 softmax 后的 ``(Lq, Lk)``。
    """
    if is_causal and attn_mask is not None:
        raise ValueError("is_causal 与自定义 attn_mask 请二选一")

    d_k = q.size(-1)
    scores = q @ k.T / math.sqrt(d_k)  # (Lq, Lk)

    if is_causal:
        l_q, l_k = q.size(0), k.size(0)
        if l_q != l_k:
            raise ValueError("因果 self-attention 需要 Lq == Lk")
        causal = torch.triu(
            torch.full(
                (l_q, l_k),
                float("-inf"),
                dtype=scores.dtype,
                device=scores.device,
            ),
            diagonal=1,
        )
        scores = scores + causal
    elif attn_mask is not None:
        if attn_mask.shape != scores.shape:
            raise ValueError(f"attn_mask 形状须为 {tuple(scores.shape)}，现为 {tuple(attn_mask.shape)}")
        scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)

    weights = F.softmax(scores, dim=-1)
    out = weights @ v
    return out, weights


class CrossAttention(nn.Module):
    """
    标准 **cross-attention**（decoder 风格）：Q 来自 ``x``，K/V 来自 ``memory``。

    - ``x``: ``(B, L_q, d_query)``
    - ``memory``: ``(B, L_k, d_memory)``
    - 内部投影到同一 ``embed_dim`` 后调用 ``nn.MultiheadAttention``（``batch_first=True``）。
    """

    def __init__(
        self,
        d_query: int,
        d_memory: int,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须整除 num_heads")
        self.q_proj = nn.Linear(d_query, embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=d_memory,
            vdim=d_memory,
        )

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.q_proj(x)
        return self.mha(q, memory, memory, need_weights=True, average_attn_weights=True)


def cross_attention(
    x: torch.Tensor,
    memory: torch.Tensor,
    num_heads: int,
    d_k: int,
    layer: CrossAttention,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    对 ``CrossAttention`` 模块的薄封装（权重在 ``layer`` 里，避免每次调用新建随机参数）。

    - ``x``: ``(B, L_x, d_query)``（须与构造 ``layer`` 时的 ``d_query`` 一致）
    - ``memory``: ``(B, L_memory, d_memory)``
    - ``d_k``: 与 ``layer`` 的 ``embed_dim`` 一致；传入用于 **assert** 形状，避免误用

    返回 ``(out, attn_weights)``，``out`` 形状 ``(B, L_x, embed_dim)``。
    """
    if x.dim() != 3 or memory.dim() != 3:
        raise ValueError("x 与 memory 须为 (B, L, D)")
    if x.shape[0] != memory.shape[0]:
        raise ValueError("batch 维须相同")
    if x.shape[-1] != layer.q_proj.in_features:
        raise ValueError("x 最后一维须等于 CrossAttention 的 d_query")
    if memory.shape[-1] != layer.mha.kdim:
        raise ValueError("memory 最后一维须等于 CrossAttention 的 d_memory")
    embed_dim = layer.q_proj.out_features
    if embed_dim != d_k:
        raise ValueError(f"d_k={d_k} 须等于 CrossAttention 的 embed_dim={embed_dim}")
    if num_heads != layer.mha.num_heads:
        raise ValueError("num_heads 须与构造 CrossAttention 时一致")
    return layer(x, memory)


def demo_self_attention() -> None:
    print("=== Self-attention：Q、K、V 都来自同一条序列 X ===\n")
    # 3 个 token，每个 4 维（玩具维度）
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # token 0
            [0.0, 1.0, 0.0, 0.0],  # token 1
            [0.5, 0.5, 0.0, 0.0],  # token 2（前两维都有）
        ]
    )
    # 极简：不做独立 W_q/W_k/W_v，直接用 x 当 Q、K、V（仅教学；真实模型里会线性投影）
    q = k = v = x
    out, w = scaled_dot_product_attention(q, k, v)
    print("X (每行一个 token):\n", x)
    print("\n每个 query 行对全体 key 的注意力权重 (Lq, Lk):\n", w)
    print("\n输出：每个位置 = 按权重对 **所有 token 的 v** 求和\n", out)
    print(
        "\n含义：token 2 同时「像」0 和 1，它的输出会混合二者；"
        "self-attention 让**序列内部**互相传递信息。\n"
    )


def demo_cross_attention() -> None:
    print("=== Cross-attention：Q 来自序列 A，K/V 来自序列 B（另一源）===\n")
    # B：2 个「记忆 / 条件」向量（例如 prompt）
    memory = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    # A：3 个「查询」向量（例如 3 个 candidate 的 hidden）
    queries = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    q = queries
    k = v = memory
    out, w = scaled_dot_product_attention(q, k, v)
    print("Queries（每行一个要解码的位置，例如 candidate）:\n", q)
    print("Memory K/V（例如 prompt 的两个 token）:\n", memory)
    print("\n权重：每个 query 在 **memory 的 Lk 个位置上**怎么分配注意力\n", w)
    print("\n输出：每个 query 只从 memory 里加权读出，**query 之间不直接互看**\n", out)
    print(
        "与 self-attention 的区别：若把 Q、K、V 都换成拼起来的长序列，"
        "candidate 之间也会互相看；cross-attention 则限制「只看 memory」。\n"
    )


def demo_masked_self_attention() -> None:
    print("=== 因果 mask self-attention（decoder）：``mask_self_attention(..., is_causal=True)`` ===\n")
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    q = k = v = x
    out, w = mask_self_attention(q, k, v, is_causal=True)
    print("因果 softmax 权重（上三角应为 0）：\n", w)
    print("\n输出：\n", out)
    print(
        "\n含义：第 0 行只看 key0；第 2 行只看 key0..2，看不到「未来」位置。\n"
    )


def demo_ranker_style_concat_is_self_attn() -> None:
    print(
        "=== 与 DiffusionRanker_Transformer 一致的做法："
        "先 cat([prompt, candidates])，再 self-attention ===\n"
    )
    hidden = 8
    # 1 个 prompt token + 3 个 candidate → 长度 4
    seq = torch.randn(4, hidden)
    mha = nn.MultiheadAttention(
        embed_dim=hidden, num_heads=2, batch_first=False, dropout=0.0
    )
    # PyTorch: (L, N, E)；N=batch，这里 batch=1
    x = seq.unsqueeze(1)  # (4, 1, 8)
    out, attn_w = mha(x, x, x, average_attn_weights=False)
    print("序列长度 L=4（位置 0 可视为 prompt，1..3 为 candidates）")
    print("调用: MultiheadAttention(x, x, x)  →  self-attention on 整条序列")
    print("输出形状 (L, N, E):", tuple(out.shape))
    print(
        "此时 prompt 与每个 candidate 都会在注意力图里出现；"
        "candidate 之间也会互相 attend（listwise 交互）。\n"
    )


if __name__ == "__main__":
    demo_self_attention()
    demo_cross_attention()
    demo_masked_self_attention()
    demo_ranker_style_concat_is_self_attn()
