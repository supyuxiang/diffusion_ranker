"""
Text2Image ranker modules.

This file mirrors the structure of `class_condition/ranker.py` and keeps
compatible class names for easier reuse in training / search scripts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        attn_output, _ = self.self_attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm2(x + attn_output)
        return self.norm3(x + self.ffn(x))


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DiffusionRanker_Transformer(nn.Module):
    def __init__(
        self,
        input_dim_candidate: int,
        input_dim_prompt: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        half: bool = True,
    ):
        super().__init__()
        self.input_dim_prompt = input_dim_prompt
        self.input_dim_candidate = input_dim_candidate
        self.hidden_dim = hidden_dim

        self.linear_candidate = nn.Linear(input_dim_candidate, hidden_dim)
        self.linear_prompt = nn.Linear(input_dim_prompt, hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_weights = nn.Parameter(torch.empty(hidden_dim, 1))
        nn.init.xavier_uniform_(self.output_weights)
        self._init_weights()
        if half:
            self.to(torch.bfloat16)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        candidate_hstates: Tensor,
        prompt_hstates: Tensor | None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # candidate_hstates: (B, K, D_c), prompt_hstates: (B, D_p) or (B, 1, D_p)
        batch_size = candidate_hstates.shape[0]
        x_c = self.linear_candidate(candidate_hstates)

        if prompt_hstates is not None:
            if prompt_hstates.dim() == 2:
                prompt_hstates = prompt_hstates.unsqueeze(1)
            x_p = self.linear_prompt(prompt_hstates)
        else:
            x_p = torch.zeros(
                batch_size,
                1,
                self.hidden_dim,
                device=x_c.device,
                dtype=x_c.dtype,
            )

        x = torch.cat((x_p, x_c), dim=1)  # (B, K+1, H)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        logits = torch.matmul(x, self.output_weights).squeeze(-1)  # (B, K+1)
        return logits[:, 1:]  # (B, K)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DiffusionRanker_MLP(nn.Module):
    def __init__(
        self,
        candidate_hidden_dim: int,
        prompt_hidden_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        half: bool = True,
        criterion: str | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.criterion = criterion
        self.process_prompt = nn.Sequential(
            nn.Linear(prompt_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.linear_candidate = nn.Linear(candidate_hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([MLPBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.projection = nn.Linear(hidden_dim, output_dim)
        self._init_weights()
        if half:
            self.to(torch.bfloat16)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, candidate_hstates: Tensor, prompt_hstates: Tensor | None) -> Tensor:
        # candidate_hstates: (B, K, D_c), prompt_hstates: (B, D_p)
        if prompt_hstates is not None:
            x_p = self.process_prompt(prompt_hstates).unsqueeze(1)
        else:
            x_p = torch.zeros(
                candidate_hstates.shape[0],
                1,
                self.hidden_dim,
                device=candidate_hstates.device,
                dtype=candidate_hstates.dtype,
            )

        x_c = self.linear_candidate(candidate_hstates)
        x = torch.cat((x_p, x_c), dim=1)  # (B, K+1, H)
        for layer in self.layers:
            x = x + layer(x)

        if self.criterion is None:
            return self.projection(x[:, 1:, :]).squeeze(-1)
        if "cos" in self.criterion:
            t = float(self.criterion.split("_t")[-1])
            x = F.normalize(x, p=2.0, dim=-1, eps=1e-4)
            prompt_x = x[:, 0]  # (B, H)
            cand_x = x[:, 1:]  # (B, K, H)
            scores = torch.einsum("bh,bkh->bk", prompt_x, cand_x)
            return scores / t
        raise NotImplementedError(f"Unsupported criterion: {self.criterion}")

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DiffusionRanker_CrossAttention(nn.Module):
    def __init__(
        self,
        candidate_hidden_dim: int,
        prompt_hidden_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        half: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_prompt_norm = nn.LayerNorm(prompt_hidden_dim)
        self.input_candidate_norm = nn.LayerNorm(candidate_hidden_dim)
        self.prompt_hidden_norm = nn.LayerNorm(hidden_dim)
        self.candidate_hidden_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)

        self.process_prompt = nn.Sequential(
            nn.Linear(prompt_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.process_candidate = nn.Sequential(
            nn.Linear(candidate_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        n_layer = max(1, int(num_layers))
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layer)
            ]
        )
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layer)])
        self.score_head = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.score_head.weight)

        self._init_weights()
        if half:
            self.to(torch.bfloat16)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, candidate_hstates: Tensor, prompt_hstates: Tensor | None) -> Tensor:
        # candidate_hstates: (B, K, D_c), prompt_hstates: (B, D_p)
        if prompt_hstates is not None:
            prompt_hstates = self.input_prompt_norm(prompt_hstates)
            prompt_hstates = self.process_prompt(prompt_hstates)
            prompt_hstates = self.prompt_hidden_norm(prompt_hstates).unsqueeze(1)  # (B,1,H)
        else:
            prompt_hstates = torch.zeros(
                candidate_hstates.shape[0],
                1,
                self.hidden_dim,
                device=candidate_hstates.device,
                dtype=candidate_hstates.dtype,
            )

        candidate_hstates = self.input_candidate_norm(candidate_hstates)
        candidate_hstates = self.process_candidate(candidate_hstates)
        candidate_hstates = self.candidate_hidden_norm(candidate_hstates)

        score_ls = []
        for k in range(candidate_hstates.shape[1]):
            ck = candidate_hstates[:, k : k + 1, :]
            kv = torch.cat([prompt_hstates, ck], dim=1)
            x = self.query_norm(ck)
            kv = self.kv_norm(kv)
            for attn, norm in zip(self.attn_layers, self.attn_norms):
                attn_out, _ = attn(query=x, key=kv, value=kv)
                x = norm(x + attn_out)
            score_ls.append(self.score_head(x.squeeze(1)))
        return torch.stack(score_ls, dim=1).squeeze(-1)  # (B, K)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DiffusionProb_SingleClass(nn.Module):
    def __init__(
        self,
        hs_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        half: bool = True,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(hs_dim)
        self.linear_candidate = nn.Linear(hs_dim, hidden_dim)
        self.layers = nn.ModuleList([MLPBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
        self._init_weights()
        if half:
            self.to(torch.bfloat16)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hs: Tensor) -> Tensor:
        x = self.input_norm(hs)
        x = self.linear_candidate(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.output_norm(x)
        return self.reward_head(x)


default_ranker = DiffusionRanker_Transformer(
    input_dim_candidate=1152,
    input_dim_prompt=4096,
    hidden_dim=64,
    num_heads=2,
    dropout=0.1,
    num_layers=2,
    half=True,
)

pointwise_ranker = DiffusionRanker_MLP(
    candidate_hidden_dim=1152,
    prompt_hidden_dim=4096,
    hidden_dim=128,
    output_dim=1,
    num_layers=8,
    dropout=0.1,
    half=True,
    criterion="cos_t1.0",
)

cross_attention_ranker = DiffusionRanker_CrossAttention(
    candidate_hidden_dim=1152,
    prompt_hidden_dim=4096,
    hidden_dim=128,
    num_heads=4,
    num_layers=8,
    dropout=0.1,
    half=True,
)


if __name__ == "__main__":
    print(default_ranker.count_params())
    dtype = next(default_ranker.parameters()).dtype
    candidate_hstates = torch.randn(4, 16, 1152, dtype=dtype)
    prompt_hstates = torch.randn(4, 4096, dtype=dtype)
    logits = default_ranker(candidate_hstates, prompt_hstates)
    print(logits.shape)
