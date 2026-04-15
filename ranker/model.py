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



class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim:int,num_heads:int,dropout:float):
        super(TransformerBlock,self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=num_heads,dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self,x:Tensor,key_padding_mask:Tensor|None=None)->Tensor:
        attn_output, _ = self.self_attention(x,x,x,key_padding_mask=key_padding_mask)
        x = self.norm2(x+attn_output)
        output = x+self.ffn(x)
        output = self.norm3(output+x)
        return output

    

class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(MLP, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DiffusionRanker_Transformer(nn.Module):
    def __init__(
        self,
        input_dim_candidate:int,
        input_dim_prompt:int,
        hidden_dim:int,
        num_heads:int,
        dropout:float,
        num_layers:int,
        half:bool=True
    ):
        super(DiffusionRanker_Transformer,self).__init__()
        self.input_dim_prompt = input_dim_prompt
        self.input_dim_candidate = input_dim_candidate

        self.linear_candidate = nn.Linear(input_dim_candidate,hidden_dim)
        self.linear_prompt = nn.Linear(input_dim_prompt,hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim,num_heads,dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_weights = nn.Parameter(torch.empty(hidden_dim,1))
        nn.init.xavier_uniform_(self.output_weights)
        self._init_weights()
        if half:
            self.to(torch.bfloat16)

    def _init_weights(self):
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        for module in self.linear_candidate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.linear_prompt.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self,candidate_hstates,prompt_hstates,key_padding_mask=None):
        # candidate_hstates: (B, K, input_dim_candidate) -> linear -> (B, K, hidden_dim)
        b, k, _ = candidate_hstates.shape

        candidate_hstates = self.linear_candidate(candidate_hstates)  # (B, K, hidden_dim)

        if prompt_hstates is not None:
            # prompt_hstates: (B, K, input_dim_prompt) -> linear -> (B, K, hidden_dim)
            prompt_hstates = self.linear_prompt(prompt_hstates)  # (B, hidden_dim)
            prompt_hstates = prompt_hstates.unsqueeze(0) # (1, B, hidden_dim)
        else:
            # 与有 prompt 时相同占位形状，便于与 (K, B, H) 在 dim=1 对齐
            prompt_hstates = torch.zeros(
                1,
                candidate_hstates.shape[0],
                candidate_hstates.shape[2],
                device=candidate_hstates.device,
                dtype=candidate_hstates.dtype,
            )
        

        candidate_hstates = candidate_hstates.permute(
            1,0,2
        )# b,k,hidden_dim -> k,b,hidden_dim
        x = torch.cat((prompt_hstates,candidate_hstates),dim=0) # k+1, b , hidden_dim

        for layer in self.layers:
            x = x + layer(x,key_padding_mask=key_padding_mask)
        # after : k+1, b ,hidden_dim
        
        logits = torch.matmul(x,self.output_weights) # k+1, b ,1
        logits = logits.squeeze(-1) # k+1,b
        logits = logits.permute(1,0) # b,k+1
        logits = logits[:,1:] # b,k
        return logits
    
    def count_params(self):
        return sum(
            p.numel() for p in self.parameters()
        )
    



class DiffusionRanker_MLP(nn.Module):
    def __init__(
        self,
        candidate_hidden_dim:int,
        prompt_hidden_dim:int,
        hidden_dim:int,
        output_dim:int,
        num_layers:int,
        dropout:float,
        half:bool=True,
        criterion=None,
    ):
        super(DiffusionRanker_MLP,self).__init__()
        self.linear_prompt = nn.Linear(prompt_hidden_dim,hidden_dim)
        self.linear_candidate = nn.Linear(candidate_hidden_dim,hidden_dim)
        self.criterion = criterion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [self.ffn for _ in range(num_layers)]
        )
        self.projection = nn.Linear(hidden_dim,output_dim)
        self._init_weights()
        if half:
            self.to(torch.bfloat16)
    
    def _init_weights(self):
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        for module in self.linear_candidate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.linear_prompt.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,candidate_hstates,prompt_hstates):
        # candidate_hstates: b,k,candidate_hidden_dim
        # prompt_hstates: b,prompt_hidden_dim
        prompt_hstates = self.linear_prompt(prompt_hstates)
        candidate_hstates = self.linear_candidate(candidate_hstates)
        # candidate_hstates: b,k,hidden_dim
        # prompt_hstates: b,hidden_dim
        prompt_hstates = prompt_hstates.unsqueeze(1)
        # prompt_hstates: b,1,hidden_dim
        x = torch.cat((prompt_hstates,candidate_hstates),dim=1) # b,k+1,hidden_dim
        for layer in self.layers:
            x = x + layer(x)
        # b,1,hidden_dim
        if self.criterion is None:
            return x[:,1:,0] # b,k
        elif 'cos' in self.criterion:
            t = float(self.criterion.split("_t")[-1])
            x = nn.functional.normalize(x,p=2.0,dim=-1,eps=1e-4)
            B,L,D = x[:,1:].shape
            out = torch.empty(B,L,device=x.device,dtype=x.dtype)
            prompt_x = x[:,0]
            cand_x = x[:,1:]
            for b in range(B):
                for l in range(L):
                    out[b,l] = (prompt_x[b] * cand_x[b,l]).sum()
            return out / t
        else:
            raise NotImplementedError
            
    def count_params(self):
        return sum(
            p.numel() for p in self.parameters()
        )
    




default_ranker = DiffusionRanker_Transformer(
    input_dim_candidate=1152,
    input_dim_prompt=4000,
    hidden_dim=64,
    num_heads=2,
    dropout=0.1,
    num_layers=2,
    half=True
)
# 0.43M


pointwise_ranker = DiffusionRanker_MLP(
    candidate_hidden_dim=1152,
    prompt_hidden_dim=3804,
    hidden_dim=64,
    output_dim=1,
    num_layers=2,
    dropout=0.1,
    half=True,
    criterion='cos_t1.0',
)



if __name__ == "__main__":
   print(default_ranker.count_params())
   dtype = next(default_ranker.parameters()).dtype
   candidate_hstates = torch.randn(4, 16, 1152, dtype=dtype)
   # 须与 default_ranker.input_dim_prompt 一致（当前为 4000，勿用 pointwise 的 3804）
   prompt_hstates = torch.randn(4, 4000, dtype=dtype)
   logits = default_ranker.forward(candidate_hstates, prompt_hstates)
   print(logits)
   print(logits.shape) # 4,16
   prob = F.softmax(logits,dim=-1)
   print(prob)
   best_idx = torch.argmax(prob,dim=-1)
   print(best_idx)

   print('--------------------------------')
   print('pointwise_ranker')
   print(pointwise_ranker.count_params())
   dtype = next(pointwise_ranker.parameters()).dtype
   candidate_hstates = torch.randn(4, 16, 1152, dtype=dtype)
   prompt_hstates = torch.randn(4, 3804, dtype=dtype)
   logits = pointwise_ranker.forward(candidate_hstates, prompt_hstates)
   print(logits)
   print(logits.shape) # 4,16
   prob = F.softmax(logits,dim=-1)
   print(prob)
   best_idx = torch.argmax(prob,dim=-1)
   print(best_idx)

