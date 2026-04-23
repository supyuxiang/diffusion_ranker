"""
Diffusion ranker


ranker_type:
    - transformer
    - mlp
    - cross_attention
"""


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ------------------------------- I love Peking University -------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim:int,num_heads:int,dropout:float):
        super(TransformerBlock,self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
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
        # candidate_hstates: mini_b,k,d_candidate, prompt_hstates: mini_b,1,d_prompt
        mini_b, k, d_candidate = candidate_hstates.shape # mini_b,k,d_candidate

        candidate_hstates = self.linear_candidate(candidate_hstates)  # (mini_b,k, hidden_dim)

        if prompt_hstates is not None: # mini_b,1, d_prompt
            # prompt_hstates: 
            prompt_hstates = self.linear_prompt(prompt_hstates)  # mini_b,1,hidden_dim
        else:
            # 与有 prompt 时相同占位形状，便于与 (K, B, H) 在 dim=1 对齐
            prompt_hstates = torch.zeros(
                mini_b,
                1,
                self.hidden_dim,
                device=candidate_hstates.device,
                dtype=candidate_hstates.dtype,
            )# mini_b,1,hidden_dim

        # concat prompt_hstates and candidate_hstates
        x = torch.cat((prompt_hstates, candidate_hstates),dim=1) # mini_b,k+1,hidden_dim

        # candidate_hstates = candidate_hstates.permute(
        #     1,0,2
        # )# b,k,hidden_dim -> k,b,hidden_dim
        # x = torch.cat((prompt_hstates,candidate_hstates),dim=0) # k+1, b , hidden_dim

        for layer in self.layers:
            x = layer(x,key_padding_mask=key_padding_mask)
        # mini_b,k+1,hidden_d

        # b,k+1,hidden_dim -> b, k+1, 1
        logits = torch.matmul(x,self.output_weights) # mini_b, k+1, 1
        logits = logits.squeeze(-1) # mini_b,k+1
        # logits = logits[:,1:] + logits[:,0:1] # mini_b,k
        logits = logits[:,1:] # mini_b,k
        # logits = nn.Sigmoid()(logits) # mini_b,k
        return logits
        
        # logits = torch.matmul(x,self.output_weights) # k+1, b ,1
        # logits = logits.squeeze(-1) # k+1,b
        # logits = logits.permute(1,0) # b,k+1
        # logits = logits[:,1:] # b,k
        # return logits
    
    def count_params(self):
        return sum(
            p.numel() for p in self.parameters()
        )





# class DiffusionRanker_Transformer_SingleClass(nn.module):
#     def __init__(
#         self,
#         input_dim_candidate:int,
#         input_dim_prompt:int,
#         hidden_dim:int,
#         num_heads:int,
#         dropout:float,
#         num_layers:int,
#     ):
#         super(DiffusionRanker_Transformer_SingleClass,self).__init__()
#         self.input_dim_prompt = input_dim_prompt
#         self.input_dim_candidate = input_dim_candidate

#         self.linear_candidate = nn.Linear(input_dim_candidate,hidden_dim)
#         self.linear_prompt = nn.Linear(input_dim_prompt,hidden_dim)
#         self.layers = nn.ModuleList(
#             [
#                 TransformerBlock(hidden_dim,num_heads,dropout)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.output_weights = nn.Parameter(torch.empty(hidden_dim,1))
#         nn.init.xavier_uniform_(self.output_weights)
#         self._init_weights()
#         if half:
#             self.to(torch.bfloat16)

#     def _init_weights(self):
#         for layer in self.layers:
#             for module in layer.modules():
#                 if isinstance(module, nn.Linear):
#                     nn.init.xavier_uniform_(module.weight)
#                     if module.bias is not None:
#                         nn.init.zeros_(module.bias)
#         for module in self.linear_candidate.modules():
#             pass

#     def forward(self,candidate_hstates,prompt_hstates,key_padding_mask=None):
#         # candidate_hstates: b,k,d_candidate
#         # prompt_hstates: 1,d_prompt
#         pass


        








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
        self.hidden_dim = hidden_dim
        # self.linear_prompt = nn.Linear(prompt_hidden_dim,hidden_dim)
        self.process_prompt = nn.Sequential(
            nn.Linear(prompt_hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
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
        for module in self.process_prompt.modules():
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
        if prompt_hstates is not None:
            # prompt_hstates = self.linear_prompt(prompt_hstates)
            prompt_hstates = self.process_prompt(prompt_hstates)
            # prompt_hstates: b,hidden_dim
            prompt_hstates = prompt_hstates.unsqueeze(1)
        else:
            prompt_hstates = torch.zeros(
                candidate_hstates.shape[0],
                1,
                self.hidden_dim,
                device=candidate_hstates.device,
                dtype=candidate_hstates.dtype,
            ) # b,1,hidden_dim 

        candidate_hstates = self.linear_candidate(candidate_hstates)
        # candidate_hstates: b,k,hidden_dim
        # prompt_hstates: b,1,hidden_dim
        
        x = torch.cat((prompt_hstates,candidate_hstates),dim=1) # b,k+1,hidden_dim
        for layer in self.layers:
            x = x + layer(x)
        # b,k+1,hidden_dim
        if self.criterion is None:
            # Pointwise regression should use an explicit head instead of picking one hidden channel.
            return self.projection(x[:,1:,:]).squeeze(-1)  # b,k
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
    



class DiffusionRanker_CrossAttention(nn.Module):
    def __init__(
        self,
        candidate_hidden_dim:int,
        prompt_hidden_dim:int,
        hidden_dim:int,
        num_heads:int,
        num_layers:int,
        dropout:float,
        half:bool=True,
    ):
        super(DiffusionRanker_CrossAttention,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.input_prompt_norm = nn.LayerNorm(prompt_hidden_dim)
        self.input_candidate_norm = nn.LayerNorm(candidate_hidden_dim)
        self.prompt_hidden_norm = nn.LayerNorm(hidden_dim)
        self.candidate_hidden_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        # self.linear_prompt = nn.Linear(prompt_hidden_dim,hidden_dim)
        self.process_prompt = nn.Sequential(
            nn.Linear(prompt_hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.process_candidate = nn.Sequential(
            nn.Linear(candidate_hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
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

    def _init_weights(self):
        for seq in (self.process_candidate, self.process_prompt):
            for module in seq.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, candidate_hstates, prompt_hstates):
        # candidate_hstates: b,k,candidate_hidden_dim
        # prompt_hstates: b,prompt_hidden_dim
        if prompt_hstates is not None:
            prompt_hstates = self.input_prompt_norm(prompt_hstates)
            prompt_hstates = self.process_prompt(prompt_hstates)
            prompt_hstates = self.prompt_hidden_norm(prompt_hstates)
            prompt_hstates = prompt_hstates.unsqueeze(1)  # b,1,hidden_dim
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
        # candidate_hstates: b,k,hidden_dim
        # prompt_hstates: b,1,hidden_dim

        score_ls = []
        for k in range(candidate_hstates.shape[1]):
            ck = candidate_hstates[:, k : k + 1, :]  # b,1,hidden_dim
            # K/V 至少 2 个 token，否则 seq_len=1 时 softmax 恒为 1，query 与 key 无法形成有效的 prompt–候选交互
            kv = torch.cat([prompt_hstates, ck], dim=1)  # b,2,hidden_dim
            x = self.query_norm(ck)
            kv = self.kv_norm(kv)
            for attn, norm in zip(self.attn_layers, self.attn_norms):
                attn_out, _ = attn(query=x, key=kv, value=kv)
                x = norm(x + attn_out)
            score = self.score_head(x.squeeze(1))  # b,1
            score_ls.append(score)

        score_ls = torch.stack(score_ls, dim=1)  # b,k,1
        return score_ls.squeeze(-1)  # b,k
            
    def count_params(self):
        return sum(
            p.numel() for p in self.parameters()
        )


class DiffusionRanker_SingleClass(nn.Module):
    def __init__(
        self,
        input_dim_candidate:int,
        hidden_dim:int,
        num_heads:int,
        dropout:float,
        num_layers:int,
        half:bool=True,
    ):
        super(DiffusionRanker_SingleClass,self).__init__()
        self.input_dim_candidate = input_dim_candidate
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.half = half
        
        self.linear_candidate = nn.Linear(input_dim_candidate,hidden_dim)
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
        for module in self.output_weights.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,candidate_hstates,prompt_hstates):
        # candidate_hstates: b,k,d_candidate
        # prompt_hstates: 1,d_prompt
        pass
    
    def count_params(self):
        return sum(
            p.numel() for p in self.parameters()
        )


class DiffusionProb_SingleClass(nn.Module):
    def __init__(
        self,
        hs_dim:int,
        hidden_dim:int,
        num_layers:int,
        dropout:float,
        half:bool=True,
    ):
        super(DiffusionProb_SingleClass,self).__init__()
        self.hs_dim = hs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.half = half
        
        self.input_norm = nn.LayerNorm(hs_dim)
        self.linear_candidate = nn.Linear(hs_dim,hidden_dim)
        self.layers = nn.ModuleList(
            [
                MLPBlock(hidden_dim,dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        self.reward_head = nn.Linear(hidden_dim,1)
        nn.init.xavier_uniform_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

        self._init_weights()
        if half:
            self.to(torch.bfloat16)
    
    def _init_weights(self):
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module,nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            for module in self.linear_candidate.modules():
                if isinstance(module,nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            
    
    def forward(self,hs):
        x = self.input_norm(hs)
        x = self.linear_candidate(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.output_norm(x)
        x = self.reward_head(x)
        return x



class MLPBlock(nn.Module):
    def __init__(self,hidden_dim:int,dropout:float):
        super(MLPBlock,self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self,x:Tensor)->Tensor:
        return self.layers(x)
    
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
    hidden_dim=128,
    output_dim=1,
    num_layers=8,
    dropout=0.1,
    half=True,
    criterion='cos_t1.0',
)


cross_attention_ranker = DiffusionRanker_CrossAttention(
    candidate_hidden_dim=1152,
    prompt_hidden_dim=3804,
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

   print('--------------------------------')
   print('cross_attention_ranker')
   print(cross_attention_ranker.count_params())
   dtype = next(cross_attention_ranker.parameters()).dtype
   candidate_hstates = torch.randn(4, 16, 1152, dtype=dtype)
   prompt_hstates = torch.randn(4, 3804, dtype=dtype)
   logits = cross_attention_ranker.forward(candidate_hstates, prompt_hstates)
   print(logits)
   print(logits.shape) # 4,16
   prob = F.softmax(logits,dim=-1)
   print(prob)
   best_idx = torch.argmax(prob,dim=-1)
   print(best_idx)

