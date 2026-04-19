import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter






############## hidden shape (sinlge) : [256,16,16] ##############


class Ranker_MLP(nn.Module):
    def __init__(self, num_layers:int, hidden_dim:int, dropout:float=0.1,input_channels:int=256):
        super(Ranker_MLP,self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.process_candidate = nn.Sequential( # b,c,h,w -> b,hidden_dim
                nn.Conv2d(input_channels,16,kernel_size=2,stride=2),# b,256,16,16 -> b,16,8,8
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Flatten(start_dim=1), # b,16,8*8 -> b,16*8*8
                nn.Linear(16*8*8,hidden_dim), # b,16*8*8 -> b,hidden_dim
        )

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

        self.score_head = nn.Linear(hidden_dim,1,bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if isinstance(m.weight, UninitializedParameter):
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None and not isinstance(m.bias, UninitializedParameter):
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.score_head.weight)

    
    def forward(self, cand_hstates:torch.Tensor) -> torch.Tensor:
        B,K,C,H,W = cand_hstates.shape
        single_cand_hs_ls=[]
        for k in range(K):
            cand_k = cand_hstates[:,k,:,:,:] # B,C,H,W
            cand_k = self.process_candidate(cand_k) # B,hidden_dim
            single_cand_hs_ls.append(cand_k)

        cand_hstates = torch.stack(single_cand_hs_ls,dim=1) # B,K,hidden_dim
        for layer in self.layers:
            cand_hstates = cand_hstates + layer(cand_hstates) # B,K,hidden_dim
        score = self.score_head(cand_hstates).squeeze(-1) # B,K
        return score
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
        


class Probe_MLP(nn.Module):
    def __init__(self,num_layers:int,hidden_dim,input_channels:int=256,dropout:float=0.1):
        super(Probe_MLP,self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        self.dropout = dropout

        self.process_candidate = nn.Sequential(
            nn.Conv2d(input_channels,16,kernel_size=2,stride=2),# b,256,16,16 -> b,16,8,8
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Flatten(start_dim=1),# b,16,8*8 -> b,16*8*8
            nn.Linear(16*8*8,hidden_dim),# b,16*8*8 -> b,hidden_dim
        )

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.score_head = nn.Linear(hidden_dim,1,bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if isinstance(m.weight, UninitializedParameter):
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None and not isinstance(m.bias, UninitializedParameter):
                    nn.init.zeros_(m.bias)




    def forward(self, cand_hstate:torch.Tensor) -> torch.Tensor:
        cand_hstate = self.process_candidate(cand_hstate) # B,hidden_dim
        for layer in self.layers:
            cand_hstate = cand_hstate + layer(cand_hstate)
        score = self.score_head(cand_hstate).squeeze(-1) # B,1
        return score
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class Ranker_Transformer(nn.Module):
    def __init__(self, num_layers:int, hidden_dim:int, num_heads:int, dropout:float=0.1,input_channels:int=256):
        super(Ranker_Transformer,self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.process_candidate = nn.Sequential( # b,c,h,w -> b,hidden_dim
                nn.Conv2d(input_channels,16,kernel_size=2,stride=2),# b,256,16,16 -> b,16,8,8
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Flatten(start_dim=1), # b,16,8*8 -> b,16*8*8
                nn.Linear(16*8*8,hidden_dim), # b,16*8*8 -> b,hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if isinstance(m.weight, UninitializedParameter):
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None and not isinstance(m.bias, UninitializedParameter):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, cand_hstates:torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = cand_hstates.shape
        cand_hstates = cand_hstates.reshape(B * K, C, H, W) # BK,C,H,W
        cand_hstates = self.process_candidate(cand_hstates) # BK,hidden_dim
        cand_hstates = cand_hstates.reshape(B, K, self.hidden_dim) # B,K,hidden_dim
        cand_hstates = self.encoder(cand_hstates) # B,K,hidden_dim
        cand_hstates = self.final_norm(cand_hstates)
        score = self.score_head(cand_hstates).squeeze(-1) # B,K
        return score

    def count_params(self):
        return sum(p.numel() for p in self.parameters())






























        

def test():
    model = Ranker_MLP(num_layers=2, hidden_dim=128)
    cand_hstates = torch.randn(128,10,256,16,16)
    score = model(cand_hstates)
    print(score.shape)
    print(model.count_params())
    ######
    print("#########################")
    model = Probe_MLP(num_layers=2, hidden_dim=128)
    cand_hstate = torch.randn(128,256,16,16)
    score = model(cand_hstate)
    print(score.shape)
    print(model.count_params())
    ######
    print("#########################")
    model = Ranker_Transformer(num_layers=2, hidden_dim=128, num_heads=8)
    cand_hstates = torch.randn(128,10,256,16,16)
    score = model(cand_hstates)
    print(score.shape)
    print(model.count_params())
    ######



if __name__ == "__main__":
    test()
            
