import torch
from pathlib import Path

test_path = Path('/root/autodl-tmp/repr_candidate_deno10_cfg1.0/mean/step_ratio0_layer_ratio0.6.pt')

a = torch.load(test_path,map_location='cpu')
t = a['tensor']

print(a)
print(t.shape) # 800000,1152



train_path = Path('/root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/mean/step_ratio0_layer_ratio0.037037.pt')
train_a = torch.load(train_path,map_location='cpu')
train_t = train_a['tensor']
print(train_t.shape) # 800000,1152






