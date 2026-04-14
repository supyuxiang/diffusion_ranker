import torch
from pathlib import Path

path = Path('/root/autodl-tmp/repr_candidate_deno10_cfg1.0/mean/step_ratio0_layer_ratio0.6.pt')
a = torch.load(path,map_location='cpu')

breakpoint()