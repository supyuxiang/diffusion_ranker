import torch

repr_path = "/root/autodl-tmp/repr_candidate_deno10_cfg1.0/last/s0_r0.1.pt"
repr = torch.load(repr_path)
print(repr.keys())
print(repr["tensor"].shape)
print(repr["repr_key"])
print(repr["step_ratio"])
print(repr["layer_ratio"])