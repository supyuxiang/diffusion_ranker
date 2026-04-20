import torch
from pathlib import Path

# ------------------------------- I love Peking University -------------------------------


path = Path('/root/autodl-tmp/dit_imagenet_hiddens_and_is_rewards_s1_6_9_l16_per64_shards/reward_s6_b00499.pt')

reward = torch.load(path,map_location='cpu',weights_only=False)
print(reward.shape)


print(reward)











