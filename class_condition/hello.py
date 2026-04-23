from dit_rollout_mc_is_single import _reward_inception_confidence, _build_inception_logits_extractor
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor
import torch
from diffusers import DiTPipeline


from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import dit_loader


# device = torch.device('cuda')
# dtype = torch.float16


# pipe = dit_loader(model_name='dit-xl-2-256',device=device,dtype=dtype)
# pipe.set_progress_bar_config(disable=True)

# pipe.scheduler.set_timesteps(10)
# timesteps = pipe.scheduler.timesteps


# with torch.inference_mode():
#     gen = torch.Generator(device=device).manual_seed(42)
#     for t in timesteps:
#         latents = torch.randn(
#             (1, 4, 64, 64),
#             dtype=dtype,
#             device=device,
#             generator=gen,
#         )
#         # cfg=1 version
#         x = latents
#         pipe.scheduler.scale_model_input(x,t)
#         timesteps_b = _broadcast_timesteps(t, x)
#         noise_pred = pipe.transformer(
#             x,
#             timestep=timesteps_b,
#             class_labels=class_labels_input,
#         ).sample

import torch

a = torch.load('dit_imagenet_hiddens_and_is_rewards_s1_6_9_l16_class153_s6_l16.pt',weights_only=True,map_location='cpu')


breakpoint()
    












