from dit_rollout_mc_is_single import _reward_inception_confidence, _build_inception_logits_extractor
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import dit_loader

from pathlib import Path

device = torch.device('cuda')

pipe = dit_loader(model_name='dit-xl-2-256',device=device)











