from dit_rollout_mc_is_single import _reward_inception_confidence, _build_inception_logits_extractor
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor
import torch



device = torch.device('cuda')
feat_extractor = _build_inception_logits_extractor(device)
img = Image.open('/root/27548.png')

reward = _reward_inception_confidence([img], feat_extractor, device)
print(reward)








