# 探测diffusion model的hiddens是否可以区分class name信息

from pathlib import Path
import sys
import json
import torch
import torch.nn.functional as F

from utils import labels_1k

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

