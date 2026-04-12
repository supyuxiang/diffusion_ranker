import torch
import os
import sys
import torch.nn as nn

from utils import *


@torch.inference_mode()
def dit_rollout_with_single_step(
    model_name,
    cfg,
    deno_step,
    batch_size,
    num_gen:int,
):
    device = set_device()
    dit = model_loader(model_name,device)
    dit.eval()

    from utils import dit_rollout_pipeline















