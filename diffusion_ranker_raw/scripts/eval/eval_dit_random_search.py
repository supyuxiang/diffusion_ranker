import torch
import torch.nn as nn
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from tqdm import tqdm
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import argparse

from utils import setup_seed

def load_dit(dit_path:str | Path, device:str, fp16:bool=True):
    '''load DiT model'''
    dit_path = Path(dit_path)
    assert dit_path.exists(), f"DiT 模型未找到: {dit_path}"
    pipe = DiTPipeline.from_pretrained(dit_path, torch_dtype=torch.float16 if fp16 else torch.float32, device=device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(dit_path, subfolder="scheduler")
    return pipe


def rollout_dit(pipe:DiTPipeline, save_dir:Path, num_samples:int, batch_size:int, k:int, device:str, seed:int):
    pass










def main():
    pass


if __name__ == "__main__":
    main()