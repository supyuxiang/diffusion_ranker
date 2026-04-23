import torch
import torch.nn as nn
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from pathlib import Path

def load_dit(dit_path: str|Path, device:str, fp16:bool=True):
    dit_path = str(dit_path)
    pipe = DiTPipeline.from_pretrained(
        dit_path,
        local_files_only=True,
        torch_dtype=torch.float16 if fp16 else torch.float32,
    )
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

def main():
    dit_path = '/data/yxfeng/models/dit_model'
    device='cuda'
    pipe = load_dit(dit_path, device)
    
    print(pipe.transformer.config)

    # generate:
    num_samples = 10
    
    for i in range(num_samples):
        latents = torch.randn(1,4,64,64, device=device)
        timesteps = torch.randint()

if __name__ == '__main__':
    main()