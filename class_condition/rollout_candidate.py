import torch
import json     
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import *





@torch.inference_mode()
def rollout_all_canddiates(args,pipe):
    cfg = args.cfg
    batch_size = args.batch_size
    num_candidates = args.num_candidates
    seed = args.seed
    num_steps = args.num_steps
    device = set_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(range(num_candidates),desc="Rollout candidates"):
        save_dir = output_dir / f"candidate_{seed}"
        rollout_single_canddiate(pipe,seed,save_dir,cfg,batch_size,num_steps,device)
        save_metrics(save_dir,device)
    return

@torch.inference_mode()
def rollout_single_canddiate(pipe,seed,save_dir,cfg,batch_size,num_steps,device):
    save_dir.mkdir(parents=True, exist_ok=True)
    class_ls = []
    for idx in range(1000):
        for _ in range(50):
            class_ls.append(idx)
    num_batches = (50000 + batch_size - 1) // batch_size
    count = 0
    for batch_idx in tqdm(range(num_batches),desc=f"Rollout candidate {seed}"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, 50000)

        batch_class_ls = torch.tensor(class_ls[batch_start:batch_end],device=device)
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
        out = pipe(
            class_labels=batch_class_ls,
            num_inference_steps=num_steps,
            guidance_scale=cfg,
            generator=gen,
        ).images
        
        for img, class_idx in zip(out, class_ls[batch_start:batch_end]):
            count += 1
            img_name = f'{count}_{class_idx}.png'
            img.save(save_dir / img_name)
    assert count == 50000


def save_metrics(save_dir,device):
    save_dir = Path(save_dir)
    clip_model_dir = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")
    clip_batch_size = 256

    metrics = compute_fid_is_clipscore(save_dir,device=device,clip_model_dir=clip_model_dir,clip_batch_size=clip_batch_size)
    save_path = save_dir / 'metrics.json'
    with open(save_path, 'w') as f:
        json.dump(metrics, f)
    

    
    




def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="dit-xl-2-256")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/rollout_candidate_step10_cfg1")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = set_device()
    pipe = model_loader(model_name=args.model_name,device=device,dtype='fp16')
    rollout_all_canddiates(args,pipe)


if __name__ == "__main__":
    main()
    
    









