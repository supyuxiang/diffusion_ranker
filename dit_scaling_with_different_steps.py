

'''
example:
CUDA_VISIBLE_DEVICES=0 python /root/diffusion_ranker/dit_scaling_with_different_steps.py \
    --model_name dit-xl-2-256 \
    --cfg 1.0 \
    --batch_size 16 \
    --num_gen 50000 \
    --save_dir /root/autodl-tmp/outputs/dit_scaling_with_different_steps

'''



import torch
import os
import sys
import torch.nn as nn

from utils import *
import json


_DENO_STEP_LS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def set_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dit-xl-2-256')
    parser.add_argument('--cfg', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_gen', type=int, default=50000)
    parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/outputs/dit_scaling_with_different_steps')
    return parser.parse_args()
    

def main():
    args = set_args()
    dit = model_loader(args.model_name,device=set_device())
    dit.eval()

    for deno_step in _DENO_STEP_LS:
        args.deno_step = deno_step
        args.save_dir = Path(args.save_dir) / f'deno_step_{deno_step}'
        args.save_dir.mkdir(parents=True, exist_ok=True)
        dit_rollout_pipeline(dit, args.cfg, args.deno_step, args.batch_size, args.num_gen, Path(args.save_dir))
        metrics = compute_metrics(Path(args.save_dir))
        print(f"Done! Saved to {args.save_dir}")
        print(f"Metrics: {metrics}")
        metrics_save_path = Path(args.save_dir) / f'metrics_deno_step_{deno_step}.json'
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_save_path}")
















