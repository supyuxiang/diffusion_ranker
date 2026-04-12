import torch
from torch.utils.data import dataset, DataLoader
import os
import sys
import json
from pathlib import Path

from utils import *
from model import *





class RankerDataset(dataset):
    pass



def train_epoch():
    pass


def train():
    pass


def eval_epoch():
    pass

def visulize():
    pass


def set_argparse():
    default_settings = Constants_default
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='dit-xl-2-256')
    p.add_argument('--cfg', type=float, default=1.0)
    p.add_argument('--num_layers', type=int, default=24)
    p.add_argument('--proj_dim', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)

    args = p.parse_args()
    for key, value in default_settings.items():
        if key not in args:
            args[key] = value

    return p.parse_args()

    

def main():
    args = set_argparse()
    device = set_device()
    set_seed(args.seed)
    pass




















