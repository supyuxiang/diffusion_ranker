# pormpt hs = None

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import hydra
import os
import numpy as np
import sys

class RankerDataset_Shuffle(Dataset):
    def __init__(
        self,
        
    ):
        self.load_data()
    
    def __len__(self):
        pass

    def __getitem__(self,idx):
        pass


class ListwiseRanker(nn.Module):
    pass



class PointwiseRanker(nn.Module):
    pass




class Trainer:
    pass



def main():
    pass
    

if __name__ == "__main__":
    main()


