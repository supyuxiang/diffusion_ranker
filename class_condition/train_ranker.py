import torch
import torch.nn as nn
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm


import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from class_condition.ranker import DiffusionRanker_Transformer
from utils import set_device


# ------------------------------- I love Peking University -------------------------------


class Trainer:
    def __init__(self,config:dict):
        self.config = config
        self.device = set_device()
        self.build_ranker()
        self.build_dataloader()
        self.build_optimizer()
        self.build_scheduler()
        self.build_loss_fn()
    

    def build_ranker(self):
        self.ranker = DiffusionRanker_Transformer(
            input_dim_candidate=self.config['input_dim_candidate'],
            input_dim_prompt=self.config['input_dim_prompt'],
            hidden_dim=self.config['hidden_dim'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout'],
            num_layers=self.config['num_layers'],
            half=self.config['half'],
        )
        self.ranker.to(self.device)
    
    def build_dataloader(self):
        from class_condition.dataset import RankerTrainingDataset
        self.train_dataset = RankerTrainingDataset(
            candidate_dir=self.config['candidate_dir'],
            prompt_type=self.config['prompt_type'],
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
        )
        self.val_dataset = None

    def build_optimizer(self):
        optimizer = self.config.get('optimizer', 'adamw')
        if optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.ranker.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config.get('weight_decay', 0.0001),
                betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999)),
                eps=self.config.get('eps', 1e-8),
            )
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.ranker.parameters(),
                lr=self.config['lr'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 0.0001),
            )
        else:
            raise NotImplementedError(f"Invalid optimizer: {optimizer}")
    
    def build_scheduler(self):
        scheduler = self.config.get('scheduler', 'cosine')
        if scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler(
                self.optimizer,
                T_max=self.config.get('T_max', 100),
                eta_min=self.config.get('eta_min', 0),
                last_epoch=-1,
            )
        elif scheduler == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.get('start_factor', 0.01),
                end_factor=self.config.get('end_factor', 0.1),
                total_iters=self.config.get('total_iters', 100),
            )
        elif scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 10),
                gamma=self.config.get('gamma', 0.1),
            )
        else:
            raise NotImplementedError(f"Invalid scheduler: {scheduler}")

    def build_loss_fn(self):
        from class_condition.utils import RankerLoss
        self.loss_fn = RankerLoss(
            alpha=self.config.get('alpha', 0.8),
            kl_type=self.config.get('kl_type', 'kl'),
        )
    
    def train_epoch(self):
        self.ranker.train()
        epoch_loss = 0.0
        batch_size = self.config.get('batch_size', 128)
        
        for batch
    

    def evaluate(self):
        pass

    def save_ckpt(self):
        save_dir = Path(self.config.get('output_dir', '')) / f'ranker_{self.config.get("run_name", "default")}'
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / 'ranker.pth'
        torch.save(self.ranker.state_dict(), ckpt_path)
    
    @staticmethod
    def load_ckpt(ckpt_path:Path):
        state_dict = torch.load(ckpt_path,map_location='cpu',weights_only=False)
        return state_dict

    


def set_args():
    p = argparse.ArgumentParser()
    return p.parse_args()



def main():
    args = set_args()
    config = OmegaConf.load(args.config_path)
    trainer = Trainer(config)
    trainer.train()





















