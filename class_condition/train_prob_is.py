import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
import wandb
import argparse
import swanlab
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
from pathlib import Path
import json
import time 
import yaml
import copy
import gc
import math

import os
import sys
from typing import Optional, Dict, Any, List, Tuple, Union
from torchvision.transform import Compose, Resize, CenterCrop





INCEPTION_WEIGHTS_PATH = (
    "/root/diffusion_ranker/third_party/torch_fidelity_weights/"
    "weights-inception-2015-12-05-6726825d.pth"
)







class ProbIS(nn.Module):
    def __init__(
        self,
        input_dim:int,
        hidden_dim:int,
        num_layers:int,
        dropout_prob:float,
        half:bool=True,
    ):
        super(ProbIS, self).__init__()
        self.half = half
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_proj = nn.Linear(input_dim,hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim,4*hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.LayerNorm(4*hidden_dim),
                nn.Linear(4*hidden_dim,hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
            ]
            for _ in range(num_layers)
        )
        self.reward_head = nn.Linear(hidden_dim,1)
        self._init_weights()
        if half:
            self.to(torch.float16)
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        if self.reward_head.bias is not None:
            nn.init.zeros_(self.reward_head.bias)
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.reward_head(x)
    
    def count_param(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HiddenRewardDataset(Dataset):
    def __init__(
        self,
        hiddens:torch.Tensor,
        rewards:torch.Tensor,
    ):
        
        self.hiddens = hiddens
        self.rewards = rewards.view(-1,1)

        # check
        assert self.rewards.shape[0] == self.hiddens.shape[0]
        assert self.rewards.shape[1] == 1
        # check end

    def __len__(self) -> int:
        return int(self.hiddens.shape[0])
    
    def __getitem__(self, idx:int):
        return self.hiddens[idx], self.rewards[idx]
        # return list of (hidden, reward)
    



def collate(batch:List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_hiddens = torch.stack(
        [
            item[0] for item in batch
        ]
    )
    batch_rewards = torch.stack(
        [
            item[1] for item in batch
        ]
    )

    return batch_hiddens, batch_rewards



def set_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout_prob", type=float, default=0.1)
    return p.parse_args()


class Trainer:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.build_dataloader()
        self.build_model()
        self.build_optimzer()
        self.build_scheduler()
        self.build_loss_fn()

    
    def build_dataloader(self,):
        data_path = Path(self.config.rollout_path)
        assert data_path.exists()

        data = torch.load(data_path)
        self.hiddens = data['hiddens']
        self.rewards = data['rewards']

        self.rewards_mean = self.rewards.mean()
        self.rewards_std = self.rewards.std()
        self.rewards = (self.rewards - self.rewards_mean) / (self.rewards_std + 1e-6)
        self.rewards = self.rewards.view(-1,1)

        assert self.hiddens.shape[0] == self.rewards.shape[0]

        train_size = round(self.config.train_ratio * len(self.hiddens))
        val_size = len(self.hiddens) - train_size

        self.dataset = HiddenRewardDataset(
            hiddens=self.hiddens,
            rewards=self.rewards,
        )

        self.gen = torch.Generator().manual_seed(self.config.seed)

        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=self.gen
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=collate,
            generator=self.gen,
            num_workers=self.config.get('num_workers',4)
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=self.config.get('num_workers',4),
        )

    def build_model(self,):
        self.model = ProbIS(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout_prob=self.config.dropout_prob,
            half=self.config.half,
        )
        print(f"Model built with {self.model.count_param()} parameters")

        self.trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]

    def build_optimzer(self,):
        self.optim_name = self.config.optim_name
        # assert self.optim_name in [
        #     'adamw',
        #     'sgd'
        # ]
        if self.optim_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.trainable_params,
                lr=self.config.lr
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1,self.config.beta2),
                eps=self.config.eps,
            )
        elif self.optim_name == 'sgd':
            self.optimzier = optim.SGD(
                self.trainable_params,
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer {self.optim_name} not implemented")
        print(f"Optimizer built with {self.optim_name}")
    

    def build_scheduler(self,):
        self.scheduler_name = self.config.scheduler_name
        if self.scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.T_max,
                eta_min=self.config.eta_min,
            )
        elif self.scheduler_name == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.config.start_factor,
                end_factor=self.config.end_factor,
                total_iters=self.config.total_iters,
            )
        else:
            raise NotImplementedError(f"Scheduler {self.scheduler_name} not implemented")
        print(f"Scheduler built with {self.scheduler_name}")

    def build_loss_fn(self):
        # only mse loss
        self.loss_fn = nn.MSELoss()

    
    def train_epoch(self,):
        self.model.train()
        epoch_loss = 0.0
        num_batchs = math.ceil(len(self.train_loader)/ self.config.train_batch_size)
        for batch_idx, (batch_hiddens, batch_rewards) in enumerate(self.train_loader):
            batch_hiddens = batch_hiddens.to(self.device)
            batch_rewards = batch_rewards.to(self.device).view(-1,1)
            # predict and compute loss
            batch_preds = self.model(batch_hiddens)
            loss = self.loss_fn(batch_preds, batch_rewards)
            epoch_loss += loss.item().float()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return epoch_loss / num_batchs


    def train(self,):
        pass
    
    def val_epoch(self,):
        self.model.eval()
        val_loss = 0.0
        num_batchs = math.ceil(len(self.val_loader) / self.config.train_batch_size)
        for batch_idx, (batch_hiddens, batch_rewards) in enumerate(self.val_loader):
            batch_hiddens = batch_hiddens.to(self.device)
            batch_rewards = batch_rewards.to(self.device).view(-1,1)
            # predict and compute loss
            batch_preds = self.model(batch_hiddens)
            loss = self.loss_fn(batch_preds, batch_rewards)
            val_loss += loss.item().float()
        return val_loss / num_batchs






def rollout_training_data(

):
    pass
    


def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
    # 把时间步 t 统一成模型可用的 batch 形状时间向量: (batch_size,)
    if not torch.is_tensor(t):
        is_mps = latent_model_input.device.type == "mps"
        is_npu = latent_model_input.device.type == "npu"
        if isinstance(t, float):
            dt = torch.float32 if (is_mps or is_npu) else torch.float64
        else:
            dt = torch.int32 if (is_mps or is_npu) else torch.int64
        timesteps = torch.tensor([t], dtype=dt, device=latent_model_input.device)
    elif len(t.shape) == 0:
        timesteps = t[None].to(latent_model_input.device)
    else:
        timesteps = t
    return timesteps.expand(latent_model_input.shape[0])



def _decode_dit_latents_to_pil(pipe, x: torch.Tensor, guidance_scale: float) -> list[Image.Image]:
    if guidance_scale > 1:
        latents, _ = x.chunk(2, dim=0)
    else:
        latents = x
    latents = 1 / pipe.vae.config.scaling_factor * latents
    samples = pipe.vae.decode(latents).sample
    samples = (samples / 2 + 0.5).clamp(0, 1)

    out_images: list[Image.Image] = []
    for bi in range(samples.shape[0]):
        arr = samples[bi].detach().cpu().permute(1, 2, 0).float().numpy() # (C, H, W) -> (H, W, C)
        out_images.append(Image.fromarray((arr * 255.0).round().clip(0, 255).astype("uint8")))
    return out_images


def _build_inception_logits_extractor(device: torch.device):
    from torch_fidelity.utils import create_feature_extractor

    return create_feature_extractor(
        "inception-v3-compat",
        ["logits_unbiased"],
        cuda=(device.type == "cuda"),
        verbose=False,
        feature_extractor_weights_path=INCEPTION_WEIGHTS_PATH,
    )


@torch.inference_mode()
def _reward_inception_confidence(images: list[Image.Image], feat_extractor, device: torch.device) -> torch.Tensor:
    from torch_fidelity.datasets import TransformPILtoRGBTensor

    transform = Compose(
        [
            Resize(256),
            CenterCrop(256),
            TransformPILtoRGBTensor()
        ]
    )

    # images: list[PIL.Image], length = B
    tensors = torch.stack([transform(img) for img in images], dim=0).to(
        device, non_blocking=(device.type == "cuda")
    )  # (B, 3, 256, 256), uint8 on device

    logits = feat_extractor(tensors)  # tuple/list with first item shaped (B, dim)
    logits = logits[0]  # (B, dim)
    probs = F.softmax(logits.float(), dim=-1)  # (B, dim)
    # reward per sample: max class confidence -> (B,)
    return probs.max(dim=-1).values.view(-1,1).detach().cpu().to(torch.float32)


@torch.inference_mode()
def mc_for_estimate_reward(
    pipe,
    x_start: torch.Tensor,
    start_step_idx: int,
    timesteps: torch.Tensor,
    scheduler_snapshot,
    class_labels_input: torch.Tensor,
    latent_channels: int,
    guidance_scale: float,
    feat_extractor,
    device: torch.device,
    seed_base: int,
    mc: int = 5,
) -> torch.Tensor:
    """
    Estimate reward from a target-step latent by MC continuation rollout.
    Returns per-sample reward mean with shape (B,).
    """

    if start_step_idx >= len(timesteps):
        images = _decode_dit_latents_to_pil(pipe, x_start, guidance_scale)
        return _reward_inception_confidence(images, feat_extractor, device)

    rewards_ls: list[torch.Tensor] = []
    for i in range(int(mc)):
        scheduler = copy.deepcopy(scheduler_snapshot)
        x = x_start.detach().clone()
        gen = torch.Generator(device=device).manual_seed(int(seed_base) + i)

        for st_i in range(int(start_step_idx), int(len(timesteps))):
            t = timesteps[st_i]
            if guidance_scale > 1:
                half = x[: len(x) // 2]
                x = torch.cat([half, half], dim=0)

            x_in = scheduler.scale_model_input(x, t)
            timesteps_b = _broadcast_timesteps(t, x_in)
            noise_pred = pipe.transformer(
                x_in,
                timestep=timesteps_b,
                class_labels=class_labels_input,
            ).sample

            if guidance_scale > 1:
                eps, tail = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                noise_pred = torch.cat([eps, tail], dim=1)

            if pipe.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            x = scheduler.step(model_output, t, x, generator=gen).prev_sample
            del noise_pred, model_output, x_in

        images = _decode_dit_latents_to_pil(pipe, x, guidance_scale)
        reward_t = _reward_inception_confidence(images, feat_extractor, device)  # (B,)
        rewards_ls.append(reward_t)
        del x, images, reward_t
        gc.collect()
        torch.cuda.empty_cache()

    return torch.stack(rewards_ls, dim=0).mean(dim=0).to(torch.float32)  # (B,)





def main():
    pass






if __name__ == "__main__":
    main()
    













