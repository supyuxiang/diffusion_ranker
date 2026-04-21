import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from class_condition.dataset import RankerDataset, training_samples_per_class
from class_condition.ranker import DiffusionRanker_Transformer
from utils import set_device, set_seed


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
        self.shuffle_class = self.config.get('shuffle_class', False)
        ds_kw = dict(
            step_idx=self.config['step_idx'],
            layer_idx=self.config['layer_idx'],
            prompt_type=self.config.get('prompt_type'),
        )

        if not self.shuffle_class:
            self.batch_size = training_samples_per_class
            self.train_dataset = RankerDataset(**ds_kw)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config.get('num_workers', 4),
            )
            self.val_dataset,self.val_dataloader = None,None
        
        else:
            self.batch_size = self.config.get('batch_size', 32)

            self.full_dataset = RankerDataset(shuffle_class=True, **ds_kw)
            
            split_ratio = self.config.get('split_ratio', [0.9, 0.1])
            self.train_dataset, self.val_dataset = random_split(self.full_dataset, split_ratio)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config.get('num_workers', 4),
            )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
            )




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
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
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
    

    def process_bootstrap(self,prompt_hs,candidate_hs,reward,ranker):
        ranker.train()
        w_dtype = next(ranker.parameters()).dtype
        prompt_hs = prompt_hs.to(dtype=w_dtype)
        candidate_hs = candidate_hs.to(dtype=w_dtype)
        student_logits = ranker(candidate_hs, prompt_hs)
        reward = reward.to(device=student_logits.device, dtype=student_logits.dtype).reshape_as(student_logits)
        loss = self.loss_fn(student_logits, reward)
        return loss



    def train_epoch(self):
        self.ranker.train()
        parallel_size = self.config.get('parallel_size', 8)
        num_bootstrap_samples = self.config.get('num_bootstrap_samples', 8)
        assert num_bootstrap_samples * parallel_size >= training_samples_per_class, '一个batch训得不充分，考虑增大parallel_size或num_bootstrap_samples'
        epoch_loss = 0.0

        if self.shuffle_class:
            pbar = tqdm(self.train_dataloader, desc='Training', leave=False)
            for batch_idx, batch in enumerate(pbar):
                prompt_hs, candidate_hs, reward, label = batch
                prompt_hs, candidate_hs, reward, label = prompt_hs.to(self.device), candidate_hs.to(self.device), reward.to(self.device), label.to(self.device)
                
                batch_loss = 0.0
                for bootstrap_idx in range(num_bootstrap_samples):
                    indices = torch.randperm(self.batch_size)[:parallel_size] # 随机采样parallel_size个样本
                    bootstrap_prompts_hs,bootstrap_candidates_hs,bootstrap_reward = prompt_hs[indices], candidate_hs[indices], reward[indices]
                    bootstrap_candidates_hs = bootstrap_candidates_hs.unsqueeze(1)  # (B,1,D) ranker expects (B,K,D)
                    bootstrap_loss = self.process_bootstrap(bootstrap_prompts_hs, bootstrap_candidates_hs, bootstrap_reward, self.ranker)
                    batch_loss += bootstrap_loss
                batch_loss /= num_bootstrap_samples
                pbar.set_postfix(loss=float(batch_loss.detach().item()))
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            epoch_loss += float(batch_loss.detach().item())
            epoch_loss /= len(self.train_dataloader)
            return epoch_loss

        else:
            pbar = tqdm(self.train_dataloader, desc='Training', leave=False)
            for class_idx, class_group in enumerate(pbar):
                batch_prompt_hs, batch_candidate_hs, batch_reward, batch_label = class_group
                batch_prompt_hs, batch_candidate_hs, batch_reward, batch_label = batch_prompt_hs.to(self.device), batch_candidate_hs.to(self.device), batch_reward.to(self.device), batch_label.to(self.device)
                batch_loss = 0.0
                for bootstrap_idx in range(num_bootstrap_samples):
                    indices = torch.randperm(len(batch_prompt_hs))[:parallel_size]
                    bootstrap_prompts_hs,bootstrap_candidates_hs,bootstrap_reward = batch_prompt_hs[indices], batch_candidate_hs[indices], batch_reward[indices]
                    bootstrap_candidates_hs = bootstrap_candidates_hs.unsqueeze(1)  # (B,1,D) ranker expects (B,K,D)
                    bootstrap_loss = self.process_bootstrap(bootstrap_prompts_hs, bootstrap_candidates_hs, bootstrap_reward, self.ranker)
                    batch_loss += bootstrap_loss
                batch_loss /= num_bootstrap_samples 
                pbar.set_postfix(loss=float(batch_loss.detach().item())) 
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            epoch_loss += float(batch_loss.detach().item())
            epoch_loss /= len(self.train_dataloader)
            return epoch_loss



    def train(self):
        best_loss = float('inf')
        for epoch in range(self.config.get('num_epochs', 100)):
            epoch_loss = self.train_epoch()
            self.scheduler.step()
            if epoch % self.config.get('save_interval', 10) == 0:
                save_interval_path = Path(self.config.get('output_dir', '')) / f'ranker_{self.config.get("run_name", "default")}_epoch_{epoch}.pth'
                self.save_ckpt(save_interval_path)
            if epoch % self.config.get('eval_interval', 10) == 0 and self.val_dataloader is not None:
                eval_loss = self.evaluate()
                print('-'*100)
                print(f'Epoch {epoch}, Loss: {epoch_loss}, Eval Loss: {eval_loss}')
                print('-'*100)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_best_path = Path(self.config.get('output_dir', '')) / f'ranker_{self.config.get("run_name", "default")}_best.pth'
                self.save_ckpt(save_best_path)
        print(f'Best Loss: {best_loss}')
            
    

    def evaluate(self):
        eval_loss = 0.0
        self.ranker.eval()
        num_bootstrap_samples = self.config.get('num_bootstrap_samples', 8)
        parallel_size = self.config.get('parallel_size', 8)
        for batch in self.val_dataloader:
            prompt_hs, candidate_hs, reward, label = batch
            prompt_hs, candidate_hs, reward, label = prompt_hs.to(self.device), candidate_hs.to(self.device), reward.to(self.device), label.to(self.device)
            batch_loss = 0.0
            for bootstrap_idx in range(num_bootstrap_samples):
                indices = torch.randperm(len(candidate_hs))[:parallel_size]
                bootstrap_prompts_hs, bootstrap_candidates_hs, bootstrap_reward = (
                    prompt_hs[indices],
                    candidate_hs[indices],
                    reward[indices],
                )
                bootstrap_candidates_hs = bootstrap_candidates_hs.unsqueeze(1)
                bootstrap_loss = self.process_bootstrap(
                    bootstrap_prompts_hs, bootstrap_candidates_hs, bootstrap_reward, self.ranker
                )
                batch_loss += bootstrap_loss
            batch_loss /= num_bootstrap_samples
            with torch.no_grad():
                eval_loss += float(batch_loss.detach().item())
        eval_loss /= len(self.val_dataloader)
        return eval_loss
    

    def save_ckpt(self, ckpt_path:Path):
        save_dir = Path(self.config.get('output_dir', '')) / f'ranker_{self.config.get("run_name", "default")}'
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.ranker.state_dict(), ckpt_path)

    


def _cfg_to_trainer_dict(cfg: DictConfig) -> dict:
    """Hydra DictConfig -> plain dict for Trainer (drops Hydra-reserved keys)."""
    d = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(d, dict)
    d.pop("hydra", None)
    return d


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.get("seed", 42)))
    trainer = Trainer(_cfg_to_trainer_dict(cfg))
    trainer.train()


if __name__ == "__main__":
    main()

















