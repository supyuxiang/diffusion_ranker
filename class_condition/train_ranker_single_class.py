import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from class_condition.ranker import DiffusionProb_SingleClass
from utils import set_device, set_seed


map_path = lambda class_idx,s,l : f'/root/autodl-tmp/dit_imagenet_hiddens_and_is_rewards_s1_6_9_l16_class{class_idx}_s{s}_l{l}.pt'


class SingleClassDataset(Dataset):
    def __init__(
        self,
        class_idx:int,
        step_idx:int,
        layer_idx:int=16,
        normalize_rewards: bool = True,
    ):
        super().__init__()
        self.class_idx = int(class_idx)
        self.step_idx = int(step_idx)
        self.layer_idx = int(layer_idx)
        self.target_key = "reward"
        self.data_path = map_path(class_idx,step_idx,layer_idx)
        payload = torch.load(self.data_path, weights_only=True,map_location='cpu')

        self.hiddens = payload["hiddens"].to(torch.float32)
        self.rewards = payload['reward'].to(torch.float32)

        if normalize_rewards:
            self.rewards = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-8)

        self.hidden_dim = int(self.hiddens.shape[1])
        self.num_samples = int(self.hiddens.shape[0])
        assert self.num_samples == self.rewards.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.hiddens[idx], self.rewards[idx]

def collate_fn(batch):
    # list of (hidden, reward)
    batch_hiddens = torch.stack(
        [
            hidden for hidden,_ in batch
        ]
        ,dim=0
    )
    batch_rewards = torch.stack(
        [
            reward for _,reward in batch
        ]
        ,dim=0
    )

    return batch_hiddens,batch_rewards



class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = set_device()
        self.build_dataloader()
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_loss_fn()

    def build_dataloader(self):
        class_idx = self.config.get("class_idx", 153)
        step_idx = self.config.get("step_idx", 6)
        layer_idx = self.config.get("layer_idx", 16)

        self.dataset = SingleClassDataset(
            class_idx=class_idx,
            step_idx=step_idx,
            layer_idx=layer_idx,
            normalize_rewards=bool(self.config.get("normalize_rewards", True)),
        )
        split_ratio = [0.9,0.1]

        n_total = len(self.dataset)
        n_train = int(round(n_total * split_ratio[0]))
        n_train = min(max(n_train, 1), n_total)
        n_val = n_total - n_train

        split_seed = int(self.config.get("seed", 42))
        generator = torch.Generator().manual_seed(split_seed)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [n_train, n_val], generator=generator)

        self.batch_size = int(self.config.get("batch_size", 128))
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(self.config.get("num_workers", 4)),
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.val_dataloader = None
        if n_val > 0:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=int(self.config.get("num_workers", 4)),
                drop_last=False,
                collate_fn=collate_fn,
            )

    def build_model(self):
        self.ranker = DiffusionProb_SingleClass(
            hs_dim=int(self.config.get("input_dim_candidate", self.dataset.hidden_dim)),
            hidden_dim=int(self.config.get("hidden_dim", 256)),
            num_layers=int(self.config.get("num_layers", 4)),
            dropout=float(self.config.get("dropout", 0.1)),
            half=bool(self.config.get("half", True)),
        ).to(self.device)

    def build_optimizer(self):
        optimizer = self.config.get("optimizer", "adamw")
        if optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.ranker.parameters(),
                lr=float(self.config.get("lr", 1e-4)),
                weight_decay=float(self.config.get("weight_decay", 1e-4)),
                betas=(float(self.config.get("beta1", 0.9)), float(self.config.get("beta2", 0.999))),
                eps=float(self.config.get("eps", 1e-8)),
            )
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.ranker.parameters(),
                lr=float(self.config.get("lr", 1e-4)),
                momentum=float(self.config.get("momentum", 0.9)),
                weight_decay=float(self.config.get("weight_decay", 1e-4)),
            )
        else:
            raise NotImplementedError(f"Invalid optimizer: {optimizer}")

    def build_scheduler(self):
        scheduler = self.config.get("scheduler", "cosine")
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(self.config.get("T_max", max(1, int(self.config.get("num_epochs", 100))))),
                eta_min=float(self.config.get("eta_min", 0.0)),
            )
        elif scheduler == "none":
            self.scheduler = None
        else:
            raise NotImplementedError(f"Invalid scheduler: {scheduler}")

    def build_loss_fn(self):
        def loss_fn(batch_pred:torch.Tensor,batch_target:torch.Tensor,alpha=0.8):
            # both are logits, shape :[b,1]
            mse_item = F.mse_loss(batch_pred,batch_target)
            kl_item = F.kl_div(F.log_softmax(batch_pred,dim=1),F.log_softmax(batch_target,dim=1),reduction='batchmean')
            return alpha * mse_item + (1-alpha) * kl_item
        self.loss_fn = loss_fn


    def bootstrap_from_batch(self,batch_hs:torch.Tensor,batch_targets:torch.Tensor,parallel_size:int,num_bootstrap:int):
        n_this_batch=batch_hs.shape[0]
        bootstrap_hiddens = []
        bootstrap_targets = []
        for _ in range(num_bootstrap):
            indices = torch.randperm(n_this_batch,device=self.device)[:parallel_size]
            bootstrap_hiddens.append(batch_hs[indices])
            bootstrap_targets.append(batch_targets[indices])
        return bootstrap_hiddens,bootstrap_targets


    def process_batch(self, batch_hs, batch_target):
        w_dtype = next(self.ranker.parameters()).dtype
        batch_hs = batch_hs.to(self.device, dtype=w_dtype)
        batch_target = batch_target.to(self.device, dtype=w_dtype).reshape(-1, 1)
        n_this_batch = batch_hs.shape[0]

        num_bootstrap = int(self.config.get("num_bootstrap", 8))
        parallel_size = int(self.config.get("parallel_size", 8))

        # bootstrap
        bootstrap_hiddens,bootstrap_targets = self.bootstrap_from_batch(batch_hs,batch_target,parallel_size,num_bootstrap)

        batch_loss_sum = torch.zeros((), device=self.device, dtype=w_dtype)
        batch_mse_sum = 0.0
        batch_kl_sum = 0.0

        for bootstrap_hidden,bootstrap_target in zip(bootstrap_hiddens,bootstrap_targets):
            pred = self.ranker(bootstrap_hidden)
            loss = self.loss_fn(pred, bootstrap_target)
            mse = F.mse_loss(pred, bootstrap_target).detach()
            kl = F.kl_div(F.log_softmax(pred,dim=1),F.log_softmax(bootstrap_target,dim=1),reduction='batchmean').detach()
            batch_loss_sum = batch_loss_sum + loss
            batch_mse_sum += float(mse.item())
            batch_kl_sum += float(kl.item())
        return batch_loss_sum / num_bootstrap, batch_mse_sum / num_bootstrap, batch_kl_sum / num_bootstrap

    def train_epoch(self):
        self.ranker.train()
        loss_sum = 0.0
        mse_sum = 0.0
        kl_sum = 0.0
        batch_cnt = 0

        pbar = tqdm(self.train_dataloader, desc="Training", leave=False)
        for batch_hs, batch_target in pbar:
            loss, mse, kl = self.process_batch(batch_hs, batch_target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            loss_sum += float(loss.item())
            mse_sum += float(mse.item())
            kl_sum += float(kl.item())
            batch_cnt += 1

        if batch_cnt == 0:
            return float("nan"), float("nan"), float("nan")
        return loss_sum / batch_cnt, mse_sum / batch_cnt, kl_sum / batch_cnt

    @torch.inference_mode()
    def evaluate(self):
        self.ranker.eval()
        loss_sum = 0.0
        mse_sum = 0.0
        kl_sum = 0.0
        batch_cnt = 0
        pbar = tqdm(self.val_dataloader, desc="Evaluating", leave=False)
        for batch_hs, batch_target in pbar:
            loss, mse, kl = self.process_batch(batch_hs, batch_target)
            loss_sum += float(loss.item())
            mse_sum += float(mse.item())   
            kl_sum += float(kl.item())
            batch_cnt += 1
        return loss_sum / batch_cnt, mse_sum / batch_cnt, kl_sum / batch_cnt

    def save_ckpt(self, ckpt_path: Path, epoch: int, train_loss: float, val_loss: float, val_mse: float, val_kl: float):
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.ranker.state_dict(),
                "config": self.config,
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mse": float(val_mse),
                "val_kl": float(val_kl),
                "data_path": str(self.dataset.data_path),
                "target_key": self.dataset.target_key,
                "num_samples": self.dataset.num_samples,
                "hidden_dim": self.dataset.hidden_dim,
                "class_idx": self.dataset.class_idx,
                "step_idx": self.dataset.step_idx,
                "layer_idx": self.dataset.layer_idx,
            },
            ckpt_path,
        )

    def train(self):
        best_loss = float("inf")
        best_mse = float("nan")
        best_kl = float("nan")
        num_epochs = int(self.config.get("num_epochs", 100))
        save_interval = int(self.config.get("save_interval", 10))
        eval_interval = int(self.config.get("eval_interval", 1))
        run_name = self.config.get("run_name", "single_class_regressor")
        output_dir = Path(self.config.get("output_dir", "/root/autodl-tmp/ranker_ckpts")).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            train_loss, train_mse, train_kl = self.train_epoch()
            val_loss, val_mse, val_kl = (float("nan"), float("nan"), float("nan"))
            if self.scheduler is not None:
                self.scheduler.step()
            if epoch % eval_interval == 0:
                val_loss, val_mse, val_kl = self.evaluate()

            lr_now = self.optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch} | train_loss: {train_loss:.6f} | train_mse: {train_mse:.6f} | "
                f"train_kl: {train_kl:.6f} | val_loss: {val_loss:.6f} | val_mse: {val_mse:.6f} | "
                f"val_kl: {val_kl:.6f} | lr: {lr_now:.3e}"
            )

            if epoch % save_interval == 0:
                save_interval_path = output_dir / f"{run_name}_epoch_{epoch}.pth"
                self.save_ckpt(save_interval_path, epoch, train_loss, val_loss,val_mse,val_kl)

            if val_loss < best_loss:
                best_loss = val_loss
                best_mse = val_mse
                best_kl = val_kl
                save_best_path = output_dir / f"{run_name}_best.pth"
                self.save_ckpt(save_best_path, epoch, train_loss, val_loss,val_mse,val_kl)

        print(f"Best Loss: {best_loss:.6f}")
        print(f"Best MSE: {best_mse:.6f}")
        print(f"Best KL: {best_kl:.6f}")


def _cfg_to_trainer_dict(cfg: DictConfig) -> dict:
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
