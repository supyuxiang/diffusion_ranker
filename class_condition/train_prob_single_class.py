import copy
import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from class_condition.ranker import DiffusionProb_SingleClass
from utils import set_device, set_seed


map_path = lambda class_idx, s, l: (
    f"/root/autodl-tmp/dit_imagenet_hiddens_and_is_rewards_s1_6_9_l16_class{class_idx}_s{s}_l{l}.pt"
)


class SingleClassDataset(Dataset):
    def __init__(
        self,
        class_idx: int,
        step_idx: int,
        layer_idx: int = 16,
        normalize_rewards: bool = True,
    ):
        super().__init__()
        self.class_idx = int(class_idx)
        self.step_idx = int(step_idx)
        self.layer_idx = int(layer_idx)
        self.target_key = "reward"
        self.data_path = map_path(class_idx, step_idx, layer_idx)
        payload = torch.load(self.data_path, weights_only=True, map_location="cpu")

        self.hiddens = payload["hiddens"].to(torch.float32)
        self.rewards = payload["reward"].to(torch.float32)
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
    batch_hiddens = torch.stack([h for h, _ in batch], dim=0)
    batch_rewards = torch.stack([r for _, r in batch], dim=0)
    return batch_hiddens, batch_rewards


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = set_device()
        self.build_dataloader()
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.best_ckpt_path = None

    def build_dataloader(self):
        class_idx = int(self.config.get("class_idx", 153))
        step_idx = int(self.config.get("step_idx", 6))
        layer_idx = int(self.config.get("layer_idx", 16))
        split_ratio = self.config.get("split_ratio", [0.9, 0.1])

        self.dataset = SingleClassDataset(
            class_idx=class_idx,
            step_idx=step_idx,
            layer_idx=layer_idx,
            normalize_rewards=bool(self.config.get("normalize_rewards", True)),
        )

        n_total = len(self.dataset)
        n_train = int(round(n_total * float(split_ratio[0])))
        n_train = min(max(n_train, 1), n_total)
        n_val = n_total - n_train

        split_seed = int(self.config.get("seed", 42))
        generator = torch.Generator().manual_seed(split_seed)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [n_train, n_val], generator=generator)

        self.batch_size = int(self.config.get("batch_size", 128))
        num_workers = int(self.config.get("num_workers", 4))
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.val_dataloader = None
        if n_val > 0:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=collate_fn,
            )

    def build_model(self):
        self.model = DiffusionProb_SingleClass(
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
                self.model.parameters(),
                lr=float(self.config.get("lr", 1e-4)),
                weight_decay=float(self.config.get("weight_decay", 1e-4)),
                betas=(float(self.config.get("beta1", 0.9)), float(self.config.get("beta2", 0.999))),
                eps=float(self.config.get("eps", 1e-8)),
            )
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
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

    def _forward(self, batch_hs: torch.Tensor, batch_target: torch.Tensor):
        w_dtype = next(self.model.parameters()).dtype
        x = batch_hs.to(self.device, dtype=w_dtype)
        y = batch_target.to(self.device, dtype=w_dtype).reshape(-1, 1)
        pred = self.model(x)
        loss = F.mse_loss(pred, y)
        return loss, pred, y

    def train_epoch(self):
        self.model.train()
        mse_sum = 0.0
        batch_cnt = 0

        pbar = tqdm(self.train_dataloader, desc="Training", leave=False)
        for batch_hs, batch_target in pbar:
            loss, _, _ = self._forward(batch_hs, batch_target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            mse_sum += float(loss.item())
            batch_cnt += 1

        if batch_cnt == 0:
            return float("nan")
        return mse_sum / batch_cnt

    @torch.inference_mode()
    def evaluate(self):
        if self.val_dataloader is None:
            return float("nan")

        self.model.eval()
        mse_sum = 0.0
        batch_cnt = 0
        pbar = tqdm(self.val_dataloader, desc="Evaluating", leave=False)
        for batch_hs, batch_target in pbar:
            loss, _, _ = self._forward(batch_hs, batch_target)
            mse_sum += float(loss.item())
            batch_cnt += 1

        if batch_cnt == 0:
            return float("nan")
        return mse_sum / batch_cnt

    @torch.inference_mode()
    def evaluate_corr(self):
        if self.val_dataloader is None:
            return float("nan"), float("nan"), 0

        self.model.eval()
        preds = []
        targets = []
        for batch_hs, batch_target in self.val_dataloader:
            _, pred, y = self._forward(batch_hs, batch_target)
            preds.append(pred.detach().float().cpu().view(-1))
            targets.append(y.detach().float().cpu().view(-1))

        pred_all = torch.cat(preds, dim=0).numpy()
        target_all = torch.cat(targets, dim=0).numpy()
        r, p_value = pearsonr(pred_all, target_all)
        return float(r), float(p_value), int(pred_all.shape[0])

    def save_ckpt(self, ckpt_path: Path, epoch: int, train_mse: float, val_mse: float):
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config,
                "epoch": int(epoch),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
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
        best_val_mse = float("inf")
        num_epochs = int(self.config.get("num_epochs", 100))
        save_interval = int(self.config.get("save_interval", 10))
        eval_interval = int(self.config.get("eval_interval", 1))
        run_name = self.config.get("run_name", "single_class_prob_regressor")
        output_dir = Path(self.config.get("output_dir", "/root/autodl-tmp/ranker_ckpts")).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            train_mse = self.train_epoch()
            val_mse = float("nan")

            if self.scheduler is not None:
                self.scheduler.step()
            if self.val_dataloader is not None and epoch % eval_interval == 0:
                val_mse = self.evaluate()

            lr_now = self.optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch} | train_mse: {train_mse:.6f} | val_mse: {val_mse:.6f} | lr: {lr_now:.3e}")

            if epoch % save_interval == 0:
                save_interval_path = output_dir / f"{run_name}_epoch_{epoch}.pth"
                self.save_ckpt(save_interval_path, epoch, train_mse, val_mse)

            if self.val_dataloader is not None and val_mse < best_val_mse:
                best_val_mse = val_mse
                self.best_ckpt_path = output_dir / f"{run_name}_best.pth"
                self.save_ckpt(self.best_ckpt_path, epoch, train_mse, val_mse)

        if self.best_ckpt_path is not None:
            best_ckpt = torch.load(self.best_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(best_ckpt["state_dict"])

        pearson_r, p_value, n_val = self.evaluate_corr()
        print(f"Final Val Pearson r: {pearson_r:.6f}")
        print(f"Final Val Pearson p-value: {p_value:.6e}")
        print(f"Final Val Sample Count: {n_val}")


def _cfg_to_trainer_dict(cfg: DictConfig) -> dict:
    d = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(d, dict)
    d.pop("hydra", None)
    return d


@hydra.main(version_base=None, config_path="config", config_name="config_single_class")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.get("seed", 42)))
    trainer = Trainer(_cfg_to_trainer_dict(cfg))
    trainer.train()


if __name__ == "__main__":
    main()
