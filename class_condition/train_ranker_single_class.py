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


class SingleClassDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        label_key: str = "inception_confidence",
        normalize_target: bool = True,
    ):
        super().__init__()
        self.data_path = Path(data_path).expanduser().resolve()
        payload = torch.load(self.data_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {self.data_path}, got {type(payload)}")

        self.hiddens = payload["hiddens"].to(torch.float32)
        self.labels = payload.get("labels", torch.full((self.hiddens.shape[0],), -1, dtype=torch.long)).reshape(-1)
        self.target_key, target_tensor = self._select_target(payload, label_key)
        self.targets = target_tensor.to(torch.float32).reshape(-1)
        self.meta = payload.get("meta", {})

        if self.hiddens.ndim != 2:
            raise ValueError(f"hiddens must be 2D (N,D), got {tuple(self.hiddens.shape)}")
        if self.hiddens.shape[0] != self.targets.shape[0]:
            raise ValueError(
                f"Row mismatch: hiddens={self.hiddens.shape[0]} targets={self.targets.shape[0]}"
            )
        if self.labels.shape[0] != self.hiddens.shape[0]:
            raise ValueError(
                f"Row mismatch: hiddens={self.hiddens.shape[0]} labels={self.labels.shape[0]}"
            )

        self.target_mean = float(self.targets.mean().item())
        self.target_std = float(self.targets.std().item())
        if normalize_target:
            self.targets = (self.targets - self.target_mean) / (self.target_std + 1e-8)

        self.hidden_dim = int(self.hiddens.shape[1])
        self.num_samples = int(self.hiddens.shape[0])

    @staticmethod
    def _select_target(payload: dict, label_key: str) -> tuple[str, torch.Tensor]:
        aliases = {
            "inception_confidence": ("inception_confidence", "inception", "inception_score", "reward"),
            "clipscore": ("clipscore", "clip_score", "clip_reward", "reward"),
            "imagereward": ("imagereward", "image_reward", "ir", "reward"),
            "reward": ("reward",),
        }
        candidates = aliases.get(label_key, (label_key,))
        for key in candidates:
            if key in payload:
                return key, payload[key]
        available = ", ".join(sorted(payload.keys()))
        raise KeyError(f"Cannot find label key '{label_key}'. Available keys: {available}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.hiddens[idx], self.targets[idx], self.labels[idx]


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
        data_path = self.config.get("single_class_data_path", self.config.get("hs_reward_path"))
        if data_path is None:
            raise ValueError("Missing `single_class_data_path` in config.")

        self.dataset = SingleClassDataset(
            data_path=data_path,
            label_key=self.config.get("label_key", "inception_confidence"),
            normalize_target=bool(self.config.get("normalize_target", True)),
        )
        split_ratio = self.config.get("split_ratio", [0.9, 0.1])
        if len(split_ratio) != 2:
            raise ValueError(f"split_ratio must have 2 values, got {split_ratio}")
        train_ratio, val_ratio = float(split_ratio[0]), float(split_ratio[1])
        ratio_sum = max(train_ratio + val_ratio, 1e-8)
        train_ratio, val_ratio = train_ratio / ratio_sum, val_ratio / ratio_sum

        n_total = len(self.dataset)
        n_train = int(round(n_total * train_ratio))
        n_train = min(max(n_train, 1), n_total)
        n_val = n_total - n_train
        if n_val == 0 and n_total > 1:
            n_train = n_total - 1
            n_val = 1

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
        )
        self.val_dataloader = None
        if n_val > 0:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=int(self.config.get("num_workers", 4)),
                drop_last=False,
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
        loss_name = self.config.get("loss", "mse")
        if loss_name == "mse":
            self.loss_fn = F.mse_loss
        elif loss_name == "l1":
            self.loss_fn = F.l1_loss
        elif loss_name == "huber":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError(f"Invalid loss: {loss_name}")

    def process_batch(self, batch_hs, batch_target):
        w_dtype = next(self.ranker.parameters()).dtype
        batch_hs = batch_hs.to(self.device, dtype=w_dtype)
        batch_target = batch_target.to(self.device, dtype=w_dtype).reshape(-1, 1)
        pred = self.ranker(batch_hs)
        loss = self.loss_fn(pred, batch_target)
        mse = F.mse_loss(pred, batch_target).detach()
        mae = F.l1_loss(pred, batch_target).detach()
        return loss, mse, mae

    def train_epoch(self):
        self.ranker.train()
        loss_sum = 0.0
        mse_sum = 0.0
        mae_sum = 0.0
        batch_cnt = 0

        pbar = tqdm(self.train_dataloader, desc="Training", leave=False)
        for batch_hs, batch_target, _ in pbar:
            loss, mse, mae = self.process_batch(batch_hs, batch_target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            loss_sum += float(loss.item())
            mse_sum += float(mse.item())
            mae_sum += float(mae.item())
            batch_cnt += 1

        if batch_cnt == 0:
            return float("nan"), float("nan"), float("nan")
        return loss_sum / batch_cnt, mse_sum / batch_cnt, mae_sum / batch_cnt

    @torch.inference_mode()
    def evaluate(self):
        if self.val_dataloader is None:
            return float("nan"), float("nan"), float("nan")

        self.ranker.eval()
        loss_sum = 0.0
        mse_sum = 0.0
        mae_sum = 0.0
        batch_cnt = 0
        for batch_hs, batch_target, _ in self.val_dataloader:
            loss, mse, mae = self.process_batch(batch_hs, batch_target)
            loss_sum += float(loss.item())
            mse_sum += float(mse.item())
            mae_sum += float(mae.item())
            batch_cnt += 1

        if batch_cnt == 0:
            return float("nan"), float("nan"), float("nan")
        return loss_sum / batch_cnt, mse_sum / batch_cnt, mae_sum / batch_cnt

    def save_ckpt(self, ckpt_path: Path, epoch: int, train_loss: float, val_loss: float):
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.ranker.state_dict(),
                "config": self.config,
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "data_path": str(self.dataset.data_path),
                "target_key": self.dataset.target_key,
                "target_mean": self.dataset.target_mean,
                "target_std": self.dataset.target_std,
                "num_samples": self.dataset.num_samples,
                "hidden_dim": self.dataset.hidden_dim,
            },
            ckpt_path,
        )

    def train(self):
        best_loss = float("inf")
        num_epochs = int(self.config.get("num_epochs", 100))
        save_interval = int(self.config.get("save_interval", 10))
        eval_interval = int(self.config.get("eval_interval", 1))
        run_name = self.config.get("run_name", "single_class_regressor")
        output_dir = Path(self.config.get("output_dir", "/root/autodl-tmp/ranker_ckpts")).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            train_loss, train_mse, train_mae = self.train_epoch()
            val_loss, val_mse, val_mae = (float("nan"), float("nan"), float("nan"))
            if self.scheduler is not None:
                self.scheduler.step()
            if epoch % eval_interval == 0:
                val_loss, val_mse, val_mae = self.evaluate()

            lr_now = self.optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch} | train_loss: {train_loss:.6f} | train_mse: {train_mse:.6f} | "
                f"train_mae: {train_mae:.6f} | val_loss: {val_loss:.6f} | val_mse: {val_mse:.6f} | "
                f"val_mae: {val_mae:.6f} | lr: {lr_now:.3e}"
            )

            if epoch % save_interval == 0:
                save_interval_path = output_dir / f"{run_name}_epoch_{epoch}.pth"
                self.save_ckpt(save_interval_path, epoch, train_loss, val_loss)

            metric_for_best = val_loss if self.val_dataloader is not None else train_loss
            if metric_for_best < best_loss:
                best_loss = metric_for_best
                save_best_path = output_dir / f"{run_name}_best.pth"
                self.save_ckpt(save_best_path, epoch, train_loss, val_loss)

        print(f"Best Loss: {best_loss:.6f}")


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
