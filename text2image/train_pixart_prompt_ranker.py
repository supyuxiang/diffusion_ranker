#!/usr/bin/env python3



'''
rollout:
python "/root/diffusion_ranker/text2image/rollout_pixart_hiddens_rewards.py" --model_path "/root/autodl-tmp/PixArt-XL-2-1024-MS" --prompts_json "/root/autodl-tmp/prompt_datasets/genai_image_527_list.json" --output_root "/root/autodl-tmp/text2image_pixart_hs_reward_rollout_527_v1" --repeats_per_prompt 1 --batch_size 4 --num_inference_steps 20 --guidance_scale 4.5 --negative_prompt "blurry, low quality, watermark, text, deformed" --height 1024 --width 1024 --target_step_idx 10 --target_layer_idx 14 --mc 8 --dtype fp16 --seed 42



train ranker:

python "/root/diffusion_ranker/text2image/train_pixart_prompt_ranker.py" \
  --hs_reward_path "/root/autodl-tmp/text2image_pixart_hs_reward_rollout_527_v1/pixart_prompt_hs_reward_n527_s10_l14_mc4.pt" \
  --model_path "/root/autodl-tmp/PixArt-XL-2-1024-MS" \
  --output_dir "/root/autodl-tmp/text2image_pixart_prompt_ranker_527_s10_l14_mc4" \
  --seed 42 \
  --dtype fp16 \
  --train_ratio 0.9 \
  --ranker_type listwise \
  --hidden_dim 128 \
  --num_heads 4 \
  --num_layers 2 \
  --dropout 0.1 \
  --batch_size 32 \
  --parallel_size 8 \
  --num_bootstrap 8 \
  --num_epochs 100 \
  --eval_interval 1 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --alpha 0.8 \
  --kl_type kl

'''
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from class_condition.utils import RankerLoss, js_loss, kl_loss, mse_loss
from text2image.ranker import (
    DiffusionRanker_CrossAttention,
    DiffusionRanker_MLP,
    DiffusionRanker_Transformer,
)
from utils import set_seed


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


class PromptGroupedDataset(Dataset):
    def __init__(
        self,
        prompt_hs: torch.Tensor,
        candidate_groups: list[torch.Tensor],
        reward_groups: list[torch.Tensor],
    ):
        self.prompt_hs = prompt_hs.float()
        self.candidate_groups = [x.float() for x in candidate_groups]
        self.reward_groups = [y.float() for y in reward_groups]
        if len(self.prompt_hs) != len(self.candidate_groups) or len(self.prompt_hs) != len(self.reward_groups):
            raise ValueError("Dataset length mismatch.")

    def __len__(self) -> int:
        return int(self.prompt_hs.shape[0])

    def __getitem__(self, idx: int):
        return self.prompt_hs[idx], self.candidate_groups[idx], self.reward_groups[idx]


def _collate_prompt_groups(batch):
    prompts = [x[0] for x in batch]
    candidates = [x[1] for x in batch]
    rewards = [x[2] for x in batch]
    return prompts, candidates, rewards


def _extract_unique_prompts(obj: dict, prompts_json: str | None) -> tuple[list[str], list[int]]:
    prompt_indices = obj.get("prompt_indices", None)
    prompts_all = obj.get("prompts", None)
    if prompt_indices is None:
        raise KeyError("Missing key `prompt_indices` in dataset pt.")
    prompt_indices_t = torch.as_tensor(prompt_indices, dtype=torch.long).view(-1)

    if prompts_all is not None and len(prompts_all) == int(prompt_indices_t.shape[0]):
        idx2prompt: dict[int, str] = {}
        for i, pid in enumerate(prompt_indices_t.tolist()):
            if pid not in idx2prompt:
                idx2prompt[pid] = str(prompts_all[i])
        sorted_old_ids = sorted(idx2prompt.keys())
        unique_prompts = [idx2prompt[i] for i in sorted_old_ids]
        return unique_prompts, sorted_old_ids

    if prompts_json is None:
        raise ValueError(
            "Cannot reconstruct unique prompts from pt. "
            "Please provide --prompts_json (JSON list[str] or list[{prompt:...}])."
        )

    p = Path(prompts_json).expanduser().resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"Invalid prompts_json: {p}")
    prompt_pool: list[str] = []
    for i, x in enumerate(raw):
        if isinstance(x, str):
            s = x.strip()
        elif isinstance(x, dict):
            s = str(x.get("prompt", "")).strip()
        else:
            raise ValueError(f"Unsupported prompt type at index={i}: {type(x)}")
        if not s:
            raise ValueError(f"Empty prompt at index={i}")
        prompt_pool.append(s)

    max_pid = int(prompt_indices_t.max().item())
    if max_pid >= len(prompt_pool):
        raise ValueError(f"prompt_indices has id={max_pid}, but prompts_json has only {len(prompt_pool)} prompts.")
    sorted_old_ids = sorted(set(prompt_indices_t.tolist()))
    unique_prompts = [prompt_pool[i] for i in sorted_old_ids]
    return unique_prompts, sorted_old_ids


@torch.inference_mode()
def _encode_prompts_pixart(
    model_path: str,
    prompts: list[str],
    *,
    device: torch.device,
    dtype_name: str,
    encode_batch_size: int,
    max_sequence_length: int,
) -> torch.Tensor:
    torch_dtype = _resolve_dtype(dtype_name, device)
    pipe = PixArtAlphaPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    all_vecs: list[torch.Tensor] = []
    pbar = tqdm(range(0, len(prompts), encode_batch_size), desc="encode prompts", leave=False)
    for st in pbar:
        ed = min(st + encode_batch_size, len(prompts))
        batch_prompts = prompts[st:ed]
        prompt_embeds, *_ = pipe.encode_prompt(
            batch_prompts,
            do_classifier_free_guidance=False,
            negative_prompt="",
            num_images_per_prompt=1,
            device=device,
            clean_caption=False,
            max_sequence_length=max_sequence_length,
        )
        all_vecs.append(prompt_embeds.mean(dim=1).float().cpu())
    return torch.cat(all_vecs, dim=0).contiguous()


def _build_grouped_data(
    *,
    hs_reward_path: Path,
    prompts_json: str | None,
    model_path: str,
    seed: int,
    train_ratio: float,
    dtype_name: str,
    encode_batch_size: int,
    max_sequence_length: int,
) -> dict:
    obj = torch.load(hs_reward_path, map_location="cpu", weights_only=False)
    hs = torch.as_tensor(obj["hiddens"]).float()
    reward = torch.as_tensor(obj["reward"]).view(-1).float()
    prompt_indices = torch.as_tensor(obj["prompt_indices"]).view(-1).long()
    if hs.shape[0] != reward.shape[0] or hs.shape[0] != prompt_indices.shape[0]:
        raise ValueError("Row mismatch among hiddens/reward/prompt_indices.")

    unique_prompts, sorted_old_ids = _extract_unique_prompts(obj, prompts_json=prompts_json)
    old2new = {old_id: i for i, old_id in enumerate(sorted_old_ids)}
    compact_pid = torch.tensor([old2new[int(x)] for x in prompt_indices.tolist()], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_vecs = _encode_prompts_pixart(
        model_path=model_path,
        prompts=unique_prompts,
        device=device,
        dtype_name=dtype_name,
        encode_batch_size=encode_batch_size,
        max_sequence_length=max_sequence_length,
    )

    grouped_hs: list[torch.Tensor] = []
    grouped_reward: list[torch.Tensor] = []
    for pid in range(len(unique_prompts)):
        mask = compact_pid == pid
        cand = hs[mask]
        rew = reward[mask]
        if cand.shape[0] < 2:
            continue
        grouped_hs.append(cand)
        grouped_reward.append(rew)
    kept_prompt_ids = [i for i in range(len(unique_prompts)) if (compact_pid == i).sum().item() >= 2]
    kept_prompts = [unique_prompts[i] for i in kept_prompt_ids]
    prompt_vecs = prompt_vecs[torch.tensor(kept_prompt_ids, dtype=torch.long)]

    if len(kept_prompts) < 2:
        raise ValueError("Need at least 2 prompts with >=2 samples each for prompt-level train/val split.")

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(kept_prompts), generator=g)
    n_train = min(max(int(round(len(kept_prompts) * float(train_ratio))), 1), len(kept_prompts) - 1)
    train_ids = perm[:n_train].tolist()
    val_ids = perm[n_train:].tolist()

    train_reward_all = torch.cat([grouped_reward[i] for i in train_ids], dim=0)
    r_mean = float(train_reward_all.mean().item())
    r_std = float(train_reward_all.std(unbiased=False).item())
    denom = r_std + 1e-8

    train_prompt = prompt_vecs[torch.tensor(train_ids, dtype=torch.long)]
    train_hs = [grouped_hs[i] for i in train_ids]
    train_reward = [(grouped_reward[i] - r_mean) / denom for i in train_ids]
    val_prompt = prompt_vecs[torch.tensor(val_ids, dtype=torch.long)]
    val_hs = [grouped_hs[i] for i in val_ids]
    val_reward = [(grouped_reward[i] - r_mean) / denom for i in val_ids]

    return {
        "train_prompts_text": [kept_prompts[i] for i in train_ids],
        "val_prompts_text": [kept_prompts[i] for i in val_ids],
        "train_dataset": PromptGroupedDataset(train_prompt, train_hs, train_reward),
        "val_dataset": PromptGroupedDataset(val_prompt, val_hs, val_reward),
        "reward_mean_train": r_mean,
        "reward_std_train": r_std,
    }


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        grouped = _build_grouped_data(
            hs_reward_path=Path(args.hs_reward_path).expanduser().resolve(),
            prompts_json=args.prompts_json,
            model_path=args.model_path,
            seed=int(args.seed),
            train_ratio=float(args.train_ratio),
            dtype_name=str(args.dtype),
            encode_batch_size=int(args.encode_batch_size),
            max_sequence_length=int(args.max_sequence_length),
        )
        self.train_prompts_text = grouped["train_prompts_text"]
        self.val_prompts_text = grouped["val_prompts_text"]
        self.train_dataset = grouped["train_dataset"]
        self.val_dataset = grouped["val_dataset"]
        self.reward_mean_train = grouped["reward_mean_train"]
        self.reward_std_train = grouped["reward_std_train"]

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=_collate_prompt_groups,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=_collate_prompt_groups,
        )
        self._build_ranker()
        self._build_opt_sched()
        self.loss_fn = RankerLoss(alpha=float(args.alpha), kl_type=str(args.kl_type))
        self.div_loss_fn = kl_loss if str(args.kl_type) == "kl" else js_loss

    def _build_ranker(self):
        d_c = int(self.train_dataset.candidate_groups[0].shape[1])
        d_p = int(self.train_dataset.prompt_hs.shape[1])
        ranker_type = str(self.args.ranker_type)
        if ranker_type == "listwise":
            self.ranker = DiffusionRanker_Transformer(
                input_dim_candidate=d_c,
                input_dim_prompt=d_p,
                hidden_dim=int(self.args.hidden_dim),
                num_heads=int(self.args.num_heads),
                dropout=float(self.args.dropout),
                num_layers=int(self.args.num_layers),
                half=bool(self.args.half),
            )
        elif ranker_type == "cross_attention":
            self.ranker = DiffusionRanker_CrossAttention(
                candidate_hidden_dim=d_c,
                prompt_hidden_dim=d_p,
                hidden_dim=int(self.args.hidden_dim),
                num_heads=int(self.args.num_heads),
                num_layers=int(self.args.num_layers),
                dropout=float(self.args.dropout),
                half=bool(self.args.half),
            )
        elif ranker_type == "pointwise":
            self.ranker = DiffusionRanker_MLP(
                candidate_hidden_dim=d_c,
                prompt_hidden_dim=d_p,
                hidden_dim=int(self.args.hidden_dim),
                output_dim=1,
                num_layers=int(self.args.num_layers),
                dropout=float(self.args.dropout),
                half=bool(self.args.half),
                criterion=None,
            )
        else:
            raise ValueError(f"Unsupported ranker_type: {ranker_type}")
        self.ranker = self.ranker.to(self.device)

    def _build_opt_sched(self):
        self.optimizer = torch.optim.AdamW(
            self.ranker.parameters(),
            lr=float(self.args.lr),
            weight_decay=float(self.args.weight_decay),
            betas=(float(self.args.beta1), float(self.args.beta2)),
            eps=float(self.args.eps),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(self.args.num_epochs)),
            eta_min=float(self.args.lr) * 0.01,
        )

    def _prompt_batch_loss(
        self,
        prompt_hs: torch.Tensor,
        candidate_hs: torch.Tensor,
        reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_hs.shape[0] < 2:
            z = torch.zeros((), device=self.device)
            return z, z, z

        reward = reward.view(-1)
        parallel_size = max(2, int(self.args.parallel_size))
        num_bootstrap = max(1, int(self.args.num_bootstrap))
        sample_size = min(parallel_size, int(candidate_hs.shape[0]))
        with_replacement = bool(candidate_hs.shape[0] < parallel_size)

        mini_batch_candidate_hs = []
        mini_batch_reward = []
        for _ in range(num_bootstrap):
            idx = torch.multinomial(
                torch.ones(candidate_hs.shape[0]),
                num_samples=sample_size,
                replacement=with_replacement,
            )
            mini_batch_candidate_hs.append(candidate_hs[idx])
            mini_batch_reward.append(reward[idx])

        cand = torch.stack(mini_batch_candidate_hs, dim=0).to(self.device)
        target = torch.stack(mini_batch_reward, dim=0).to(self.device)
        w_dtype = next(self.ranker.parameters()).dtype
        cand = cand.to(dtype=w_dtype)
        target = target.to(dtype=w_dtype)
        prompt_hs = prompt_hs.to(self.device, dtype=w_dtype)

        if str(self.args.ranker_type) == "pointwise":
            mini_b, k, d_c = cand.shape
            cand_flat = cand.reshape(mini_b * k, 1, d_c)
            target_flat = target.reshape(mini_b * k, 1)
            prompt_flat = prompt_hs.unsqueeze(0).expand(mini_b * k, -1)
            student_logits = self.ranker(cand_flat, prompt_flat)
            loss = self.loss_fn(student_logits, target_flat)
            mse_term = mse_loss(student_logits, target_flat).detach()
            div_term = self.div_loss_fn(student_logits, target_flat).detach()
            return loss, mse_term, div_term

        if str(self.args.ranker_type) == "listwise":
            prompt_batch = prompt_hs.unsqueeze(0).unsqueeze(0).expand(cand.shape[0], 1, -1)
            student_logits = self.ranker(cand, prompt_batch)
        else:
            prompt_batch = prompt_hs.unsqueeze(0).expand(cand.shape[0], -1)
            student_logits = self.ranker(cand, prompt_batch)

        loss = self.loss_fn(student_logits, target)
        mse_term = mse_loss(student_logits, target).detach()
        div_term = self.div_loss_fn(student_logits, target).detach()
        return loss, mse_term, div_term

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float, float]:
        if training:
            self.ranker.train()
        else:
            self.ranker.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_div = 0.0
        total_prompts = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, desc="train" if training else "eval", leave=False)
            for prompt_list, cand_list, reward_list in pbar:
                batch_losses = []
                batch_mse = []
                batch_div = []
                for prompt_hs, cand, rew in zip(prompt_list, cand_list, reward_list):
                    loss, mse_term, div_term = self._prompt_batch_loss(prompt_hs, cand, rew)
                    if torch.isfinite(loss):
                        batch_losses.append(loss)
                        batch_mse.append(mse_term)
                        batch_div.append(div_term)
                if len(batch_losses) == 0:
                    continue

                loss_mean = torch.stack(batch_losses).mean()
                mse_mean = torch.stack(batch_mse).mean()
                div_mean = torch.stack(batch_div).mean()
                if training:
                    self.optimizer.zero_grad()
                    loss_mean.backward()
                    torch.nn.utils.clip_grad_norm_(self.ranker.parameters(), max_norm=3.0, norm_type=2)
                    self.optimizer.step()

                bsz_eff = len(batch_losses)
                total_prompts += bsz_eff
                total_loss += float(loss_mean.detach().item()) * bsz_eff
                total_mse += float(mse_mean.detach().item()) * bsz_eff
                total_div += float(div_mean.detach().item()) * bsz_eff

        if total_prompts <= 0:
            return float("inf"), float("inf"), float("inf")
        return total_loss / total_prompts, total_mse / total_prompts, total_div / total_prompts

    def train(self) -> dict:
        output_dir = Path(self.args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = output_dir / "ranker_best.pt"
        history: list[dict] = []
        best_val = float("inf")
        best_epoch = -1

        for epoch in range(int(self.args.num_epochs)):
            tr_loss, tr_mse, tr_div = self._run_epoch(self.train_loader, training=True)
            self.scheduler.step()
            if epoch % int(self.args.eval_interval) != 0:
                continue

            va_loss, va_mse, va_div = self._run_epoch(self.val_loader, training=False)
            rec = {
                "epoch": int(epoch),
                "train_loss": float(tr_loss),
                "train_mse": float(tr_mse),
                "train_div": float(tr_div),
                "val_loss": float(va_loss),
                "val_mse": float(va_mse),
                "val_div": float(va_div),
            }
            history.append(rec)
            print(
                f"epoch {epoch:03d} | train={tr_loss:.6f} (mse={tr_mse:.6f}, div={tr_div:.6f}) "
                f"| val={va_loss:.6f} (mse={va_mse:.6f}, div={va_div:.6f})"
            )
            if va_loss < best_val:
                best_val = va_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": int(epoch),
                        "val_loss": float(va_loss),
                        "model_state_dict": self.ranker.state_dict(),
                        "config": vars(self.args),
                        "reward_mean_train": float(self.reward_mean_train),
                        "reward_std_train": float(self.reward_std_train),
                    },
                    best_ckpt,
                )

        result = {
            "best_ckpt": str(best_ckpt),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "num_train_prompts": int(len(self.train_prompts_text)),
            "num_val_prompts": int(len(self.val_prompts_text)),
            "parallel_size": int(self.args.parallel_size),
            "num_bootstrap": int(self.args.num_bootstrap),
            "train_ratio": float(self.args.train_ratio),
            "reward_mean_train": float(self.reward_mean_train),
            "reward_std_train": float(self.reward_std_train),
        }
        (output_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "train_prompts.json").write_text(
            json.dumps(self.train_prompts_text, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "val_prompts.json").write_text(
            json.dumps(self.val_prompts_text, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "final_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train text2image prompt ranker with PixArt prompt encoding.")
    p.add_argument("--hs_reward_path", type=str, required=True, help="PT from rollout_pixart_hiddens_rewards.py")
    p.add_argument("--prompts_json", type=str, default=None, help="Fallback prompt pool if PT has no per-sample prompts.")
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/PixArt-XL-2-1024-MS")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/text2image_pixart_prompt_ranker")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="fp16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--encode_batch_size", type=int, default=32)
    p.add_argument("--max_sequence_length", type=int, default=120)

    p.add_argument("--ranker_type", type=str, default="listwise", choices=("listwise", "pointwise", "cross_attention"))
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--half", action="store_true")

    p.add_argument("--batch_size", type=int, default=32, help="Prompt groups per batch.")
    p.add_argument("--parallel_size", type=int, default=8, help="Candidate subset size in each bootstrap.")
    p.add_argument("--num_bootstrap", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--kl_type", type=str, default="kl", choices=("kl", "js"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    trainer = Trainer(args)
    result = trainer.train()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
