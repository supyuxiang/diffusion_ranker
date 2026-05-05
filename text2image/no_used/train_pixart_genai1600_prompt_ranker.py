#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_pixart_prompt_ranker import Trainer
from utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Train text2image prompt ranker on GenAI-1600 rollout data. "
        "Prompt hstate uses PixArt prompt encoding; prompt split is 9:1 by default."
    )
    p.add_argument("--hs_reward_path", type=str, required=True, help="PT from rollout_pixart_genai1600_hiddens_rewards.py")
    p.add_argument(
        "--prompts_json",
        type=str,
        default="/root/diffusion_ranker/t2v_metrics-main/prompts/genai_image_1600.json",
        help="Fallback prompt source when PT does not include enough prompt text.",
    )
    p.add_argument("--model_path", type=str, default="/root/autodl-tmp/PixArt-XL-2-1024-MS")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/text2image_pixart_genai1600_prompt_ranker")
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


def _validate_train_ratio(ratio: float) -> None:
    if not (0.0 < ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1).")


def main() -> None:
    args = parse_args()
    _validate_train_ratio(float(args.train_ratio))
    set_seed(int(args.seed))
    trainer = Trainer(args)
    result = trainer.train()

    # Append a lightweight run summary for quick inspection.
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "note": "GenAI-1600 prompt ranker run",
        "hs_reward_path": str(args.hs_reward_path),
        "model_path": str(args.model_path),
        "train_ratio": float(args.train_ratio),
        "best_ckpt": result.get("best_ckpt"),
        "best_epoch": result.get("best_epoch"),
        "best_val_loss": result.get("best_val_loss"),
        "num_train_prompts": result.get("num_train_prompts"),
        "num_val_prompts": result.get("num_val_prompts"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

