from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from ranker.loss import mse_kl_combined_loss
from ranker.repr_training_common import (
    RankerReprGroupDataset,
    forward_listwise_transformer_ranker,
    iter_training_groups,
)


@torch.inference_mode()
def eval_epoch(
    model: torch.nn.Module,
    pt_obj: dict,
    selected_keys: list[str],
    image_root: Path,
    device: torch.device,
    reward_by_name: dict[str, float] | None,
    teacher,
    labels: list[str],
    prompt_template: str,
    ir_device: str,
    dataset: RankerReprGroupDataset,
    args: argparse.Namespace,
    ranker_type: str,
    max_groups: int = 32,
    epoch: int = 0,
) -> dict[str, float]:
    if ranker_type != "listwise":
        raise RuntimeError("eval 当前仅支持 listwise（DiffusionRanker_Transformer）")
    model.eval()
    dataset.set_epoch(epoch)
    sum_tot = sum_mse = sum_kl = 0.0
    n_steps = 0
    eval_rng = random.Random(args.seed + 17_971 + epoch)
    prompt_table = getattr(args, "prompt_table", None)
    for tok_cpu, t_scores_ir, class_id in iter_training_groups(
        dataset,
        pt_obj,
        selected_keys,
        image_root,
        reward_by_name,
        teacher,
        labels,
        prompt_template,
        ir_device,
        max_groups,
        shuffle_rng=eval_rng,
        pool_mode=args.pool_mode,
    ):
        tok = tok_cpu.to(device, non_blocking=True)
        t_scores = t_scores_ir.to(device)
        logits = forward_listwise_transformer_ranker(
            model, tok, class_id, prompt_table=prompt_table
        )
        loss, mse_v, kl_v = mse_kl_combined_loss(
            logits, t_scores, args.listwise_tau, args.alpha
        )
        sum_tot += float(loss)
        sum_mse += float(mse_v)
        sum_kl += float(kl_v)
        n_steps += 1
        if n_steps >= max_groups:
            break
    if n_steps == 0:
        return {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    return {
        "loss": sum_tot / n_steps,
        "mse": sum_mse / n_steps,
        "kl": sum_kl / n_steps,
    }
