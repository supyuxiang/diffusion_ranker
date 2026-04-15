#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from text2image.common import dump_json, ensure_dir, load_prompts_jsonl, safe_image_name, set_seed


def _update_fk_state(
    log_belief: torch.Tensor,
    scores: torch.Tensor,
    *,
    tau: float,
    potential: str,
    prev_scores: torch.Tensor | None,
    running_max: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if potential == "difference":
        inc = scores if prev_scores is None else (scores - prev_scores)
        log_belief = log_belief + float(tau) * inc
        prev_scores = scores.clone()
    elif potential == "max":
        running_max = scores.clone() if running_max is None else torch.maximum(running_max, scores)
        log_belief = float(tau) * running_max
    else:
        raise ValueError(f"unknown potential {potential}")
    w = torch.softmax(log_belief, dim=0)
    return log_belief, w, prev_scores, running_max


def main() -> None:
    p = argparse.ArgumentParser("FK-style selection on pre-scored candidate groups")
    p.add_argument("--prompts_jsonl", type=str, required=True)
    p.add_argument("--pool_root", type=str, required=True)
    p.add_argument("--score_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--parallel_size", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--potential", type=str, choices=("difference", "max"), default="difference")
    p.add_argument("--pick", type=str, choices=("sample", "argmax"), default="argmax")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    g = torch.Generator().manual_seed(int(args.seed))
    prompts = load_prompts_jsonl(args.prompts_jsonl)
    pool_root = Path(args.pool_root).expanduser().resolve()
    score_dir = Path(args.score_dir).expanduser().resolve()
    out_dir = ensure_dir(args.output_dir)

    rows: list[dict] = []
    for item in tqdm(prompts, desc="fk prompts"):
        spt = score_dir / item.benchmark / f"{item.prompt_id}.pt"
        if not spt.is_file():
            continue
        rec = torch.load(spt, map_location="cpu", weights_only=False)
        scores_full = rec["scores"].float()
        n = int(scores_full.numel())
        if n == 0:
            continue

        psize = max(1, min(int(args.parallel_size), n))
        n_rounds = (n + psize - 1) // psize
        log_belief = torch.zeros(psize, dtype=torch.float64)
        prev_scores = None
        running_max = None
        last_round_indices = torch.full((psize,), -1, dtype=torch.long)

        for r in range(n_rounds):
            st = r * psize
            ed = min((r + 1) * psize, n)
            idx = torch.arange(st, ed, dtype=torch.long)
            if idx.numel() < psize:
                pad = idx[-1].repeat(psize - idx.numel())
                idx = torch.cat([idx, pad], dim=0)
            round_scores = scores_full[idx].double()
            log_belief, w, prev_scores, running_max = _update_fk_state(
                log_belief,
                round_scores,
                tau=float(args.tau),
                potential=str(args.potential),
                prev_scores=prev_scores,
                running_max=running_max,
            )
            last_round_indices = idx

        if args.pick == "sample":
            chosen_particle = int(torch.multinomial(torch.softmax(log_belief, dim=0).float(), 1, generator=g).item())
        else:
            chosen_particle = int(torch.argmax(log_belief).item())
        chosen_idx = int(last_round_indices[chosen_particle].item())

        src = pool_root / item.benchmark / item.prompt_id / "images" / safe_image_name(chosen_idx)
        if not src.is_file():
            continue
        dst = out_dir / f"{item.prompt_id}.png"
        shutil.copy2(src, dst)
        rows.append(
            {
                "prompt_id": item.prompt_id,
                "benchmark": item.benchmark,
                "chosen_idx": chosen_idx,
                "chosen_score": float(scores_full[chosen_idx].item()),
                "potential": args.potential,
                "tau": float(args.tau),
                "parallel_size": int(args.parallel_size),
            }
        )

    (out_dir / "selections.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )
    dump_json(
        {
            "num_selected": len(rows),
            "mean_chosen_score": (
                float(torch.tensor([r["chosen_score"] for r in rows], dtype=torch.float32).mean().item())
                if rows
                else float("nan")
            ),
            "potential": args.potential,
            "tau": float(args.tau),
            "parallel_size": int(args.parallel_size),
            "pick": args.pick,
        },
        out_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
