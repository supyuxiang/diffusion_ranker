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

from text2image.common import dump_json, ensure_dir, load_prompts_jsonl, safe_image_name


def _pick_best_with_cap(scores: torch.Tensor, n: int) -> int:
    if scores.numel() == 0:
        return -1
    n_cap = max(1, min(int(n), int(scores.numel())))
    return int(torch.argmax(scores[:n_cap]).item())


def main() -> None:
    p = argparse.ArgumentParser("Best-of-N selection from pre-scored candidate pool")
    p.add_argument("--prompts_jsonl", type=str, required=True)
    p.add_argument("--pool_root", type=str, required=True)
    p.add_argument("--score_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--n_values", type=str, default="1,2,4,8,10")
    args = p.parse_args()

    prompts = load_prompts_jsonl(args.prompts_jsonl)
    pool_root = Path(args.pool_root).expanduser().resolve()
    score_dir = Path(args.score_dir).expanduser().resolve()
    out_root = ensure_dir(args.output_dir)
    n_values = [int(x.strip()) for x in str(args.n_values).split(",") if x.strip()]
    n_values = sorted(set(n_values))

    for n in n_values:
        od = ensure_dir(out_root / f"best_of_{n:02d}")
        rows: list[dict] = []
        for item in tqdm(prompts, desc=f"best-of-{n}"):
            score_pt = score_dir / item.benchmark / f"{item.prompt_id}.pt"
            if not score_pt.is_file():
                continue
            rec = torch.load(score_pt, map_location="cpu", weights_only=False)
            scores = rec["scores"].float()
            best_idx = _pick_best_with_cap(scores, n)
            src = pool_root / item.benchmark / item.prompt_id / "images" / safe_image_name(best_idx)
            if best_idx < 0 or (not src.is_file()):
                continue
            dst_name = f"{item.prompt_id}.png"
            shutil.copy2(src, od / dst_name)
            rows.append(
                {
                    "prompt_id": item.prompt_id,
                    "benchmark": item.benchmark,
                    "chosen_idx": int(best_idx),
                    "chosen_score": float(scores[best_idx].item()),
                    "dst_image": dst_name,
                }
            )
        (od / "selections.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        dump_json(
            {
                "n": int(n),
                "num_selected": len(rows),
                "mean_chosen_score": (
                    float(torch.tensor([r["chosen_score"] for r in rows], dtype=torch.float32).mean().item())
                    if rows
                    else float("nan")
                ),
            },
            od / "summary.json",
        )


if __name__ == "__main__":
    main()
