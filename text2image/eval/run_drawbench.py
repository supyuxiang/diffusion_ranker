#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser("Run DrawBench evaluation (CLIP proxy)")
    p.add_argument("--prompts_jsonl", type=str, required=True)
    p.add_argument("--selected_dir", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--clip_model_dir", type=str, default="/root/autodl-tmp/openai_clip-vit-large-patch14")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    this_dir = Path(__file__).resolve().parent
    geneval_py = this_dir / "run_geneval.py"
    cmd = [
        sys.executable,
        str(geneval_py),
        "--prompts_jsonl",
        args.prompts_jsonl,
        "--selected_dir",
        args.selected_dir,
        "--output_json",
        args.output_json,
        "--clip_model_dir",
        args.clip_model_dir,
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
