#!/usr/bin/env python3
from __future__ import annotations

from train_common import build_parser, run_train_for_type


def main() -> None:
    parser = build_parser(
        description="Train pointwise MLP ranker for one (layer_ratio, step_ratio).",
        default_num_layers=8,
        default_num_attention_heads=4,
        default_hidden_dim=128,
    )
    parser.set_defaults(output_prefix="/root/autodl-tmp/ranker_ckpt")
    args = parser.parse_args()
    run_train_for_type(args, ranker_type="pointwise")


if __name__ == "__main__":
    main()
