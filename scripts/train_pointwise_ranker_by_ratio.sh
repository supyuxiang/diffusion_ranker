#!/usr/bin/env bash
set -euo pipefail

bash /root/diffusion_ranker/class-condition/train_pointwise_ranker_by_ratio.sh "$@"

