#!/usr/bin/env bash
set -euo pipefail

cd /root/diffusion_ranker

bash class-condition/run_build_ranker_training_data.sh
bash class-condition/train_pointwise_ranker_by_ratio.sh 0.6 0.25
bash class-condition/run_random_search_with_trained_ranker.sh 0.6 0.25
