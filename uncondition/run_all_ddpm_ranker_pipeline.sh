#!/usr/bin/env bash
set -euo pipefail

# 1) build train data
bash /root/difusion_ranker/uncondition/run_ddpm_build_train_data.sh

# 2) train ranker
bash /root/difusion_ranker/uncondition/run_ddpm_train_ranker.sh

# 3) random search with ranker
bash /root/difusion_ranker/uncondition/run_ddpm_random_search_ranker.sh

# 4) FK with ranker
bash /root/difusion_ranker/uncondition/run_ddpm_fk_ranker.sh
