cd /root/diffusion_ranker

bash scripts/run_build_ranker_training_data.sh
bash scripts/train_pointwise_ranker_by_ratio.sh 0.6 0.25
bash scripts/run_random_search_with_trained_ranker.sh 0.6 0.25