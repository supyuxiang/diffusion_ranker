#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python /root/diffusion_ranker/ranker/train.py \
    --num_epochs 10 \
    --output_dir /root/autodl-tmp/ranker_ckpt_pointwise_l06_s2 \
    --image_dir /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10 \
    --candidate_repr_pt /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/mean/step_0.25_layer_0.6.pt \
    --prompt_repr_pt /root/diffusion_ranker/ranker/prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt \
    --reward_path /root/autodl-tmp/dit_training_data_perclass32_cfg1.0_deno10/reward_ls.json \
    --batch_size 32 \
    --num_cand 10 \
    --input_dim_candidate 1152 \
    --input_dim_prompt 3584 \
    --hidden_dim 128 \
    --num_attention_heads 4 \
    --num_layers 8 \
    --dropout 0.1 \
    --optimizer adamw \
    --lr_scheduler cosine \
    --warmup_ratio 0.03 \
    --lr_min_ratio 0.01 \
    --alpha 0.8 \
    --tau 1.0 \
    --seed 42 \
    --num_candidate_groups 8 \
    --ranker_type pointwise \
    --has_prompt_hstates