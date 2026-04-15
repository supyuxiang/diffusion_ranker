# text2image pipeline

This directory provides a complete runnable pipeline for prompt-based text-to-image experiments:

- Candidate pool generation
- Reward scoring (CLIPScore / ImageReward / PickScore)
- Best-of-N selection
- FK-style selection
- Ranker train + eval
- Metric aggregation and plotting

## Prompt format

Use JSONL with one prompt per line:

```json
{"prompt_id":"p0001","prompt_text":"a red car on snow","benchmark":"geneval","tags":["color"]}
```

Required keys:
- `prompt_id` (string, unique)
- `prompt_text` (string)

Optional keys:
- `benchmark` (default: `custom`)
- `tags` (list of strings)

## 1) Build candidate pool

```bash
python text2image/generate/build_candidate_pool.py \
  --prompts_jsonl text2image/data/prompts/example_prompts.jsonl \
  --output_root /root/autodl-tmp/t2i_runs/candidate_pool \
  --model_name_or_path /root/autodl-tmp/stable-diffusion-v1-5 \
  --num_candidates 64
```

## 2) Score candidates

```bash
python text2image/reward/score_clip.py \
  --pool_root /root/autodl-tmp/t2i_runs/candidate_pool \
  --output_dir /root/autodl-tmp/t2i_runs/scores_clip
```

```bash
python text2image/reward/score_imagereward.py \
  --pool_root /root/autodl-tmp/t2i_runs/candidate_pool \
  --output_dir /root/autodl-tmp/t2i_runs/scores_imagereward
```

```bash
python text2image/reward/score_pickscore_or_hpsv2.py \
  --pool_root /root/autodl-tmp/t2i_runs/candidate_pool \
  --output_dir /root/autodl-tmp/t2i_runs/scores_pickscore \
  --mode pickscore \
  --model_name_or_path /root/autodl-tmp/PickScore_v1
```

## 3) Best-of-N

```bash
python text2image/search/best_of_n.py \
  --prompts_jsonl text2image/data/prompts/example_prompts.jsonl \
  --pool_root /root/autodl-tmp/t2i_runs/candidate_pool \
  --score_dir /root/autodl-tmp/t2i_runs/scores_imagereward \
  --output_dir /root/autodl-tmp/t2i_runs/best_of_n/imagereward \
  --n_values 1,2,4,8,10
```

## 4) FK-style selection

```bash
python text2image/search/fk_steering.py \
  --prompts_jsonl text2image/data/prompts/example_prompts.jsonl \
  --pool_root /root/autodl-tmp/t2i_runs/candidate_pool \
  --score_dir /root/autodl-tmp/t2i_runs/scores_imagereward \
  --output_dir /root/autodl-tmp/t2i_runs/fk/imagereward_difference_ps10 \
  --parallel_size 10 \
  --potential difference \
  --tau 0.1
```

## 5) Ranker (teacher-supervised)

Build trainset:

```bash
python text2image/ranker/build_ranker_trainset.py \
  --pool_root /root/autodl-tmp/t2i_runs/candidate_pool \
  --teacher_score_dir /root/autodl-tmp/t2i_runs/scores_imagereward \
  --output_pt /root/autodl-tmp/t2i_runs/ranker/trainset.pt
```

Train:

```bash
python text2image/ranker/train_ranker.py \
  --train_pt /root/autodl-tmp/t2i_runs/ranker/trainset.pt \
  --output_dir /root/autodl-tmp/t2i_runs/ranker/ckpt
```

Eval:

```bash
python text2image/ranker/eval_ranker.py \
  --prompts_jsonl text2image/data/prompts/example_prompts.jsonl \
  --pool_root /root/autodl-tmp/t2i_runs/candidate_pool \
  --ranker_ckpt /root/autodl-tmp/t2i_runs/ranker/ckpt/ranker_final.pt \
  --prompt_table_pt /root/autodl-tmp/t2i_runs/ranker/ckpt/prompt_table.pt \
  --output_dir /root/autodl-tmp/t2i_runs/ranker/selected_by_ranker
```

## 6) Eval aggregation and plotting

```bash
python text2image/eval/aggregate_metrics.py \
  --input_root /root/autodl-tmp/t2i_runs \
  --out_csv /root/autodl-tmp/t2i_runs/summary.csv
```

```bash
python text2image/eval/plot_curves.py \
  --csv /root/autodl-tmp/t2i_runs/summary.csv \
  --out_png /root/autodl-tmp/t2i_runs/summary.png
```

## Convenience scripts

- `text2image/scripts/run_all_baselines.sh`
- `text2image/scripts/run_all_fk.sh`
- `text2image/scripts/run_all_ranker.sh`
