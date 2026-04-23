#!/bin/bash
# Sweep num_candidates (2,3,4,5) for Random Search FID
# Run from project1/idm directory

set -e
cd /data/home/yxfeng/project1/idm

DIT_PATH="${DIT_PATH:-/data/yxfeng/models/dit_model}"
IMAGENET_VAL="${IMAGENET_VAL:-/data/yxfeng/datasets/imagenet/val}"
BASE_OUT="/data/yxfeng/random_search_num_candidates_sweep_FID_Reward"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
INIT_RANDOM="${INIT_RANDOM:-1024}"
DENO_STEPS=10
BATCH_SIZE=48

echo "=== num_candidates sweep: 2,3,4,5,6,7,8 ==="
echo "num_samples=$NUM_SAMPLES, init_random_samples=$INIT_RANDOM, deno_steps=$DENO_STEPS"
echo ""

export CUDA_VISIBLE_DEVICES=4

for K in 2 3 4 5 6 7 8; do
    OUT_DIR="${BASE_OUT}_K${K}"
    echo ">>> Running num_candidates=$K -> $OUT_DIR"
    python3 search_algoriths/dit_random_search_fid_reward.py \
        --dit_path "$DIT_PATH" \
        --imagenet_val "$IMAGENET_VAL" \
        --output_dir "$OUT_DIR" \
        --num_samples $NUM_SAMPLES \
        --batch_size $BATCH_SIZE \
        --num_candidates $K \
        --init_random_samples $INIT_RANDOM \
        --deno_steps $DENO_STEPS
    echo ""
done

echo "=== Collecting results ==="
python3 -c "
import json
from pathlib import Path
base = '$BASE_OUT'
results = []
for k in [2,3,4,5,6,7,8]:
    p = Path(f'{base}_K{k}/final_metrics.json')
    if p.exists():
        d = json.load(open(p))
        results.append({'num_candidates': k, 'FID': d['FID'], 'IS': d['IS']})
    else:
        results.append({'num_candidates': k, 'FID': None, 'IS': None})
print(json.dumps(results, indent=2))
print()
for r in results:
    fid = f\"{r['FID']:.2f}\" if r['FID'] is not None else 'N/A'
    isv = f\"{r['IS']:.2f}\" if r['IS'] is not None else 'N/A'
    print(f\"K={r['num_candidates']}: FID={fid}, IS={isv}\")
"
