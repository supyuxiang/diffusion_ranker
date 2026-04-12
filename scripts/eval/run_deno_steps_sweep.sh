#!/bin/bash
# Sweep denoising steps (5, 10, 15, 20) for Random Search FID
# Run from project1/idm directory

set -e
cd /data/home/yxfeng/project1/idm

DIT_PATH="${DIT_PATH:-/data/home/yxfeng/data/models/dit_model}"
IMAGENET_VAL="${IMAGENET_VAL:-/data/yxfeng/datasets/imagenet/val}"
BASE_OUT="/data/yxfeng/eval_dit_random_search_deno_steps"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
INIT_RANDOM="${INIT_RANDOM:-512}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3}"
BATCH_SIZE=24

echo "=== deno_steps sweep: 5, 10, 15, 20 ==="
echo "num_samples=$NUM_SAMPLES, init_random_samples=$INIT_RANDOM, num_candidates=$NUM_CANDIDATES"
echo ""

for STEPS in 5 10 15 20; do
    OUT_DIR="${BASE_OUT}_S${STEPS}_num_candidates${NUM_CANDIDATES}"
    echo ">>> Running deno_steps=$STEPS -> $OUT_DIR"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 search_algoriths/dit_random_search_fid_reward.py \
        --dit_path "$DIT_PATH" \
        --imagenet_val "$IMAGENET_VAL" \
        --output_dir "$OUT_DIR" \
        --num_samples $NUM_SAMPLES \
        --batch_size $BATCH_SIZE \
        --num_candidates $NUM_CANDIDATES \
        --init_random_samples $INIT_RANDOM \
        --deno_steps $STEPS
    echo ""
done

echo "=== Collecting results ==="
python3 -c "
import json
from pathlib import Path
base = '$BASE_OUT'
results = []
for s in [5, 10, 15, 20]:
    p = Path(f'{base}_S{s}/final_metrics.json')
    if p.exists():
        d = json.load(open(p))
        results.append({'deno_steps': s, 'FID': d['FID'], 'IS': d['IS']})
    else:
        results.append({'deno_steps': s, 'FID': None, 'IS': None})
print(json.dumps(results, indent=2))
print()
for r in results:
    fid = f\"{r['FID']:.2f}\" if r['FID'] is not None else 'N/A'
    isv = f\"{r['IS']:.2f}\" if r['IS'] is not None else 'N/A'
    print(f\"steps={r['deno_steps']}: FID={fid}, IS={isv}\")
"
