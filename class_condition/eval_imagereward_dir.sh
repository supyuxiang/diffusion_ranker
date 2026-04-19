#!/usr/bin/env bash
set -euo pipefail

# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/random_search_clipscore_50k_cfg1.0_deno10_search32/num_candidates_2"
# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/random_search_clipscore_50k_cfg1.0_deno10_search32/num_candidates_4"
# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/random_search_clipscore_50k_cfg1.0_deno10_search32/num_candidates_8"
# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/random_search_imagereward_50k_cfg1.0_deno10_search32/num_candidates_2"
# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/random_search_imagereward_50k_cfg1.0_deno10_search32/num_candidates_4"
# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/random_search_imagereward_50k_cfg1.0_deno10_search32/num_candidates_8"
# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/dit_random_search_fid_reward/num_candidates_2_cfg1.0_steps10/best_images"
# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/dit_random_search_fid_reward/num_candidates_4_cfg1.0_steps10/best_images"
# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/dit_random_search_fid_reward/num_candidates_8_cfg1.0_steps10/best_images"
# bash /root/diffusion_ranker/class-condition/eval_imagereward_dir.sh "/root/autodl-tmp/outputs/dit_scaling_with_different_steps_cfg1.0/deno_step_10_rollout"



# 用法:
#   bash eval_imagereward_dir.sh <TARGET_DIR> [OUT_JSON]

TARGET_DIR="${1:?need TARGET_DIR}"
OUT_JSON="${2:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python - << 'PY' "$TARGET_DIR" "$OUT_JSON"
import json
import sys
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

target_dir = Path(sys.argv[1]).expanduser().resolve()
out_json_arg = sys.argv[2].strip()

root = Path("/root/diffusion_ranker")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from search.reward import load_imagereward

exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])

best_images_dir = target_dir / "best_images"
if best_images_dir.is_dir() and len(list_images(best_images_dir)) > 0:
    image_dir = best_images_dir
else:
    image_dir = target_dir

paths = list_images(image_dir)
if not paths:
    raise FileNotFoundError(f"No images found under {image_dir}")

labels_path = Path("/root/diffusion_ranker/imagenet-simple-labels.json")
labels = None
if labels_path.is_file():
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

default_prompt = "A photo of a human face."

def infer_prompt(p: Path) -> str:
    last = p.stem.split("_")[-1]
    if labels is not None and last.isdigit():
        idx = int(last)
        if 0 <= idx < len(labels):
            return f"A photo of a {labels[idx]}."
    return default_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rm = load_imagereward(device)

vals = []
for p in tqdm(paths, desc=f"ImageReward @ {image_dir.name}"):
    img = Image.open(p).convert("RGB")
    vals.append(float(rm.score(infer_prompt(p), img)))

t = torch.tensor(vals, dtype=torch.float32)
res = {
    "target_dir": str(target_dir),
    "image_dir": str(image_dir),
    "num_images": int(len(paths)),
    "imagereward_mean": float(t.mean().item()),
    "imagereward_std": float(t.std(unbiased=False).item()),
}

out_json = Path(out_json_arg).expanduser().resolve() if out_json_arg else (image_dir / "imagereward_only.json")
out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
print(f"[saved] {out_json}")
PY
