#!/usr/bin/env python3
"""
用 ``DitRolloutReprCapture`` 保存的 ``.pt``（``batches[*]['repr']``）跑通
:class:`~ranker.model.DiffusionRanker_Transformer`（与 ``ranker/train.py`` listwise 一致）。

用法::

    python3 -m ranker.test_repr_ranker
    python3 -m ranker.test_repr_ranker --repr_pt /path/to/repr.pt --pointwise_batch 32 --listwise_k 8


cd /root/diffusion_ranker
python3 -m ranker.test_repr_ranker
# 可选
python3 -m ranker.test_repr_ranker \
  --repr_pt /root/autodl-tmp/repr_candidate_deno10_cfg1.0/idx0.pt \
  --batch_index 0 \
  --pointwise_batch 32 \
  --listwise_k 8 \
  --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ranker.model import DiffusionRanker_Transformer
from ranker.repr_training_common import forward_listwise_transformer_ranker


_DEFAULT_REPR_PT = Path("/root/autodl-tmp/repr_candidate_deno10_cfg1.0/idx0.pt")


def repr_dict_to_tokens_per_sample(repr_dict: dict, sample_idx: int) -> torch.Tensor:
    """单张图：按 key 排序，每个 (s,r) 用 mean、last 两个 token，形状 ``(N_tokens, H)``。"""
    keys = sorted(repr_dict.keys())
    parts: list[torch.Tensor] = []
    for k in keys:
        parts.append(repr_dict[k]["mean"][sample_idx].float())
        parts.append(repr_dict[k]["last"][sample_idx].float())
    return torch.stack(parts, dim=0)


def repr_dict_to_pointwise_batch(repr_dict: dict, *, max_samples: int | None = None) -> torch.Tensor:
    """整 batch：``(B, N_tokens, H)``。"""
    keys = sorted(repr_dict.keys())
    chunks: list[torch.Tensor] = []
    for k in keys:
        m = repr_dict[k]["mean"].float()
        l = repr_dict[k]["last"].float()
        chunks.append(torch.stack([m, l], dim=1))
    x = torch.cat(chunks, dim=1)
    if max_samples is not None and x.size(0) > max_samples:
        x = x[:max_samples]
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repr_pt", type=Path, default=_DEFAULT_REPR_PT)
    p.add_argument("--batch_index", type=int, default=0, help="batches[i] 的下标")
    p.add_argument("--pointwise_batch", type=int, default=32, help="pointwise 最多用多少条样本")
    p.add_argument("--listwise_k", type=int, default=8, help="listwise 候选数 K")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    path = args.repr_pt.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"missing repr file: {path}")

    obj = torch.load(path, map_location="cpu", weights_only=False)
    batches = obj["batches"]
    meta = obj.get("meta", {})
    bi = args.batch_index
    if bi < 0 or bi >= len(batches):
        raise SystemExit(f"batch_index {bi} out of range [0, {len(batches) - 1}]")

    rec = batches[bi]["repr"]
    x_pw = repr_dict_to_pointwise_batch(rec, max_samples=args.pointwise_batch)
    h = x_pw.shape[-1]
    n_tok = x_pw.shape[1]
    print(f"loaded {path.name} batch[{bi}]: pointwise input {tuple(x_pw.shape)} (N_tokens={n_tok}, H={h})")
    if meta:
        print(f"  meta: steps={meta.get('target_step_indices')}, layers={len(meta.get('layer_ratios', []))}")

    dev = torch.device(args.device)
    # 与 train.py 一致：DiffusionRanker_Transformer，候选 repr 先 mean pool 再 (1,K,H)
    k = min(args.listwise_k, x_pw.size(0))
    tokens = [repr_dict_to_tokens_per_sample(rec, i) for i in range(k)]
    x_ls = torch.stack(tokens, dim=0).to(dev)
    assert x_ls.shape == (k, n_tok, h), x_ls.shape

    ls = DiffusionRanker_Transformer(
        input_dim_candidate=h,
        input_dim_prompt=h,
        hidden_dim=args.proj_dim,
        num_heads=8,
        dropout=0.1,
        num_layers=2,
        half=False,
    ).to(dev)
    with torch.inference_mode():
        logits = forward_listwise_transformer_ranker(ls, x_ls, class_id=0, prompt_table=None)
    probs = torch.softmax(logits, dim=-1)
    print(f"listwise logits: shape={tuple(logits.shape)} probs sum={probs.sum().item():.4f}")
    print(f"  argmax candidate idx={logits.argmax(dim=-1).item()}")
    print("ok")


if __name__ == "__main__":
    main()
