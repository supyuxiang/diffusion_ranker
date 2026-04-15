"""
预计算 ImageNet 类对应的文本 prompt 表征（hidden 或 vocab logits），供 ranker 侧与候选表征对齐。

运行示例::

    python -m ranker.pre_compute_prompt_repr
"""

from __future__ import annotations

import sys
from pathlib import Path
import gc
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from utils import labels_1k


def get_tokenizer(model_dir: str | Path) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def class2prompt(class_id: int, tokenizer: AutoTokenizer) -> str:
    class_name = labels_1k[class_id]
    prompt = f"Describe the image of class {class_name} in detail."
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=False,
    )


def _pool_seq(
    x: Tensor,
    attention_mask: Tensor,
    mode: str,
) -> Tensor:
    """x: (B, seq, D); attention_mask: (B, seq) with 1 for real tokens."""
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype) # b,seq,1
        mean_x = (x * mask).sum(dim=1) # b,D
        num_tokens = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=x.dtype) # b,1
        return mean_x / num_tokens
    if mode == "last":
        b = x.shape[0]
        last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0).long()
        return x[torch.arange(b, device=x.device), last_idx]
    raise ValueError(f"Invalid pooling mode: {mode}")


def get_repr(
    class_ids: list[int],
    model_dir: str | Path,
    tokenize_type: str,
    pooling_mode: str,
    *,
    batch_size: int = 8,
    max_length: int = 1024,
    device: torch.device | str | None = None,
) -> Tensor:
    assert tokenize_type in ("LLM_logits","LLM_hidden_states")
    assert pooling_mode in ("mean", "last")

    model_dir = Path(model_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    tokenizer = get_tokenizer(model_dir)
    formatted_prompt_ls = [class2prompt(cid, tokenizer) for cid in class_ids]

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    outs: list[Tensor] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(formatted_prompt_ls), batch_size),desc="Computing prompt representations"):
            batch_texts = formatted_prompt_ls[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            if tokenize_type == "LLM_logits":
                seq = model(**enc).logits # b,sq_len,vocab_size
            elif tokenize_type == "LLM_hidden_states":
                o = model(
                    **enc,
                    output_hidden_states=True,
                )
                hiddens = o.hidden_states
                layer_ratio = 0.6
                layer_idx = round(layer_ratio * model.config.num_hidden_layers)
                seq = hiddens[layer_idx].clone()  # (B, seq, hidden_dim)
                del hiddens,o
                gc.collect()
                torch.cuda.empty_cache()

            pooled = _pool_seq(seq, enc["attention_mask"], pooling_mode) 
            outs.append(pooled.float().cpu())
            del enc,seq,pooled
            gc.collect()
            torch.cuda.empty_cache()

    return torch.cat(outs, dim=0)


def main_logits() -> None:
    save_path = ROOT_DIR / "ranker" / "prompt_repr_mean_pooling/from_logits.pt"
    model_dir = Path("/root/autodl-tmp/Qwen2.5-7B-Instruct")
    tokenize_type = "LLM_logits"
    pooling_mode = "mean"
    class_ids = list(range(1000))

    prompt_repr = get_repr(
        class_ids,
        model_dir,
        tokenize_type,
        pooling_mode,
        batch_size=4,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_repr, save_path)
    print(f"Prompt representations saved to {save_path} shape={tuple(prompt_repr.shape)}")

def main_hidden_states():
    save_path = ROOT_DIR / "ranker" / "prompt_repr_mean_pooling/from_hidden_states_ratio0.6.pt"
    model_dir = Path("/root/autodl-tmp/Qwen2.5-7B-Instruct")
    tokenize_type = "LLM_hidden_states"
    pooling_mode = "mean"
    layer_ratio = 0.6
    class_ids = list(range(1000))
    prompt_repr = get_repr(
        class_ids,
        model_dir,
        tokenize_type,
        pooling_mode,
        batch_size=4,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_repr, save_path)
    print(f"Prompt representations saved to {save_path} shape={tuple(prompt_repr.shape)}")


if __name__ == "__main__":
    main_logits()
    main_hidden_states()
