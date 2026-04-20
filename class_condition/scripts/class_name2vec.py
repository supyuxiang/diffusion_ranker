from pathlib import Path
import sys
import json
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import labels_1k



vecterizer_ls = [
    'one-hot',
    'bert',
    'clip',
    'llm_tokenizer',
    'llm_hidden_state',
]




def one_hot() -> torch.Tensor:
    idx = torch.arange(len(labels_1k), dtype=torch.long)
    vec = F.one_hot(idx, num_classes=len(labels_1k)).float()
    return vec  # (N, N), e.g. (1000, 1000)



@torch.inference_mode()
def bert() -> torch.Tensor:
    bert_path = Path("/root/autodl-tmp/bert-large-cased")
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(bert_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(bert_path), local_files_only=True).to(device).eval()

    texts = [str(x) for x in labels_1k]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=16, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)
    cls_vec = out.last_hidden_state[:, 0, :].float().cpu()
    return cls_vec  # (N, H_bert), e.g. (1000, 1024) for bert-large-cased


@torch.inference_mode()
def clip() -> torch.Tensor:
    from transformers import CLIPModel, CLIPTokenizer
    clip_path = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(str(clip_path), local_files_only=True)
    model = CLIPModel.from_pretrained(str(clip_path), local_files_only=True).to(device).eval()

    texts = [str(name) for name in labels_1k]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    text_out = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    pooled = text_out.pooler_output
    text_feat = model.text_projection(pooled)
    text_feat = text_feat.float().cpu()
    return text_feat  # (N, D_clip), e.g. (1000, 768)



@torch.inference_mode()
def llm_tokenizer() -> torch.Tensor:
    llm_path = '/root/autodl-tmp/Qwen2.5-7B-Instruct'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    out = tokenizer(
        [str(x) for x in labels_1k],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    return out["input_ids"].long().cpu()  # (N, L_tok)


@torch.inference_mode()
def llm_hidden_state(layer_ratio: float = 0.6) -> torch.Tensor:
    llm_path = '/root/autodl-tmp/Qwen2.5-7B-Instruct'
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(llm_path)
    model.eval()

    layer_idx = max(round(model.config.num_hidden_layers * layer_ratio),1)

    prompt_ls = [
        f'label:{class_name}'
        for class_name in labels_1k
    ]

    formatted_prompt_ls = [
        [{'role':'user','content':prompt}]
        for prompt in prompt_ls
    ]

    formatted_prompt_ls = [
        tokenizer.apply_chat_template(prompt,tokenize=False)
        for prompt in formatted_prompt_ls
    ]

    inputs = tokenizer(
        formatted_prompt_ls,
        padding=True,
        truncation=True,
        max_length=10,
        return_tensors='pt'
    )
    inputs = {k:v.to(model.device) for k,v in inputs.items()}
    out = model(**inputs,output_hidden_states=True)
    hidden_states = out.hidden_states[layer_idx].float().cpu()
    return hidden_states  # (N, L, H_llm), e.g. (1000, <=10, 3584)
    

def check():
    print(one_hot().shape) # 1000,1000
    print(bert().shape) # 1000,H_bert
    print(clip().shape) # 1000,768
    print(llm_tokenizer().shape) # 1000,L_tok
    print(llm_hidden_state().shape) # 1000,10,3584



def main():
    for vecterizer in vecterizer_ls:
        vec = eval(f"{vecterizer}()")
        save_path = Path(f'/root/diffusion_ranker/class_condition/class_name_vecs/class_name2vec_{vecterizer}.pt')
        torch.save(vec, save_path)

if __name__ == "__main__":
    check()