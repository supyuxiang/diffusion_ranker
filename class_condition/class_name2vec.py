from pathlib import Path
import sys
import json
import torch
import torch.nn.functional as F

from utils import labels_1k


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


vecterizer_ls = [
    'one-hot',
    'bert',
    'clip',
    'llm_tokenizer',
    'llm_hidden_state',
]



def one_hot():
    vec = F.one_hot(torch.tensor(labels_1k), num_classes=len(labels_1k))
    return vec


@torch.inference_mode()
def bert():
    bert_path = ''
    from transformers import AutoTokenizer, AutoModelForMaskedLM


@torch.inference_mode()
def clip():
    clip_path = ''




@torch.inference_mode()
def llm_tokenizer():
    llm_path = ''
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    return tokenizer.encode(class_name_ls, add_special_tokens=False)


@torch.inference_mode()
def llm_hidden_state(layer_ratio:float=0.6):
    llm_path = ''
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    


def main():
    for vecterizer in vecterizer_ls:
        vec = eval(f"{vecterizer}()")
        save_path = Path(f'/root/diffusion_ranker/class_condition/class_name_vecs/class_name2vec_{vecterizer}.pt')
        torch.save(vec, save_path)

if __name__ == "__main__":
    main()