import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

path = "/data/yxfeng/models/qwen/Qwen3-4B"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    path,
    quantization_config=quantization_config,
    device_map='cuda',
)
tokenizer = AutoTokenizer.from_pretrained(path)

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

result = pipe("Hello, how are you?",max_new_tokens=100)
print(result)