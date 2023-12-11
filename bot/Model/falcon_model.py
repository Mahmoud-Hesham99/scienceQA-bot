# !nvidia-smi
# !pip install -Uqqq pip --progress-bar off
# !pip install -qqq bitsandbytes==0.39.0 --progress-bar off
# !pip install -qqq torch==2.0.1 --progress-bar off
# !pip install -qqq -U git+https://github.com/huggingface/transformers.git@e03a9cc --progress-bar off
# !pip install -qqq -U git+https://github.com/huggingface/peft.git@42a184f --progress-bar off
# !pip install -qqq -U git+https://github.com/huggingface/accelerate.git@c9fbb71 --progress-bar off
# !pip install -qqq datasets==2.12.0 --progress-bar off
# !pip install -qqq loralib==0.1.1 --progress-bar off
# !pip install -qqq einops==0.6.1 --progress-bar off


import os

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "tiiuae/falcon-7b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

def generate_response(question: str) -> str:
    prompt = f"""
    {question}
    """.strip()
    encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response[1+len(question):].strip()