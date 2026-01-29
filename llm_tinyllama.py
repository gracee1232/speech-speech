# llm_tinyllama.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)

def refine_text(text: str, target_lang: str) -> str:
    """
    Refine translated text using TinyLlama
    """

    prompt = f"""
You are a helpful assistant.
Rewrite the following text naturally in {target_lang}.
Keep the meaning the same.

Text:
{text}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean response
    return result.split("Answer:")[-1].strip()
