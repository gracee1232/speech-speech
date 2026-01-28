# translate_nllb.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class NLLBTranslator:
    def __init__(self, target_lang="fra_Latn", source_lang="eng_Latn"):  # English → French
        self.model_name = "facebook/nllb-200-distilled-600M"
        print("Loading NLLB Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        print(f"Using device: {device}")
        
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate(self, text: str) -> str:
        # Set source language
        self.tokenizer.src_lang = self.source_lang
        
        inputs = self.tokenizer(text, return_tensors="pt")
        # Move inputs to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
