# translate_nllb.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Global model loading
print("Loading NLLB Model...")
model_name = "facebook/nllb-200-distilled-600M"
try:
    GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    GLOBAL_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    GLOBAL_MODEL = GLOBAL_MODEL.to(device)
    print(f"Using device: {device}")
except Exception as e:
    print(f"Failed to load NLLB model: {e}")
    GLOBAL_MODEL = None
    GLOBAL_TOKENIZER = None

class NLLBTranslator:
    def __init__(self, target_lang="fra_Latn", source_lang="eng_Latn"):  # English → French
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate(self, text: str) -> str:
        if GLOBAL_MODEL is None:
            return text
            
        # Set source language
        GLOBAL_TOKENIZER.src_lang = self.source_lang
        
        inputs = GLOBAL_TOKENIZER(text, return_tensors="pt")
        # Move inputs to same device as model
        inputs = {k: v.to(GLOBAL_MODEL.device) for k, v in inputs.items()}
        
        # Get target language token ID
        target_lang_id = GLOBAL_TOKENIZER.convert_tokens_to_ids(self.target_lang)
        
        generated_tokens = GLOBAL_MODEL.generate(
            **inputs,
            forced_bos_token_id=target_lang_id
        )
        return GLOBAL_TOKENIZER.batch_decode(generated_tokens, skip_special_tokens=True)[0]
