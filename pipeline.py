# pipeline.py
import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from asr_whisper import speech_to_text
from translate_nllb import NLLBTranslator
from llm_tinyllama import refine_text
from tts_coqui import text_to_speech


def full_pipeline(audio_input, target_lang="fra_Latn"):
    import time
    
    print("STEP 1: Speech → Text (ASR)")
    start_time = time.time()
    source_text = speech_to_text(audio_input)
    asr_time = time.time() - start_time
    print(f"Recognized Text: {source_text}")
    print(f"ASR Time: {asr_time:.2f} seconds")

    print("\nSTEP 2: Translation (NLLB)")
    start_time = time.time()
    translator = NLLBTranslator(target_lang)
    translated_text = translator.translate(source_text)
    mt_time = time.time() - start_time
    print(f"Translated Text: {translated_text}")
    print(f"Translation Time: {mt_time:.2f} seconds")

    print("\nSTEP 3: LLM Refinement (TinyLlama)")
    start_time = time.time()
    refined_text = refine_text(translated_text, target_lang)
    llm_time = time.time() - start_time
    print(f"LLM Output: {refined_text}")
    print(f"LLM Refinement Time: {llm_time:.2f} seconds")

    print("\nSTEP 4: Text → Speech (Coqui TTS)")
    start_time = time.time()
    output_audio = "final_output.wav"
    text_to_speech(refined_text, output_audio)
    tts_time = time.time() - start_time
    print(f"Generated Audio: {output_audio}")
    print(f"TTS Time: {tts_time:.2f} seconds")
    
    total_time = asr_time + mt_time + llm_time + tts_time
    print(f"\n⏱️  Total Pipeline Time: {total_time:.2f} seconds")
    print(f"   - ASR: {asr_time:.2f}s ({asr_time/total_time*100:.1f}%)")
    print(f"   - Translation: {mt_time:.2f}s ({mt_time/total_time*100:.1f}%)")
    print(f"   - LLM Refinement: {llm_time:.2f}s ({llm_time/total_time*100:.1f}%)")
    print(f"   - TTS: {tts_time:.2f}s ({tts_time/total_time*100:.1f}%)")

    return refined_text, output_audio


if __name__ == "__main__":
    full_pipeline(
        audio_input="sample.wav",
        target_lang="deu_Latn"  # EN → DE
    )

