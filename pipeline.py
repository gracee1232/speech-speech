# pipeline.py
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from asr_whisper import speech_to_text
from translate_nllb import NLLBTranslator
from tts_coqui import text_to_speech

def full_pipeline(audio_input, target_lang="fra_Latn"):
    print("STEP 1: Speech -> Text (ASR)")
    source_text = speech_to_text(audio_input)
    print("Recognized Text:", source_text)

    print("\nSTEP 2: Translation (MT)")
    translator = NLLBTranslator(target_lang)
    translated_text = translator.translate(source_text)
    print("Translated Text:", translated_text)

    print("\nSTEP 3: Text -> Audio (TTS)")
    output_audio = "final_output.mp3"
    text_to_speech(translated_text, output_audio)
    print("Generated Audio:", output_audio)

    return translated_text, output_audio

if __name__ == "__main__":
    full_pipeline("sample.wav", target_lang="deu_Latn")  # EN -> DE example
