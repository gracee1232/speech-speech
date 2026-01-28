# tts_coqui.py
try:
    from TTS.api import TTS
    import soundfile as sf
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: TTS not installed. Using gTTS fallback.")
    from gtts import gTTS
    import os

def text_to_speech(text: str, output_file: str = "output.wav"):
    if TTS_AVAILABLE:
        print("Loading Coqui TTS model...")
        try:
            # Use a simpler, more reliable model
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            wav = tts.tts(text)
            sf.write(output_file, wav, 22050)
            return output_file
        except Exception as e:
            print(f"Coqui TTS failed: {e}. Using gTTS fallback.")
    
    # gTTS fallback
    print("Using gTTS fallback...")
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_file = output_file.replace('.wav', '.mp3')
    tts.save(mp3_file)
    return mp3_file
