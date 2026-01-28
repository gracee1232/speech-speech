# asr_whisper.py
import whisper
import torch
import os
import subprocess
import tempfile
import speech_recognition as sr

def convert_to_wav(input_path: str) -> str:
    """Convert audio to WAV format using ffmpeg"""
    try:
        output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
        subprocess.run([
            'ffmpeg', '-i', input_path, '-ar', '16000', '-ac', '1', 
            '-c:a', 'pcm_s16le', '-y', output_path
        ], check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError:
        return input_path
    except FileNotFoundError:
        print("FFmpeg not found, using original file")
        return input_path

def fallback_asr(audio_path: str) -> str:
    """Fallback ASR using speech_recognition library"""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            
        # Try Google Speech Recognition
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"Google ASR result: '{text}'")
            return text
        except:
            # Try Sphinx as offline fallback
            try:
                text = recognizer.recognize_sphinx(audio_data)
                print(f"Sphinx ASR result: '{text}'")
                return text
            except:
                return ""
                
    except Exception as e:
        print(f"Fallback ASR error: {e}")
        return ""

def speech_to_text(audio_path: str) -> str:
    print("Loading Whisper Tiny model...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = whisper.load_model("tiny", device=device)
    except Exception as e:
        print(f"Failed to load Whisper: {e}")
        print("Using fallback ASR...")
        return fallback_asr(audio_path)
    
    try:
        # Convert to WAV if needed
        if not audio_path.endswith('.wav'):
            print("Converting audio to WAV format...")
            audio_path = convert_to_wav(audio_path)
        
        # Check if file exists and has content
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return fallback_asr(audio_path)
        
        file_size = os.path.getsize(audio_path)
        print(f"Audio file size: {file_size} bytes")
        
        if file_size < 1000:  # Less than 1KB
            print("Audio file too small, likely empty")
            return ""
        
        # Try transcription with different parameters
        print("Starting transcription...")
        print(f"Audio file path: {audio_path}")
        print(f"File exists: {os.path.exists(audio_path)}")
        print(f"File readable: {os.access(audio_path, os.R_OK)}")
        
        try:
            result = model.transcribe(
                audio_path, 
                fp16=False,  # Force CPU mode
                language="en",
                task="transcribe",
                verbose=True  # Enable verbose output
            )
            
            text = result["text"].strip()
            print(f"Raw transcription result: '{text}'")
            
        except Exception as whisper_error:
            print(f"Whisper transcription failed: {whisper_error}")
            print("Trying fallback ASR...")
            return fallback_asr(audio_path)
        
        # Clean up the result
        if text:
            # Remove common filler words and clean up
            filler_words = ["um", "uh", "like", "you know", "actually", "basically"]
            for filler in filler_words:
                text = text.replace(filler, "").strip()
            
            # Remove extra spaces and punctuation
            text = " ".join(text.split())
            text = text.strip(".,!?;:")
        
        print(f"Cleaned transcription: '{text}'")
        return text
        
    except Exception as e:
        print(f"ASR Error: {e}")
        print("Using fallback ASR...")
        return fallback_asr(audio_path)
    finally:
        # Cleanup converted file
        if '_converted.wav' in audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
