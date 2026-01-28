import numpy as np
import wave
import tempfile
import os

def create_test_audio():
    """Create a simple test audio with speech-like patterns"""
    sample_rate = 16000
    duration = 3  # 3 seconds
    frequency = 440  # A4 note
    
    # Generate a simple tone
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a more complex signal (simulating speech)
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Add some modulation to make it more speech-like
    envelope = np.exp(-t * 0.5)  # Decay envelope
    signal = signal * envelope
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(signal))
    signal = signal + noise
    
    # Normalize to 16-bit PCM
    signal = signal / np.max(np.abs(signal)) * 0.8
    signal = (signal * 32767).astype(np.int16)
    
    # Create WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal.tobytes())
    
    return temp_file.name

if __name__ == "__main__":
    # Test the ASR with generated audio
    from asr_whisper import speech_to_text
    
    print("Creating test audio...")
    audio_path = create_test_audio()
    print(f"Test audio created: {audio_path}")
    print(f"File size: {os.path.getsize(audio_path)} bytes")
    
    print("Testing ASR...")
    result = speech_to_text(audio_path)
    print(f"ASR Result: '{result}'")
    
    # Cleanup
    os.unlink(audio_path)
