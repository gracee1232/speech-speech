import streamlit as st
import tempfile
import os
import sys
import time
from io import BytesIO
import base64

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Import all models
from asr_whisper import speech_to_text
from translate_nllb import NLLBTranslator
from tts_coqui import text_to_speech

# Page configuration
st.set_page_config(
    page_title="Speech Translation Pipeline",
    page_icon="🎤",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .step-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success {
        border-left-color: #2ca02c;
        background: #f0fff0;
    }
    .error {
        border-left-color: #d62728;
        background: #fff0f0;
    }
    .audio-player {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎤 Speech Translation Pipeline</h1>
        <p>Speak in English → Get translated speech output</p>
    </div>
    """, unsafe_allow_html=True)

    # Language selection
    col1, col2 = st.columns([2, 1])
    with col1:
        target_language = st.selectbox(
            "Select Target Language:",
            options=[
                ("French", "fra_Latn"),
                ("German", "deu_Latn"),
                ("Spanish", "spa_Latn"),
                ("Hindi", "hin_Deva"),
                ("Chinese", "zho_Hans"),
                ("Arabic", "arb_Arab"),
                ("Russian", "rus_Cyrl")
            ],
            format_func=lambda x: x[0]
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("🔄 Reset All", type="secondary"):
            st.rerun()

    # Audio input section
    st.markdown("---")
    st.header("🎙️ Step 1: Record Audio")
    
    # Audio recording
    audio_bytes = audio_recorder()
    
    if audio_bytes:
        st.success("✅ Audio recorded successfully!")
        
        # Save audio to temporary file with proper extension
        file_extension = '.wav'  # Default to WAV
        if hasattr(audio_bytes, 'name'):
            if audio_bytes.name.endswith('.mp3'):
                file_extension = '.mp3'
            elif audio_bytes.name.endswith('.m4a'):
                file_extension = '.m4a'
            elif audio_bytes.name.endswith('.webm'):
                file_extension = '.webm'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(audio_bytes)
            audio_path = tmp_file.name
        
        # Verify file was created
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            st.info(f"📊 Audio file: {os.path.basename(audio_path)} ({file_size:,} bytes)")
        else:
            st.error("❌ Failed to create audio file")
            return
        
        # Play recorded audio
        st.audio(audio_bytes, format=f'audio/{file_extension[1:]}')
        
        # Process button
        if st.button("🚀 Start Translation Pipeline", type="primary", use_container_width=True):
            run_translation_pipeline(audio_path, target_language[1])
            # Cleanup will be handled inside run_translation_pipeline

def audio_recorder():
    """Improved audio recorder with better format handling"""
    audio_file = st.file_uploader(
        "📁 Upload audio file (WAV, MP3, M4A):",
        type=['wav', 'mp3', 'm4a', 'webm', 'ogg'],
        help="Upload an audio file or use the microphone button below"
    )
    
    if audio_file:
        st.success(f"✅ File uploaded: {audio_file.name} ({audio_file.size} bytes)")
        return audio_file.read()
    
    # Microphone recording
    st.write("🎤 **Or record using microphone:**")
    recorded_audio = st.audio_input("Record your voice (speak clearly for 3-10 seconds)")
    
    if recorded_audio:
        st.success("✅ Audio recorded successfully!")
        return recorded_audio.read()
    
    # Add instructions
    st.info("""
    💡 **Tips for better transcription:**
    - Speak clearly and at a moderate pace
    - Record in a quiet environment
    - Ensure microphone is close to your mouth
    - Record for at least 3 seconds
    - Use English language only
    """)
    
    return None

def run_translation_pipeline(audio_path, target_lang):
    """Run the complete translation pipeline"""
    
    # Store original path for cleanup
    original_audio_path = audio_path
    
    # Step 1: ASR
    st.markdown("---")
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader("🔍 Step 1: Speech to Text (ASR)")
    
    with st.spinner("🎧 Transcribing audio..."):
        try:
            start_time = time.time()
            
            # Add debug information
            st.write(f"📁 Processing file: {os.path.basename(audio_path)}")
            
            source_text = speech_to_text(audio_path)
            asr_time = time.time() - start_time
            
            if source_text.strip():
                st.markdown(f'<div class="step-card success">', unsafe_allow_html=True)
                st.success(f"✅ Transcription completed in {asr_time:.2f}s")
                st.info(f"**Recognized Text:** {source_text}")
                
                # Add confidence indicator
                if len(source_text.split()) > 2:
                    st.success("🎯 Good quality transcription detected")
                else:
                    st.warning("⚠️ Short transcription - consider speaking more clearly")
                    
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="step-card error">', unsafe_allow_html=True)
                st.error("❌ No speech detected in the audio")
                
                # Provide troubleshooting tips
                st.warning("""
                🔧 **Troubleshooting Tips:**
                - Check if your microphone is working
                - Try speaking louder and more clearly
                - Ensure you're speaking in English
                - Record for at least 3-5 seconds
                - Try uploading a different audio file
                """)
                
                st.markdown('</div>', unsafe_allow_html=True)
                cleanup_files(original_audio_path)
                return
                
        except Exception as e:
            st.markdown('<div class="step-card error">', unsafe_allow_html=True)
            st.error(f"❌ ASR Error: {str(e)}")
            st.code(f"Full error: {repr(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            cleanup_files(original_audio_path)
            return
    
    # Step 2: Translation
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader("🌐 Step 2: Translation (MT)")
    
    with st.spinner("🔄 Translating text..."):
        try:
            start_time = time.time()
            translator = NLLBTranslator(target_lang)
            translated_text = translator.translate(source_text)
            mt_time = time.time() - start_time
            
            st.markdown(f'<div class="step-card success">', unsafe_allow_html=True)
            st.success(f"✅ Translation completed in {mt_time:.2f}s")
            st.info(f"**Translated Text:** {translated_text}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown('<div class="step-card error">', unsafe_allow_html=True)
            st.error(f"❌ Translation Error: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            cleanup_files(original_audio_path)
            return
    
    # Step 3: TTS
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader("🔊 Step 3: Text to Speech (TTS)")
    
    with st.spinner("🗣️ Generating speech..."):
        try:
            start_time = time.time()
            output_audio = "translated_output.mp3"
            text_to_speech(translated_text, output_audio)
            tts_time = time.time() - start_time
            
            st.markdown(f'<div class="step-card success">', unsafe_allow_html=True)
            st.success(f"✅ Speech generation completed in {tts_time:.2f}s")
            
            # Display audio player
            if os.path.exists(output_audio):
                st.audio(output_audio, format='audio/mp3')
                
                # Download button
                with open(output_audio, "rb") as file:
                    st.download_button(
                        label="📥 Download Translated Audio",
                        data=file.read(),
                        file_name="translated_speech.mp3",
                        mime="audio/mpeg"
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Summary
            st.markdown("---")
            st.markdown("### 🎉 Translation Complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Time", f"{asr_time + mt_time + tts_time:.2f}s")
            with col2:
                st.metric("Source Language", "English")
                st.metric("Target Language", target_lang.split('_')[0].upper())
            
            # Cleanup
            cleanup_files(original_audio_path)
            if os.path.exists(output_audio):
                try:
                    os.unlink(output_audio)
                except:
                    pass
                
        except Exception as e:
            st.markdown('<div class="step-card error">', unsafe_allow_html=True)
            st.error(f"❌ TTS Error: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            cleanup_files(original_audio_path)

def cleanup_files(audio_path):
    """Clean up temporary files"""
    try:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        # Also cleanup converted files
        converted_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        if os.path.exists(converted_path):
            os.unlink(converted_path)
    except:
        pass

if __name__ == "__main__":
    main()
