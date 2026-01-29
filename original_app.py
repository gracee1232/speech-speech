import streamlit as st
import tempfile
import os
import sys
import time
from io import BytesIO

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
    page_title="Speech Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .step-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border-left-color: #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .audio-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Speech Translator</h1>
        <p>Speak in English • Translate Instantly • Hear Result</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.write("")  # Spacer
    
    with col2:
        # Language selection
        target_lang = st.selectbox(
            "🎯 Translate to:",
            options=[
                ("French", "fra_Latn"),
                ("German", "deu_Latn"),
                ("Spanish", "spa_Latn"),
                ("Hindi", "hin_Deva"),
                ("Chinese", "zho_Hans"),
                ("Arabic", "arb_Arab"),
                ("Russian", "rus_Cyrl")
            ],
            format_func=lambda x: x[0],
            index=0
        )
        
        # Audio input section
        st.markdown("---")
        st.subheader("🎤 Step 1: Record Your Voice")
        
        # Simple recording interface
        recorded_audio = st.audio_input("🎙️ Click to record (speak clearly for 3-10 seconds)")
        
        if recorded_audio:
            st.success("✅ Recording complete! Processing...")
            audio_bytes = recorded_audio.read()
            process_audio(audio_bytes, target_lang[1])
    
    with col3:
        st.write("")  # Spacer

def process_audio(audio_bytes, target_lang):
    """Process audio through the complete pipeline"""
    
    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        audio_path = tmp_file.name
    
    try:
        # Step 1: ASR
        with st.spinner("🎧 Listening..."):
            source_text = speech_to_text(audio_path)
            
            if not source_text.strip():
                st.error("❌ No speech detected. Please try again.")
                return
        
        # Step 2: Translation
        with st.spinner("🌐 Translating..."):
            translator = NLLBTranslator(target_lang)
            translated_text = translator.translate(source_text)
        
        # Step 3: TTS
        with st.spinner("🔊 Generating speech..."):
            output_audio = "translated_speech.mp3"
            text_to_speech(translated_text, output_audio)
        
        # Display results
        st.markdown("---")
        st.subheader("📤 Translation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="step-box">
                <h4>🇺🇸 English (Input)</h4>
                <p style="font-size: 18px; margin: 10px 0;">{}</p>
            </div>
            """.format(source_text), unsafe_allow_html=True)
        
        with col2:
            lang_name = target_lang.split('_')[0]
            st.markdown("""
            <div class="step-box">
                <h4>🌍 {} (Output)</h4>
                <p style="font-size: 18px; margin: 10px 0;">{}</p>
            </div>
            """.format(lang_name.title(), translated_text), unsafe_allow_html=True)
        
        # Audio player
        if os.path.exists(output_audio):
            st.markdown("---")
            st.subheader("🔊 Audio Output")
            st.audio(output_audio, format='audio/mp3')
            
            # Download button
            with open(output_audio, "rb") as file:
                st.download_button(
                    label="📥 Download Audio",
                    data=file.read(),
                    file_name=f"translated_speech_{lang_name}.mp3",
                    mime="audio/mpeg"
                )
        
        # Success message
        st.success("🎉 Translation complete!")
        
        # Cleanup
        os.unlink(output_audio)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Please try recording again")
    
    finally:
        # Cleanup temp file
        if os.path.exists(audio_path):
            os.unlink(audio_path)

# Instructions section
with st.expander("💡 How to use"):
    st.markdown("""
    1. **Select Language**: Choose your target language from the dropdown
    2. **Record Audio**: Click the record button and speak clearly (3-10 seconds)
    3. **Wait for Processing**: The app will automatically:
       - Convert your speech to text
       - Translate to target language  
       - Generate speech in target language
    4. **See Results**: View English input and translated output side-by-side
    5. **Listen & Download**: Play the translated audio or download it
    
    **Tips:**
    - Speak clearly and at moderate pace
    - Record in a quiet environment
    - Use English language for input
    - Keep recordings under 10 seconds
    """)

if __name__ == "__main__":
    main()
