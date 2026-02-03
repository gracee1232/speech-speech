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

# Import models
from asr_whisper import speech_to_text
from translate_nllb import NLLBTranslator
from tts_coqui import text_to_speech

# Page configuration
st.set_page_config(
    page_title="Fast Speech Translator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, clean design
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Main container */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Language selector */
    .language-selector {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
    
    /* Audio input area */
    .audio-area {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 2px dashed #667eea;
        transition: all 0.3s ease;
    }
    
    .audio-area:hover {
        border-color: #764ba2;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .input-card {
        border-left-color: #28a745;
    }
    
    .output-card {
        border-left-color: #ffc107;
    }
    
    /* Audio player */
    .audio-player {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    /* Progress indicators */
    .progress-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        color: #155724;
    }
    
    /* Smooth animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Custom button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header fade-in">
        <h1>⚡ Fast Speech Translator</h1>
        <p style="margin: 0; font-size: 1.1rem;">Quick • Accurate • Simple</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    container = st.container()
    
    with container:
        # Language selection
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div class="language-selector fade-in">
                <h3 style="margin: 0; color: #495057;">🎯 Choose Target Language</h3>
            </div>
            """, unsafe_allow_html=True)
            
            target_lang = st.selectbox(
                "",
                options=[
                    ("🇫🇷 French", "fra_Latn"),
                    ("🇩🇪 German", "deu_Latn"),
                    ("🇪🇸 Spanish", "spa_Latn"),
                    ("🇮🇳 Hindi", "hin_Deva"),
                    ("🇨🇳 Chinese", "zho_Hans"),
                    ("🇸🇦 Arabic", "arb_Arab"),
                    ("🇷🇺 Russian", "rus_Cyrl")
                ],
                format_func=lambda x: x[0],
                index=0,
                key="language_selector"
            )
            
            # Audio input area
            st.markdown("""
            <div class="audio-area fade-in">
                <h3 style="margin: 0; color: #495057;">🎤 Record Your Voice</h3>
                <p style="margin: 0.5rem 0; color: #6c757d;">Click below and speak clearly (3-10 seconds)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Audio input
            recorded_audio = st.audio_input(
                "🎙️ Tap to Record",
                key="audio_input"
            )
            
            if recorded_audio:
                st.markdown('<div class="fade-in">', unsafe_allow_html=True)
                st.success("✅ Recording complete! Processing...")
                audio_bytes = recorded_audio.read()
                process_audio_fast(audio_bytes, target_lang[1])
                st.markdown('</div>', unsafe_allow_html=True)

def process_audio_fast(audio_bytes, target_lang):
    """Fast audio processing without LLM"""
    
    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        audio_path = tmp_file.name
    
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        timings = {}
        
        # Step 1: ASR
        status_text.text("🎧 Listening to your speech...")
        start_time = time.time()
        source_text = speech_to_text(audio_path)
        timings['ASR'] = time.time() - start_time
        progress_bar.progress(25)
        
        if not source_text.strip():
            st.error("❌ No speech detected. Please try again.")
            return
        
        # Step 2: Translation
        status_text.text("🌐 Translating to target language...")
        start_time = time.time()
        translator = NLLBTranslator(target_lang)
        translated_text = translator.translate(source_text)
        timings['Translation'] = time.time() - start_time
        progress_bar.progress(75)
        
        # Step 3: TTS
        status_text.text("🔊 Generating translated speech...")
        start_time = time.time()
        output_audio = "translated_speech.mp3"
        text_to_speech(translated_text, output_audio)
        timings['TTS'] = time.time() - start_time
        progress_bar.progress(100)
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # Calculate total time
        total_time = sum(timings.values())
        
        # Success message
        st.markdown(f"""
        <div class="success-box fade-in">
            <h3 style="margin: 0;">🎉 Translation Complete!</h3>
            <p style="margin: 0.5rem 0;">⚡ Total time: {total_time:.1f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Results
        st.markdown("---")
        st.markdown("### 📤 Translation Results")
        
        # Input text
        st.markdown(f"""
        <div class="result-card input-card fade-in">
            <h4>🇺🇸 English (Input)</h4>
            <p style="font-size: 1.2rem; margin: 0.5rem 0; color: #495057;">{source_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Output text
        lang_name = target_lang.split('_')[0]
        st.markdown(f"""
        <div class="result-card output-card fade-in">
            <h4>🌍 {lang_name.title()} (Translation)</h4>
            <p style="font-size: 1.2rem; margin: 0.5rem 0; color: #495057;">{translated_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio player
        if os.path.exists(output_audio):
            st.markdown(f"""
            <div class="audio-player fade-in">
                <h4 style="margin: 0;">🔊 Translated Speech</h4>
                <p style="margin: 0.5rem 0; color: #6c757d;">Listen to the pronunciation</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.audio(output_audio, format='audio/mp3')
            
            # Download button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with open(output_audio, "rb") as file:
                    st.download_button(
                        label="📥 Download Audio",
                        data=file.read(),
                        file_name=f"translated_speech_{lang_name}.mp3",
                        mime="audio/mpeg",
                        use_container_width=True
                    )
            
            # Cleanup
            os.unlink(output_audio)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Please try recording again")
    
    finally:
        # Cleanup temp file
        if os.path.exists(audio_path):
            os.unlink(audio_path)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 2rem; padding: 1rem; color: #6c757d;'>
    <p>Made with ❤️ • Fast & Accurate Translation</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
