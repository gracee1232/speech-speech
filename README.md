---
title: Fast Speech Translator
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Speech-to-Speech Translation Pipeline

A comprehensive speech translation system that converts spoken English to multiple languages and generates translated audio output.

## Features

- **🎤 Audio Input**: Upload audio files or record using microphone
- **🔍 Speech Recognition**: Whisper ASR with fallback to SpeechRecognition
- **🌐 Translation**: NLLB-200 multilingual translation model
- **🔊 Text-to-Speech**: Coqui TTS with gTTS fallback
- **🖥️ Web Interface**: Streamlit UI with real-time processing
- **📱 Multi-language Support**: French, German, Spanish, Hindi, Chinese, Arabic, Russian

## Architecture

```
Audio Input → ASR (Whisper) → Translation (NLLB) → TTS (gTTS/Coqui) → Audio Output
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gracee1232/speech-speech.git
cd speech-speech
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Command Line
```bash
python pipeline.py
```

## Components

- **asr_whisper.py**: Speech-to-text conversion using Whisper
- **translate_nllb.py**: Text translation using NLLB-200
- **tts_coqui.py**: Text-to-speech synthesis
- **streamlit_app.py**: Web interface
- **pipeline.py**: Command-line pipeline

## Supported Languages

- English (source)
- French (fra_Latn)
- German (deu_Latn) 
- Spanish (spa_Latn)
- Hindi (hin_Deva)
- Chinese (zho_Hans)
- Arabic (arb_Arab)
- Russian (rus_Cyrl)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Streamlit
- Whisper
- gTTS

## License

MIT License
