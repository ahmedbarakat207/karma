# Karma

An offline physical companion robot built with Python, OpenCV, and local models. It combines local vision (YOLOv8, face and hand tracking), local speech recognition (Whisper), local TTS (Kokoro), an animated companion face with interactive touchscreen kiosk menu, and servo-controlled head tilt.

Runs locally on Apple Silicon (MPS) or a Raspberry Pi 4 with zero required cloud services (optional Groq support via `--groq`).

---

## Hardware

- **Compute**: Raspberry Pi 4 (4GB / 8GB) or Apple Silicon Mac
- **Display**: 7" 800x480 Capacitive Touch LCD
- **Neck Tilt**: TowerPro MG90S Micro Servo (GPIO 18 / PWM)
- **Camera**: Raspberry Pi Camera v2 or standard USB webcam
- **Audio**: USB microphone + speaker / 3.5mm DAC

---

## Quickstart

### 1. Install dependencies
```bash
# macOS
brew install portaudio espeak-ng
pip install -r requirements.txt

# Raspberry Pi / Debian
sudo apt-get update && sudo apt-get install -y portaudio19-dev espeak-ng libatlas-base-dev pigpio python3-pigpio
pip install -r requirements.txt
```

### 2. Run
```bash
# Standard local mode
python main.py

# With camera debug window
python main.py --debug

# Cloud inference (Groq)
python main.py --groq
```

---

## Features

### Companion Face & Kiosk UI
- **Procedural Animated Face**: Glowing eyes with gaze tracking, natural blinking, mood palette shifts, and lip-sync audio waveforms.
- **Kiosk Menu**: Press `m` or tap the top-right `[ :: MENU ]` button to open the 7" LCD touchscreen kiosk:
  - **Facility Map**: Multi-floor visual layout with floor switching.
  - **Documents / RAG**: On-screen document reader powered by MarkItDown + sqlite-vec embeddings.
  - **Student Projects & Achievements**: Interactive project cards and milestone showcase.
- **135° Neck Tilt**: MG90S servo tilts the head down to 135° when the kiosk menu opens so the touchscreen is easy to use, then returns to 90° for face-to-face interaction.
- **Coding Display Mode**: When asked coding questions, code snippets are filtered out of speech and displayed in a center code viewer while Karma speaks the conversational explanation. Tap the code card to dismiss it.

### Voice & Perception
- **Speech-to-Text**: Local `faster-whisper` (tiny.en) with `silero-vad` voice activity detection.
- **Text-to-Speech**: Local `Kokoro-82M` synthesis with dynamic prosody and speed modulation based on emotion tags.
- **Vision Pipeline**: Real-time object detection via YOLOv8n, face recognition with automatic name learning, and MediaPipe 3D hand tracking.
- **Episodic Memory**: SQLite vector store (`sqlite-vec` + `all-MiniLM-L6-v2`) with automatic sleep consolidation for long-term memory retrieval.

---

## Controls

### Keyboard
| Key | Action |
|---|---|
| `m` | Toggle touchscreen kiosk menu (triggers 135° neck tilt) |
| `f` | Toggle fullscreen mode |
| `d` | Toggle debug camera HUD |
| `Ctrl+D` / `q` | Clean exit |

### Voice Commands
- *"Open the map"* / *"Show floor 2"*
- *"Show student projects"* / *"Open apps"*
- *"Open documents"* / *"Read manual"*
- *"Close menu"* / *"Back to face"*

---

## Document RAG CLI

Index and query PDFs or text documents into the local vector database:

```bash
# Ingest a PDF
python -m src.memory.rag --ingest path/to/document.pdf

# List indexed documents
python -m src.memory.rag --list

# Query from terminal
python -m src.memory.rag --query "battery specifications"
```

---

## Project Structure

```
karma/
├── main.py                     # Root entrypoint
├── chat.py                     # Terminal chat & model validation CLI
├── recall.py                   # Memory recall query CLI
├── setup.sh                    # Automated system & hardware installer
├── data/
│   ├── student_apps.json       # Kiosk showcase app list
│   ├── achievements.json       # Kiosk achievements data
│   └── maps/                   # Floor plan images
├── src/
│   ├── config.py               # Settings & CLI arguments
│   ├── state.py                # Thread-safe shared state & UI bus
│   ├── audio/
│   │   └── pipeline.py         # Mic input, VAD & faster-whisper STT
│   ├── cognition/
│   │   ├── engine.py           # llama.cpp (GGUF) & Groq clients
│   │   ├── interaction.py      # Conversation turn handling
│   │   └── think.py            # Idle reflection & spontaneous thoughts
│   ├── hardware/
│   │   └── neck.py             # MG90S servo driver & smooth ramp thread
│   ├── memory/
│   │   ├── working.py          # Short-term event buffer
│   │   ├── store.py            # sqlite-vec vector database
│   │   ├── rag.py              # PDF parsing (MarkItDown) & RAG pipeline
│   │   ├── face_registry.py    # Face embedding matcher
│   │   └── consolidation.py    # Background memory consolidation
│   ├── speech/
│   │   ├── tts.py              # Kokoro TTS engine
│   │   └── prosody.py          # Real-time token streaming & code filter
│   ├── ui/
│   │   └── kiosk.py            # 7" touchscreen menu state machine
│   └── vision/
│       ├── pipeline.py         # OpenCV camera loop
│       ├── detector.py         # YOLOv8 object detection
│       ├── face.py             # Face & smile tracker
│       ├── hand.py             # MediaPipe hand landmarks
│       └── render.py           # FaceRenderer & debug HUD
├── body/                       # 3D printable CAD models & assembly
└── tests/                      # Pytest test suite
```

---

## Testing

Run unit and integration tests:

```bash
pytest tests/ -v
```
