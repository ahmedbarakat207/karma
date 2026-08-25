# Karma — 100% Local Autonomous AI Companion

A 100% offline, local AI companion with sight, hearing, voice, and memory. It runs entirely in-process using Apple Silicon Metal or CPU (e.g. Raspberry Pi), with **zero cloud dependencies, zero external servers (no Ollama required), and zero network calls**.

---

## 100% Offline Architecture

```
webcam ──▶ YOLOv8 Nano (MPS / CPU) ──┐
                                     ├──▶ Working Memory ──▶ Local Cognition (Qwen 2.5 1.5B via llama_cpp) ──▶ Kokoro-82M TTS
mic ─────▶ Silero VAD + faster-whisper ┘                                                                               │
    ▲                                                                                                                  │
    └───────────────────────────────────── Muted during agent speech ──────────────────────────────────────────────────┘

                        │ (idle timeout, or type "sleep")
                        ▼
                Consolidation: LLM summarizes the day's experiences
                        │
                        ▼
             chunk ──▶ embed ──▶ memory.db (sqlite-vec)   <- Vector Long-Term Memory
                        │
                        ▼
              raw log archived to memory_archive/*.jsonl
```

---

## Subsystems & Resource Footprint

| Subsystem | Local Engine | Model | RAM Usage |
|---|---|---|---|
| **LLM Reasoning** | `llama-cpp-python` (in-process) | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | **~1,100 MB** |
| **Speech-to-Text (STT)** | `faster-whisper` | `tiny.en` (INT8) | **~75 MB** |
| **Voice Activity Detection** | `silero-vad` (PyTorch) | `silero_vad` | **~15 MB** |
| **Speech Synthesis (TTS)** | `kokoro` | Kokoro-82M (`af_bella`) | **~140 MB** |
| **Computer Vision** | `ultralytics` | `yolov8n.pt` (Nano) | **~70 MB** |
| **Face & Hands Tracking** | `face_recognition` + MediaPipe | HOG + 3D Hand Landmarks | **~60 MB** |
| **Vector RAG Storage** | `sentence-transformers` + `sqlite-vec` | `all-MiniLM-L6-v2` | **~90 MB** |
| **Total System RAM** | | | **~1.55 GB** |

---

## Quickstart

### 1. Setup Environment
```bash
# System dependencies (macOS)
brew install portaudio espeak-ng

# Install Python packages
pip install -r requirements.txt
```

### 2. Run Karma
```bash
python3 main.py
```

### 3. Controls (while running)
- Type `sleep` + Enter: Triggers subconscious memory consolidation into `memory.db`.
- Type `quit` + Enter: Exits cleanly.

---

## Query Long-Term Memory
Search past episodic memories from the terminal:
```bash
python3 recall.py "what did we talk about earlier"
```

---

## Tests
Run the test suite:
```bash
pytest tests/test_suite.py -v
```
