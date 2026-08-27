# Karma — 100% Local Autonomous AI Companion

A state-of-the-art multimodal AI companion with **sight, hearing, voice, cognition, and episodic vector memory**. Karma runs entirely in-process using Apple Silicon Metal (MPS) or CPU (e.g. Raspberry Pi 4), with **zero mandatory cloud dependencies, zero external servers (no Ollama required), and zero network calls**.

Karma can also be optionally paired with cloud LPUs via the `--groq` flag for sub-second reasoning with `openai/gpt-oss-20b`.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Threading & Concurrency Model](#threading--concurrency-model)
3. [Perception Subsystem (Sight & Tracking)](#perception-subsystem-sight--tracking)
4. [Auditory Subsystem (Hearing & STT)](#auditory-subsystem-hearing--stt)
5. [Cognitive Engine (Mind & Reasoning)](#cognitive-engine-mind--reasoning)
6. [Speech Synthesis & Streaming Prosody (Voice & TTS)](#speech-synthesis--streaming-prosody-voice--tts)
7. [Memory Systems & Sleep Consolidation](#memory-systems--sleep-consolidation)
8. [Companion Face UI & Rendering](#companion-face-ui--rendering)
9. [Subsystem Resource Footprint](#subsystem-resource-footprint)
10. [CLI Flags, Hotkeys & Configuration](#cli-flags-hotkeys--configuration)
11. [Installation & Setup](#installation--setup)
12. [Project File Map](#project-file-map)
13. [Test Suite & Verification](#test-suite--verification)

---

## System Architecture

Karma is organized as an event-driven, multimodal feedback loop where perception, memory, cognition, and speech run in parallel across specialized background workers.

```
                                      ┌────────────────────────────────────────────────────────┐
                                      │                   HARDWARE INTERFACES                  │
                                      └──────────────────────────┬─────────────────────────────┘
                                                                 │
                  ┌──────────────────────────────────────────────┼──────────────────────────────────────────────┐
                  ▼                                              ▼                                              ▼
           [ HD Webcam ]                                   [ Microphone ]                                [ Speakers / DAC ]
                  │                                              │                                              ▲
                  ▼                                              ▼                                              │
      ┌───────────────────────┐                      ┌───────────────────────┐                                  │
      │   VISION PIPELINE     │                      │    AUDIO PIPELINE     │                                  │
      │  YOLOv8n Object Det.  │                      │   Silero VAD (JIT)    │                                  │
      │  MediaPipe 3D Hands   │                      │  faster-whisper INT8  │                                  │
      │  Face / Gaze Tracking │                      │ Hallucination Filter  │                                  │
      └───────────┬───────────┘                      └───────────┬───────────┘                                  │
                  │                                              │                                              │
                  │              ┌───────────────────────────────┘                                              │
                  │              ▼                                                                              │
                  │   ┌─────────────────────────────────────────────────────┐                                   │
                  └──▶│                   WORKING MEMORY                    │                                   │
                      │  Thread-Safe Event Stream (Speech, Vision, Triggers)│                                   │
                      │  Global Workspace Consciousness (Saliency / Surprise)│                                  │
                      │  Recent Conversation History & Context Window       │                                   │
                      └──────────────────────────┬──────────────────────────┘                                   │
                                                 │                                                              │
                                                 ▼                                                              │
                      ┌─────────────────────────────────────────────────────┐                                   │
                      │                 COGNITIVE ORCHESTRATOR              │                                   │
                      │  • Spontaneous Urgent Thoughts (Saliency > 0.70)    │                                   │
                      │  • Conversational Reply Generation (User Speech)    │                                   │
                      │  • Ambient Idle Reflection (Subconscious)           │                                   │
                      └──────────────────────────┬──────────────────────────┘                                   │
                                                 │                                                              │
                        ┌────────────────────────┴────────────────────────┐                                     │
                        ▼                                                 ▼                                     │
         ┌──────────────────────────────┐                  ┌──────────────────────────────┐                     │
         │   LOCAL LLM ENGINE (Mind)    │                  │      GROQ CLOUD ENGINE       │                     │
         │  llama_cpp (In-Process GGUF) │        OR        │   openai/gpt-oss-20b (LPUs)  │                     │
         │  Qwen 2.5 1.5B Instruct Q4   │                  │   Chain-of-Thought Stream    │                     │
         │  FlashAttn + Q8_0 KV Cache   │                  │   Activated via: --groq      │                     │
         └──────────────┬───────────────┘                  └──────────────┬───────────────┘                     │
                        │                                                 │                                     │
                        └────────────────────────┬────────────────────────┘                                     │
                                                 ▼                                                              │
                                  ┌──────────────────────────────┐                                              │
                                  │      PROSODY & STREAMING     │                                              │
                                  │  Real-Time JSON Prefix Stream│                                              │
                                  │  Emotion & Inflection Parsing│                                              │
                                  │  Dynamic Speed Modulation    │                                              │
                                  └──────────────┬───────────────┘                                              │
                                                 │                                                              │
                                                 ▼                                                              │
                                  ┌──────────────────────────────┐                                              │
                                  │      SPEECH SYNTHESIS        │                                              │
                                  │  Kokoro-82M High-Fidelity    │                                              │
                                  │  Cosine Edge Fade (No Clicks)│                                              │
                                  │  Peak Volume Normalization   │                                              │
                                  │  Continuous OutputStream     ├──────────────────────────────────────────────┘
                                  └──────────────────────────────┘
```

---

## Threading & Concurrency Model

Karma runs **5 concurrent background threads** coordinated alongside the main OpenCV display loop:

| Thread Name | Module | Responsibility |
|---|---|---|
| `MainThread` | `src/vision/pipeline.py` | Captures camera frames, runs YOLO/Face/Hands, renders procedural fullscreen UI at native screen refresh rate. |
| `audio_streamer` | `src/audio/pipeline.py` | Listens to the microphone, evaluates Silero VAD frames, executes faster-whisper transcription, and feeds working memory. |
| `consciousness_orchestrator` | `src/cognition/interaction.py` | Evaluates conscious salience, queries vector memory RAG, and invokes LLM generation. |
| `prosody_synth` | `src/speech/prosody.py` | Consumes streaming text tokens from LLM and synthesizes audio chunks in parallel. |
| `prosody_drain` | `src/speech/prosody.py` | Feeds synthesized audio waveforms into a continuous non-blocking `sounddevice.OutputStream`. |
| `idle_watcher` | `src/memory/consolidation.py` | Tracks conversation inactivity and triggers automatic episodic sleep consolidation after 20 minutes. |
| `stdin_listener` | `src/main.py` | Listens for interactive keyboard commands (`sleep`, `quit`) and handles `Ctrl+D` (EOF) clean termination. |

---

## Perception Subsystem (Sight & Tracking)

The vision pipeline is managed in `src/vision/` and runs in real time on Apple Silicon MPS or multi-core CPU.

```
Camera Frame (640x480) ──┬──▶ YOLOv8 Nano (MPS) ─────────────▶ Object Labels & Spatial Bounding Boxes
                         ├──▶ MediaPipe Hands (3D) ──────────▶ 21 Keypoints & Hand Gesture Landmarks
                         └──▶ Face & Gaze Tracker (HOG/CNN) ──▶ 128D Face Encodings + Normalized (gx, gy) Gaze
```

### 1. Object Detection (`detector.py`)
- **Engine**: Ultralytics YOLOv8n (Nano) quantized weights (`yolov8n.pt`).
- **Resolution**: Scaled dynamically to 320x320 for 4x faster CPU inference or native MPS acceleration.
- **Output**: Detects common household objects, deduplicates repeated observations within 3 seconds, and updates Karma's spatial working memory consciousness.

### 2. Face Recognition & Automatic Name Learning (`face.py` & `face_registry.py`)
- **Face Encodings**: Generates 128-dimensional biometric embeddings using dlib HOG / deep metric networks.
- **Persistent Face Registry (`faces.json`)**: Matches faces using Euclidean distance thresholding (`tolerance = 0.55`).
- **Conversational Name Learning**: Listens to conversational introductions (e.g., *"I'm Ahmed"*, *"My name is Sarah"*, *"Call me Alex"*), extracts the name using linguistic heuristics, pairs it with the current visual face embedding, and saves it permanently to `faces.json`. Subsequent interactions greet the person by their real name.

### 3. Gaze Tracking (`face.py`)
- Computes the bounding box center of the primary face relative to the frame.
- Normalizes gaze offsets to `[-1.0, 1.0]` and updates `internal_state.set_gaze(x, y)` with gentle exponential moving average (EMA) smoothing, allowing Karma's animated eyes to look directly at the user as they move around the room.

### 4. 3D Hand Landmark Tracking (`hand.py`)
- Utilizes MediaPipe Hand Landmarker Task (`hand_landmarker.task`) to extract 21 3D joint landmarks for real-time gesture recognition.

---

## Auditory Subsystem (Hearing & STT)

The audio subsystem in `src/audio/` delivers instant, low-latency voice capture with zero cloud roundtrips.

```
Mic Stream (16kHz) ──▶ Silero VAD (32ms blocks) ──▶ Energy Gate ──▶ faster-whisper (tiny.en) ──▶ Hallucination Filter ──▶ Working Memory
```

### 1. Voice Activity Detection (Silero VAD)
- **Model**: `silero_vad.jit` TorchScript model evaluated in 512-sample (32ms) blocks.
- **Dynamic Energy Gate**: Calculates exponential moving average background room noise (`bg_energy`) and requires `energy > bg_energy * 2.2` alongside VAD confidence > 0.35 to eliminate false triggers from keyboard clicks, air conditioners, or breathing.
- **Pre-Roll Ring Buffer**: Preserves 800ms of audio prior to speech onset so the first syllable is never clipped.

### 2. Local Speech-to-Text (`faster-whisper`)
- **Model**: Whisper `tiny.en` running with INT8 quantization across 4 CPU threads.
- **Cold-Start Warmup**: Pre-warms the transcription engine on boot to eliminate first-utterance latency.

### 3. Hallucination & Subtitle Filter (`is_valid_transcript`)
- Automatically rejects Whisper hallucination artifacts (e.g., `"[BLANK_AUDIO]"`, `"(wind blowing)"`, `"Thank you for watching!"`, `"Subtitles by..."`, or repetitive character loops).

### 4. Acoustic Barge-In & Echo Suppression
- Microphone input is muted during agent speech playback to prevent self-triggering loops.
- When `BARGE_IN_ENABLED = True` (ideal for AirPods / headphones), a user speaking over Karma triggers immediate audio interruption and sound device flushing.

---

## Cognitive Engine (Mind & Reasoning)

Karma's mind (`src/cognition/`) operates on **Global Workspace Theory (GWT)**, where sensory observations compete for attention based on salience and prediction error.

```
                        ┌──────────────────────────────────────────────┐
                        │              WORKING MEMORY                  │
                        │  - Visible Objects (Recent 8s)               │
                        │  - Recognized People in Room                 │
                        │  - Conversation History (Last 5 turns)       │
                        │  - Long-Term Vector RAG (sqlite-vec)         │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     COGNITIVE ORCHESTRATOR       │
                              └────────────────┬─────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
        [ Saliency > 0.70 / High Surprise ]             [ User Spoke / Dialogue Turn ]
                       │                                               │
                       ▼                                               ▼
            `think_immediately()`                          `run_interaction_response()`
        Spontaneous Natural Remark                     Real-Time Streaming Reply
```

### 1. Local In-Process LLM Engine (`LocalEngine`)
- **Default Engine**: `qwen2.5-1.5b-instruct-q4_k_m.gguf` loaded directly in-process via `llama-cpp-python`.
- **Optimization**: FlashAttention enabled with `Q8_0` KV cache quantization (50% RAM reduction + 3x faster prompt evaluation).
- **Anti-Looping Penalties**: Configured with `repeat_penalty = 1.20`, `frequency_penalty = 0.35`, and `presence_penalty = 0.15` to eliminate token looping.
- **C-Level Stderr Silencing**: Uses OS file descriptor redirection (`SilenceStderrFD`) to suppress noisy engine diagnostic outputs.

### 2. Cloud Groq LPU Engine (`GroqEngine` — `--groq`)
- **Model**: `openai/gpt-oss-20b` (or custom models via `--groq=model_name`).
- **Chain-of-Thought Reasoning Handling**: `GroqEngine` allocates a dedicated 1024-token budget (`max_completion_tokens = 1024`), buffering internal reasoning tokens and streaming clean user-facing dialogue in sub-250ms bursts.
- **Activated Only on Demand**: Passing `--groq` switches to the cloud LPU; omitting it keeps the entire system 100% offline.

### 3. Phrase Loop Deduplication (`_deduplicate_phrase_loops`)
- Safeguards dialogue against repetition loops by detecting multi-word and single-word repeats and pruning them before speech synthesis or storage.

---

## Speech Synthesis & Streaming Prosody (Voice & TTS)

Karma features an expressive voice pipeline (`src/speech/`) that modulates vocal speed and inflection based on emotion.

```
LLM Streaming Tokens ──▶ _JSONPrefixParser ──▶ Emotion / Inflection Extraction ──▶ Speed Resolver
                                │                                                        │
                                ▼                                                        ▼
                        text_chunks ──────────────────────────────────────────▶ Parallel Kokoro TTS
                                                                                         │
                                                                                         ▼
                                                                              Cosine Edge Ramp (2.5ms)
                                                                                         │
                                                                                         ▼
                                                                             Continuous OutputStream
```

### 1. Real-Time Streaming JSON Prefix Parser (`_JSONPrefixParser`)
- Parses streaming LLM tokens on the fly without waiting for the full JSON object to close.
- Extracts `emotion` (e.g., *playful, curious, excited, tired, warm*) and `inflection` (e.g., *question, whisper, emphatic*) within the first 15 tokens to dynamically calculate speech speed (`0.80x` to `1.25x`) before text synthesis begins.
- Immune to markdown fences (` ```json `) and compatible with all schema keys (`text_chunks`, `response`, `reply`, `message`).

### 2. High-Fidelity Kokoro-82M Synthesis (`TTSEngine`)
- **Engine**: Kokoro-82M (`af_bella` default voice, 54 voices available).
- **Zero-Whine Native PyTorch Synthesis**: Runs unquantized floating-point synthesis by default, eliminating the high-frequency metallic "whine" and buzz associated with 4-bit INT4 vocoders.
- **Anti-Click Cosine Ramps**: Applies a 2.5ms smooth cosine fade-in and fade-out to chunk boundaries, preventing DC offset clicks and pops.
- **Peak Volume Normalization**: Normalizes peak amplitude to `0.88` to prevent DAC digital clipping.

### 3. Continuous Audio Streaming (`sd.OutputStream`)
- Rather than opening and closing the audio device per sentence, `prosody_stream` feeds synthesized audio buffers into an open, persistent `sounddevice.OutputStream`. This eliminates stutter, gaps, and hardware clicks between consecutive sentences.

---

## Memory Systems & Sleep Consolidation

Karma combines short-term in-memory working consciousness with persistent long-term vector RAG (`src/memory/`).

```
Working Memory (In-Memory Ring) ──▶ Sleep Consolidation (LLM Summary) ──▶ Vector Embedder (MiniLM) ──▶ memory.db (sqlite-vec)
                                                                      ──▶ JSONL Transcript Archive ──▶ memory_archive/
```

### 1. Working Memory (`working.py`)
- Thread-safe store for recent perceptual events, recognized faces, spatial object maps, and the last 10 conversation turns.

### 2. Long-Term Episodic Vector Store (`store.py`)
- **Database**: SQLite with `sqlite-vec` vector extension (`memory.db`).
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings, 100% local).
- **Retrieval**: Automatically retrieves the top-2 most semantically relevant memories during dialogue turns and injects them into the prompt.

### 3. Sleep Consolidation (`consolidation.py`)
- **Automatic Trigger**: After 20 minutes of room inactivity (`IDLE_SLEEP_MINUTES = 20`), or manually by typing `sleep` into the terminal.
- **Consolidation Process**:
  1. The LLM reviews the entire day's raw observations and generates a coherent episodic summary.
  2. Extracts key facts about people, preferences, and events.
  3. Embeds each memory chunk and commits it to `memory.db`.
  4. Archives the raw event logs to timestamped files in `memory_archive/*.jsonl`.
  5. Clears working memory, preparing Karma for the next awake cycle.

---

## Companion Face UI & Rendering

Karma features an animated companion face (`FaceRenderer` in `src/vision/render.py`) rendered via OpenCV.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  KARMA  ●  [SPEAKING]                                            NRG [█████] CUR [███] │  <-- Top Bar (Pinned y=0)
│                                                                                        │
│                                                                                        │
│                             ( ● )                  ( ● )                               │  <-- Procedural Glowing Eyes
│                                                                                        │      (Gaze Tracking & Blinking)
│                                      ~~~~~~~~                                          │  <-- Reactive Mouth Waveform
│                                                                                        │
│                                                                                        │
│                          ┌───────────────────────────────┐                             │
│                          │ Karma: "Hey! How's it going?" │                             │  <-- Subtitle Overlay Pill
│                          └───────────────────────────────┘                             │
│                                                             Ctrl+D to exit | 'f' | 'd' │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key UI Features
- **Zero-Letterbox Auto-Resolution**: Automatically detects display resolution via macOS Cocoa `NSScreen` / `cv2.getWindowImageRect` (e.g. 1710x1112 on MacBook Pro), scaling canvas pixels with zero letterboxing gaps.
- **Expressive Eyes**: Procedural glowing eyes with natural blinking curves, specular highlights, and real-time gaze tracking.
  - *Happy / Playful*: Upturned curved eyes (`^ ^`).
  - *Excited*: Wide open round eyes (`O O`) with pupil sparkles.
  - *Tired / Sleepy*: Relaxed horizontal slits (`- -`).
  - *Attentive / Curious*: Stadium capsule eyes tracking user position.
- **Reactive Mouth**: Dynamic lip-sync waveform that animates in real time while speaking.
- **Top HUD & Meters**: Displays companion mood badge, live state pulse indicator, Energy (`NRG`) bar, and Curiosity (`CUR`) meter.
- **Live Subtitles**: Word-wrapped translucent subtitles pill displaying recent dialogue for both user and Karma.

---

## Subsystem Resource Footprint

| Subsystem | Engine / Library | Model / Weight | RAM Footprint | Compute Target |
|---|---|---|---|---|
| **Language Model (LLM)** | `llama-cpp-python` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | **~1,100 MB** | Metal / CPU |
| **Speech-to-Text (STT)** | `faster-whisper` | `tiny.en` (INT8) | **~75 MB** | Multi-Thread CPU |
| **Voice Activity Detection** | `silero-vad` (PyTorch) | `silero_vad.jit` | **~15 MB** | Multi-Thread CPU |
| **Speech Synthesis (TTS)** | `kokoro` (PyTorch) | Kokoro-82M (`af_bella`) | **~140 MB** | Multi-Thread CPU |
| **Computer Vision** | `ultralytics` | `yolov8n.pt` (Nano) | **~70 MB** | Metal MPS / CPU |
| **Face & Hands Tracking** | `face_recognition` + MediaPipe | HOG + 3D Hand Landmarks | **~60 MB** | Multi-Thread CPU |
| **Vector RAG Storage** | `sentence-transformers` + `sqlite-vec` | `all-MiniLM-L6-v2` | **~90 MB** | Multi-Thread CPU |
| **Total System RAM** | | | **~1.55 GB** | |

---

## CLI Flags, Hotkeys & Configuration

### Command-Line Arguments

| Flag | Shorthand | Description |
|---|---|---|
| `--groq` | `-g` | Activates Groq cloud LPUs with `openai/gpt-oss-20b` for ultra-fast reasoning. |
| `--groq=<model>` | | Specifies a custom Groq model (e.g. `--groq=llama-3.3-70b-versatile`). |
| `--debug` | `-d`, `-v` | Enables verbose technical logging and displays live OpenCV debug camera HUD. |
| `--camera` | `-c` | Forces the debug camera window to show without enabling full verbose logs. |
| `--windowed` | `-w` | Launches companion face in a standard floating window instead of fullscreen. |
| `--fullscreen` | `-f` | Forces companion face into fullscreen mode (default). |

### Keyboard Shortcuts & Runtime Controls

- **Exit Application**: Press `Ctrl+D` (terminal or UI), or press `Esc` / `q` in the face window.
- **Toggle Fullscreen**: Press `f` in the face window.
- **Toggle Debug Camera Window**: Press `d` in the face window.
- **Trigger Sleep Consolidation**: Type `sleep` and press Enter in the terminal.

---

## Installation & Setup

### 1. Prerequisites (macOS)
```bash
# Install audio I/O and speech libraries via Homebrew
brew install portaudio espeak-ng

# Clone repository
git clone https://github.com/ahmedbarakat207/karma.git
cd karma

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for Groq)
Create a `.env` file in the project root:
```bash
# Add your Groq API key if using --groq
GROQ_API_KEY="gsk_..."
```

### 3. Run Karma

```bash
# 1. Standard Mode (100% Offline, Fullscreen Face, Quiet Console)
python3 main.py

# 2. Groq Cloud Mode (Ultra-Fast Reasoning via gpt-oss-20b)
python3 main.py --groq

# 3. Debug Mode (Verbose Subsystem Logs + Camera Detection HUD)
python3 main.py --debug
```

### 4. Query Past Memories
Search through consolidated episodic vector memories from the CLI:
```bash
python3 recall.py "what did we discuss about the project?"
```

---

## Project File Map

```
karma/
├── main.py                     # Root CLI entrypoint (invokes src.main)
├── recall.py                   # CLI tool to query SQLite vector memories
├── requirements.txt            # Python dependencies
├── faces.json                  # Persistent recognized face biometric registry
├── memory.db                   # sqlite-vec episodic vector database
├── models/                     # 100% local model weights directory
│   ├── qwen2.5-1.5b-instruct-q4_k_m.gguf
│   ├── yolov8n.pt
│   ├── silero_vad.jit
│   ├── whisper-tiny.en/
│   ├── all-MiniLM-L6-v2/
│   └── hand_landmarker.task
├── src/
│   ├── config.py               # Central configuration, .env loader & CLI parser
│   ├── main.py                 # Thread coordinator & lifecycle manager
│   ├── state.py                # Internal emotional state, gaze & subtitles store
│   ├── audio/
│   │   ├── __init__.py
│   │   └── pipeline.py         # Silero VAD, faster-whisper STT & audio streamer
│   ├── cognition/
│   │   ├── __init__.py
│   │   ├── engine.py           # Local llama_cpp engine & Groq cloud engine
│   │   ├── interaction.py      # Conversational prompt formatting & response generator
│   │   └── think.py            # Spontaneous thoughts & subconscious reflection
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── working.py          # Working memory & Global Workspace consciousness
│   │   ├── store.py            # sqlite-vec vector database interface
│   │   ├── face_registry.py    # Biometric face encoding matcher & name registry
│   │   └── consolidation.py    # Nightly sleep consolidation & JSONL archiver
│   ├── speech/
│   │   ├── __init__.py
│   │   ├── tts.py              # Kokoro-82M synthesis, audio normalization & edge ramps
│   │   └── prosody.py          # Streaming JSON prefix parser & continuous OutputStream
│   └── vision/
│       ├── __init__.py
│       ├── detector.py         # YOLOv8n object detector
│       ├── face.py             # Face recognition & normalized gaze tracker
│       ├── hand.py             # MediaPipe 3D hand landmark tracker
│       ├── pipeline.py         # High-FPS vision loop & window resolution manager
│       └── render.py           # Expressive procedural FaceRenderer & debug HUD
└── tests/
    ├── __init__.py
    └── test_suite.py           # 49 unit and integration test cases
```

---

## Test Suite & Verification

Karma includes a complete unit and integration test suite covering LLM prompt formatting, streaming JSON prefix parsing, speech prosody, face biometrics, VAD filtering, and CLI parsing.

Run the test suite via `pytest`:
```bash
pytest tests/test_suite.py -v
```

All 49 tests validate 100% offline and Groq cloud integration paths without requiring live hardware.
