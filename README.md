# Karma

An offline physical companion robot. It hangs out in the room as a witty, chill friend — talking, listening, watching, remembering — in English and Egyptian Arabic. Everything runs locally: speech recognition, voices, vision, memory, and the language model. No cloud account needed (Groq is an optional fallback).

Runs on Apple Silicon (MPS) or a Raspberry Pi 4.

---

## Hardware

- **Compute**: Raspberry Pi 4 (4GB / 8GB) or Apple Silicon Mac
- **Display**: 7" 800x480 capacitive touch LCD
- **Neck**: TowerPro MG90S micro servo on GPIO 18 (PWM)
- **Camera**: Raspberry Pi Camera v2 or USB webcam
- **Audio**: USB microphone + speaker / 3.5mm DAC
- **Body**: 3D-printable chassis (STL files generated from `body/`, 11 parts)

---

## Quickstart

```bash
# macOS
brew install portaudio espeak-ng
pip install -r requirements.txt

# Raspberry Pi / Debian — or just run ./setup.sh (does all of this + models + service)
sudo apt-get update && sudo apt-get install -y portaudio19-dev espeak-ng libatlas-base-dev pigpio python3-pigpio
pip install -r requirements.txt
```

```bash
python main.py                 # standard local mode
python main.py --debug         # + camera window and verbose logging
python main.py --groq          # cloud inference instead of local model
python main.py --no-electron   # OpenCV face window instead of the Electron UI
```

`main.py` also accepts: `--camera` / `--no-camera`, `--windowed` / `--fullscreen`, `--electron`, `--groq=<model>`, `--groq-model=<model>`. While running, type `sleep` to force a memory-consolidation nap, `quit` to exit cleanly.

---

## What it does

### Brain (local LLM)

- **Qwen2.5-0.5B-Instruct at Q4_K_M** (`models/model.gguf`, ~379 MB) served by llama.cpp, ChatML-prompted. CTX 4096, `q8_0` KV cache, flash attention. Expects ~12–18 tok/s generation on a Pi 4 overclocked to 2 GHz.
- **Custom fine-tune**: the weights are LoRA-trained on Karma's own 1426-sample bilingual dataset (persona, thoughts, vision grounding, coding, refusals). See `training/`.
- **Model auto-download**: if `models/model.gguf` is missing, it pulls the stock Qwen GGUF from Hugging Face on first run.
- **Groq fallback**: `--groq` routes inference to a cloud model (`openai/gpt-oss-20b` default) through the same engine interface.
- **Prompt-lookup speculative decoding** is wired in (`SPECULATIVE_DECODING=prompt_lookup`) but off by default; helps most on RAG-grounded answers.

### A conversation turn

Each reply is assembled from, in order: mood instruction (playful / curious / tired / attentive, driven by the energy model) → Arabic auto-detect (answers in Egyptian Arabic when spoken to in Arabic) → people currently recognized → YOLO objects seen in the last 8 s → top episodic memories → top RAG document excerpts → kiosk notices. Last 4 conversation turns go along as history. Replies stream token-by-token into speech (see voices), code blocks are split off to the screen instead of being read aloud, and both sides are stored back to memory.

### Spontaneous thoughts

Karma thinks out loud on its own, in two ways:

- **Urgent**: something startling happens (loud noise, sudden movement, prediction error spike) → immediate remark or `[silence]`.
- **Idle**: every few seconds there is a small chance of a quiet remark about recent activity, throttled so it can't ramble.
- Thoughts come back as JSON (`emotion`, `inflection`, `text_chunks`) which drives the speaking speed and the face mood — or `[silence]`, in which case nothing happens. Spoken thoughts are off by default (`SPEAK_THOUGHTS`).

### Sleep and long-term memory

After 20 minutes idle, Karma announces it's resting, archives the raw session log to `memory_archive/`, and has the LLM summarize the session into first-person bullet memories stored as vectors. `recall.py "query"` searches all of it from the terminal (top 5 hits with timestamps and distances).

### Hearing

- **Speech-to-text**: local `faster-whisper` (`tiny`, int8 on CPU) with Silero VAD. Tunables: speech confidence, silence timeout (0.35 s), post-speech grace, minimum speech length.
- **Hallucination filter**: drops classic Whisper ghosts ("thank you", subtitle-site spam, <3 chars).
- **Loud-noise trigger**: a sudden bang registers as a high-salience event and can interrupt into an urgent thought.
- **Barge-in** (interrupt Karma by talking over it) is fully plumbed through audio → TTS → prosody but stays **off** by default (`BARGE_IN_ENABLED=False`).

### Voices (bilingual TTS)

- **English**: Kokoro-82M (`af_bella` default, any bundled voice selectable), with a quantized ONNX option.
- **Arabic**: Nabra-82M, auto-selected whenever the reply contains Arabic script. Downloaded on first Arabic reply.
- **Streaming prosody**: the reply is synthesized sentence-by-sentence while the LLM is still generating, and each chunk's speed follows the reply's emotion (excited ~1.15x, tired whisper ~0.85x, warm ~0.95x).
- **Code is never spoken**: fenced code blocks are filtered to the on-screen code panel while only the explanation is voiced. Arabic punctuation (`،؟`) is treated as sentence boundaries.

### Seeing

- **Objects**: YOLOv8n at 320 px, confidence 0.50, on MPS or CPU. Detections feed the conversation ("Current Environment: ...") and the vision table in the knowledge base. A person moving suddenly fires a high-salience trigger.
- **Faces**: Haar + HOG detection with smile tracking. Karma **learns names**: say "my name is Sara" while visible and the face embedding is registered (tolerance 0.55, averaged on re-register). Recognized people are greeted by name and listed in the prompt context.
- **Hands**: MediaPipe landmarks (up to 2 hands) with three named gestures — waving, thumbs up, pointing.

### Face and displays

Two interchangeable front-ends:

- **OpenCV window** (default off, `--no-electron`): procedural face with 8 mood palettes, gaze tracking that follows you (plus idle wander), blinking, talking mouth waveforms, energy/curiosity HUD bars, 7-second subtitle pills, and a side code panel with syntax tint.
- **Electron app** (default): 800x480 kiosk UI over WebSocket (`127.0.0.1:8765`) showing the animated SVG face, battery/telemetry, a CAD floor map with room beacons, project/achievement grids, a document reader, and tilt buttons. Reconnects automatically; also runs under plain Chromium kiosk if Electron is absent.

### Touchscreen kiosk

Five views — face, facility map (2 floors, switchable), documents (RAG reader with chunk paging), student apps, achievements — all operable by touch and by voice, in both languages:

| View | English triggers | Arabic triggers |
|---|---|---|
| Map | "open the map", "show floor 2", "where are we" | "افتح الخريطة", "الدور التاني", "احنا فين" |
| Achievements | "show achievements", "milestones" | "افتح الانجازات", "الشهادات" |
| Apps | "show student apps", "open projects" | "افتح المشاريع", "مشاريع الطلاب" |
| Documents | "open documents", "read manual" | "افتح الملفات", "الكتالوج" |
| Face (close) | "close menu", "back to face" | "اقفل القائمة", "ارجع للوش" |

Opening anything but the face tilts the head up for touchscreen use (see neck).

### Neck servo

MG90S on GPIO 18 (pigpio, mock when no daemon). 90° face-to-face, 135° kiosk angle, cosine-ramped moves at 60°/s, pulse cut on cleanup. Same 90→135 range of motion is mirrored in the CAD hinge.

### Memory and knowledge (RAG)

- **Working memory**: thread-safe event buffer with dedup windows, 10-turn conversation history, 180 s recent window, face frames, recognized people, and a salience system (`conscious_trigger` events above 0.7 force an urgent thought).
- **Vector store**: SQLite + `sqlite-vec`, `all-MiniLM-L6-v2` 384-dim embeddings, auto-pruned past 60 days / 1500 rows.
- **Document RAG**: hybrid dense + Arabic-aware keyword search fused per chunk (450 chars, 80 overlap), markdown/PDF/TXT ingestion with per-section attribution. `data/documents/karma_knowledge.md` (~550 lines: persona, coding manual, science, vision table, culture, music, safety) is ingested at setup and retrieved at k=2 every turn.
- **CLIs**: `python -m src.memory.rag --ingest/--dir/--query/--list/--clear`, `python recall.py "..."`, in-chat `/pdf <path>` and `/docs` (chat.py).

### Body

`body/` generates the printable robot: 11 STL parts (base with motors + battery, column with Pi + speakers, servo neck with 90→135° hinge, head with 7" LCD window + camera port + dome). `assembly_server.py` serves a Three.js assembly viewer on `:8787`.

### Training its brain

`training/` holds the full fine-tuning pipeline for the GGUF: a 1426-sample EN/AR dataset builder (persona 276+312, thoughts 291, vision 222, coding 110, QA 103, refusals 112), a Kaggle notebook (train → merge → Q4_K_M in one run), and a local `train_lora.py` (LoRA r=16, completion-only loss, works on old and new `trl`). Details in [`training/README.md`](./training/README.md).

---

## Controls

**Keyboard** (face window): `m` kiosk menu · `f` fullscreen · `d` debug camera HUD · `q` / `Ctrl+D` quit.

**Voice**: the kiosk table above, plus anything conversational. Say your name with "my name is ..." once to be remembered.

**Terminal** (while running): `sleep` forces a consolidation nap, `quit`/`exit` shuts down.

---

## Configuration

Everything is an env var (or `.env` file) read by `src/config.py`. The service profile in `setup.sh` runs lean (`N_THREADS=2`, `N_BATCH=256`, CTX 4096).

**LLM**: `MODEL_PATH` · `CTX_SIZE` (4096) · `N_BATCH` (512) · `N_THREADS` (min(4,cpu)) · `N_GPU_LAYERS` (auto: Metal on Mac, 0 on Pi) · `DEFAULT_TEMPERATURE` (0.7) · `DEFAULT_TOP_P` (0.9) · `DEFAULT_REPEAT_PENALTY` (1.05) · `DEFAULT_FREQUENCY_PENALTY` / `DEFAULT_PRESENCE_PENALTY` (0.0) · `KV_CACHE_TYPE` (q8_0) · `FLASH_ATTN` (true) · `SPECULATIVE_DECODING` (none) + `SPECULATIVE_NGRAM_SIZE` (2) / `SPECULATIVE_NUM_PRED_TOKENS` (8).

**Vision**: `YOLO_MODEL` · `HAND_LANDMARKER_MODEL` · `YOLO_DEVICE` (auto mps/cpu) · `YOLO_IMGSZ` (320) · `ENABLE_YOLO` (true) · `CAMERA_INDEX` (0). Fixed: confidence 0.50, 3 s object dedup.

**Hearing**: `WHISPER_MODEL_PATH` · `WHISPER_MODEL_SIZE` (tiny) · `WHISPER_LANGUAGE` (auto) · `SILERO_VAD_MODEL_PATH` · `VAD_SPEECH_CONFIDENCE` (0.35) · `VAD_SILENCE_TIMEOUT` (0.35) · `VAD_POST_SPEECH_GRACE_MS` (200) · `MIN_SPEECH_DURATION` (0.20).

**Voices**: `TTS_VOICE` (af_bella) · `USE_KOKORO_ONNX` (false) · `KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH` · `NABRA_ENABLED` (true) · `NABRA_MODEL_DIR` / `NABRA_REPO_ID` / `NABRA_VOICE`.

**Memory**: `EMBED_MODEL_PATH` (models/all-MiniLM-L6-v2). Fixed: `memory.db`, `memory_archive/`, prune at 60 d / 1500 rows, think every 5 s, 180 s recent window, 8 s vision window, 20 min idle sleep.

**Display/UI**: `UI_WS_HOST` / `UI_WS_PORT` (127.0.0.1:8765, Electron, localhost-only) · `UI_DASH_HOST` / `UI_DASH_PORT` (0.0.0.0:8080, LAN dashboard) · `KARMA_UI_PASSWORD` (dashboard password, auto-generated to `data/.dashboard_pass` if unset) · `USE_ELECTRON` (true). CLI-only: `--debug`, `--camera`, `--windowed`, `--fullscreen`, `--no-electron`, `--groq`.

---

## Remote dashboard (phone/laptop on the same WiFi)

When Karma runs, it also serves a password-gated dashboard (default `http://<pi-ip>:8080`). The URL is printed at boot and shown on the robot's own TELEMETRY tab. First boot generates a password into `data/.dashboard_pass` (mode 600) and prints it to the console; set `KARMA_UI_PASSWORD` to use your own. Sessions last 12 h, login is rate-limited, and every route (HTTP + live socket + camera) requires auth — the Electron websocket stays localhost-only and untouched. Tick "trust this device" at login to stay signed in for 30 days (a revocable token — logout kills it); only hashes are stored, so the secret never touches disk. Open it over wifi (`http://…`), never as a local file: a `file://` page has no provable origin and is deliberately never let in.

From the dashboard you can:

- **Logs** — live/filterable stream of thoughts, replies, heard speech, kiosk actions, system and error events
- **Camera** — real-time MJPEG of what the robot sees (or 503 if no camera)
- **Thoughts** — the think-loop feed, including `[silence]` decisions
- **Knowledge** — upload PDF/MD/TXT to the RAG (auto-ingested), list/delete indexed docs, test retrieval
- **Prompt** — view/edit the system prompt live (takes effect on the next reply, persisted to `data/persona.json`, reset restores default)
- **Settings** — temperature, top_p, repeat penalty, spoken thoughts, think interval, YOLO on/off, VAD silence timeout (validated, persisted to `data/config_overrides.json`, restored at boot)
- **Shell** — live terminal on the Pi as the service user (same login, idle sessions die after 15 min, `SHELL_ENABLED=0` disables it)
- **Status** — real telemetry: CPU %, SoC temp, throttle state, RAM/disk, load, uptime, LLM tok/s + TTFT averages, host IPs

Opening `ui/dashboard.html` straight from disk shows a labeled demo of the same UI (sample data, everything fenced off) — the live version only ever comes from `http://…:8080`.

The on-device TELEMETRY tab shows the same real data (it used to be hardcoded) plus the dashboard URL for discovery.

**Cloud (optional)**: `USE_GROQ` (false) · `GROQ_MODEL` (openai/gpt-oss-20b) · `GROQ_API_KEY` (in `.env`, never committed).

---

## Terminal CLIs

**`chat.py`** — talk to the GGUF without the robot: `python chat.py [--model/-m] [--ctx-size/-c] [--threads/-t] [--temperature] [--top-p] [--repeat-penalty] [--max-tokens] [--pdf/-p file] [--validate/-v] [--system-prompt/-s]`. In-chat: `/exit`, `/clear`, `/docs`, `/pdf <path>`. Prints tok/s + time-to-first-token per reply; `--validate` runs a 3-prompt smoke suite.

**RAG**: `python -m src.memory.rag --ingest file | --dir dir | --query "..." [--k N] | --list | --clear`.

**Memory**: `python recall.py "what did we talk about"`.

---

## Setup and service

- **`setup.sh`**: full Pi provision — IPv4/DNS fix, apt packages (build, audio, camera, GPIO, Xorg/Chromium), `karma` user + autologin, `config.txt` (camera, I2C/SPI, PWM, gpu_mem), audio levels, venv + Pi-wheels PyTorch/whisper/ultralytics stack, Node + Electron UI build, all model downloads, openbox kiosk config, `start_robot.sh` + `karma.service` (`Restart=always`), knowledge-base ingest, validation run.
- **`start_robot.sh`**: kiosk X setup (no blanking, no cursor) and a restart loop around `main.py` logging to `karma.log`.
- **`scripts/`**: `overclock.sh [moderate|turbo]` (Pi 4: 1800/2000 MHz, Pi 5: 2800/3000) · `revert_clock.sh` (back to stock) · `repair_numpy.sh` (venv numpy fix + service restart).

---

## Models (`models/`)

| File | What | Size |
|---|---|---|
| `model.gguf` | Qwen2.5-0.5B-Instruct Q4_K_M (the brain) | 469 MB on disk |
| `yolov8n.pt` | Object detection | ~6 MB |
| `hand_landmarker.task` | MediaPipe hand landmarks | ~8 MB |
| `whisper-tiny.en/` | Speech recognition | ~72 MB |
| `silero_vad.jit` | Voice activity detection | ~2 MB |
| `kokoro_q4.onnx` + `voices-v1.0.bin` | English TTS (quantized option) | ~291 + 27 MB |
| `all-MiniLM-L6-v2/` | Memory/RAG embeddings | ~87 MB |
| `nabra/` | Arabic TTS (auto-downloaded on first Arabic reply) | — |

---

## Project structure

```
karma/
├── main.py                 # launcher (env silencing, then src.main)
├── chat.py                 # terminal chat + validation for the GGUF
├── recall.py               # memory search CLI
├── setup.sh / start_robot.sh
├── Modelfile              # unused Ollama stub (llama.cpp path doesn't read it)
├── data/
│   ├── documents/karma_knowledge.md   # RAG knowledge base (ingested at setup)
│   ├── maps/               # kiosk floor plans (floor_1/2)
│   ├── student_apps.json / achievements.json  # kiosk content
│   └── memory.db / faces.json / memory_archive/  # created at runtime
├── src/
│   ├── config.py           # all settings + CLI flags
│   ├── main.py             # thread orchestration + shutdown
│   ├── state.py            # shared internal state + energy/mood model
│   ├── audio/pipeline.py   # mic, VAD, Whisper STT
│   ├── cognition/
│   │   ├── engine.py       # llama.cpp + Groq engines
│   │   ├── interaction.py  # turn assembly, intents, face learning
│   │   └── think.py        # urgent + idle thoughts
│   ├── hardware/neck.py    # MG90S servo driver
│   ├── memory/             # working, sqlite-vec store, RAG, consolidation, faces
│   ├── speech/             # kokoro/nabra TTS, streaming prosody, arabic g2p
│   ├── ui/                 # kiosk state machine + Electron app + WS server
│   └── vision/             # camera loop, YOLO, faces, hands, renderers
├── body/                   # CAD generators + assembly viewer
├── models/                 # see table above
├── scripts/                # overclock / revert / numpy repair
├── training/               # dataset + fine-tuning pipeline
└── tests/                  # pytest suite
```

---

## Testing

```bash
pytest tests/ -v
```

| File | Covers |
|---|---|
| `test_suite.py` | thinking-strip, prosody/JSON parser, audio helpers, reply parsing, prompt format, working memory, face registry, mood/subtitles, renderer, CLI flags |
| `test_bilingual_tts.py` | Arabic detection, speech cleanup, Arabic G2P phonemes, EN/AR TTS routing, Arabic kiosk intents |
| `test_coding_display.py` | code-block extraction, streaming code filter, coding renderer frame |
| `test_kiosk.py` | servo angles/pulses, kiosk views + touch zones, English intents |
| `test_rag.py` | parsing, chunking, ingest/query/list/clear, knowledge-base content |
| `test_ui_server.py` | websocket state updates, kiosk actions over WS |
