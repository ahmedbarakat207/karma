# Ambient Agent

A local, promptless agent for Apple Silicon Macs. It watches through your
webcam, listens through your mic, quietly "thinks" about what it notices,
and — when it goes idle (or you tell it to) — consolidates everything into
a summarized, embedded long-term memory it can recall from later.

No chat interface. It isn't waiting for you to ask it anything.

## How it fits together

```
webcam ──▶ YOLO object detection ──┐
                                    ├──▶ working memory (rolling log) ──▶ think loop (Qwen3.5) ──▶ appends thoughts ──▶ Kokoro TTS speaks them aloud
mic ─────▶ whisper speech-to-text ─┘                                                                     │
    ▲                                                                                                     │
    └────────────────────── muted while TTS is speaking ───────────────────────────────────────────────┘

                    │ (idle timeout, or you type "sleep")
                    ▼
            consolidation: LLM summarizes the whole log
                    │
                    ▼
         chunk ──▶ embed ──▶ memory.db (sqlite-vec)   <- this is the RAG file
                    │
                    ▼
          raw log archived to memory_archive/*.jsonl
```

### Speech (fully local, no API)

Uses **Kokoro-82M** (Apache-2.0) — an open, 82-million-parameter TTS model
that runs faster than real-time on Apple Silicon CPU alone. Weights (~350MB)
download once from Hugging Face on first run; after that, zero network
calls happen for speech. It's the closest thing to natural human narration
you'll get locally without a much heavier model — it doesn't clone a
specific voice, it uses one of 54 built-in presets (default here is
`af_heart`; see the [hexgrad/Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M)
for the full voice list to swap in `config.py`).

**Feedback loop guard:** the mic thread checks a shared flag and mutes
itself the instant the agent starts talking, so it doesn't transcribe its
own voice back into memory as something it "heard." This isn't bulletproof
against overlap on chunk boundaries (see Known Limitations) — if you're
using headphones/AirPods instead of speakers, the problem disappears
entirely since the mic never picks up the output at all.

Set `SPEAK_THOUGHTS = False` in `config.py` to go back to silent/text-only.

## Setup

```bash
# system deps
brew install portaudio espeak-ng   # espeak-ng is used by Kokoro (TTS) for phonemization

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Metal acceleration is on by default in recent llama-cpp-python wheels on
# Apple Silicon. If it isn't picking up your GPU, force a rebuild:
CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

**Note on VLM / Deep Scene Understanding:**
To enable deep scene understanding via Moondream, you must have [Ollama](https://ollama.com) running in the background. Open a separate terminal and run:
```bash
ollama run moondream
```
If Ollama is not running, the VLM analyzer will silently fail (which is safe), but you won't get advanced scene descriptions.

Download the model (Q4_K_M is a good balance of speed/quality for a 4B model):

```bash
pip install huggingface_hub
huggingface-cli download bartowski/Qwen_Qwen3.5-4B-GGUF \
  --include "Qwen_Qwen3.5-4B-Q4_K_M.gguf" \
  --local-dir ./models
```

Confirm the filename in `config.py` (`MODEL_PATH`) matches what got downloaded.

## First run: macOS permissions

The first time you run this, macOS will prompt for camera and microphone
access **for whatever terminal app you're running it from** (Terminal,
iTerm2, etc). Approve both in System Settings → Privacy & Security, or
vision/audio will silently fail to capture anything.

## Running it

```bash
python main.py
```

While it's running:
- Type `sleep` + Enter to force consolidation immediately.
- Type `quit` + Enter to shut down.
- Otherwise, it auto-sleeps after `IDLE_SLEEP_MINUTES` (default 20) of no new
  sensor activity — thinking to itself doesn't count as activity, only new
  sights/sounds do, so it will actually wind down when the room goes quiet.

Check what it's remembered:

```bash
python recall.py "what happened this afternoon"
```

## Tuning knobs (all in `config.py`)

- `THINK_INTERVAL_SECONDS` — how often it reflects. Lower = more "alive"
  feeling but more LLM calls.
- `OBJECT_DEDUP_SECONDS` — how long before it'll re-log the same object.
  Too low and the log turns into noise; too high and it misses things
  leaving/returning.
- `IDLE_SLEEP_MINUTES` — how much silence before auto-sleep.
- `WHISPER_MODEL_SIZE` — `tiny` is faster/lower quality, `small`/`medium`
  slower/better, if the base model isn't accurate enough.

## Known limitations / natural next steps

- **Audio chunking is fixed-length, not VAD-segmented** — it'll occasionally
  cut a sentence at a chunk boundary. Swap in `webrtcvad` or `silero-vad`
  for cleaner segmentation if this bothers you.
- **Mic muting during speech isn't perfectly tight** — if a recording chunk
  started right as the agent began talking, that chunk can still catch some
  of its own voice. Using headphones instead of speakers sidesteps this
  completely. A cleaner fix is switching audio.py to VAD-based segmentation
  so recording starts/stops on speech boundaries rather than a fixed timer.
- **Kokoro doesn't clone a specific voice** — if you want it to sound like
  one particular voice (yours, a character, etc.) rather than a preset,
  look at Chatterbox or Qwen3-TTS Base, both local and Apache-2.0, both
  heavier to run than Kokoro.
- **No wake-time memory injection yet** — `recall.py` proves retrieval
  works, but the think loop doesn't currently pull past memories into its
  context automatically. To close the loop: embed the current working-memory
  window, query `MemoryStore` for similar past memories, and prepend the
  top few to the prompt in `think.py`. That's what gives it real continuity
  ("I remember this happening before") instead of a blank slate each wake.
- **Single always-on process, not a background daemon** — for real
  "always running" behavior, wrap `main.py` in a `launchd` plist so macOS
  keeps it alive across reboots/logouts instead of running it in a terminal
  tab.
- **Raw video/audio is never persisted** — only text (labels, transcripts,
  summaries) gets written anywhere. If you want to keep footage too, that's
  a deliberate separate decision — worth thinking about where it goes and
  who else might end up in frame before you turn it on.
