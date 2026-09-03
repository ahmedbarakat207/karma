
import os
import warnings

import sys

# Suppress framework & dependency warnings across all modules
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ApplePersistenceIgnoreState"] = "YES"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"
warnings.filterwarnings("ignore")


class SilenceStderrFD:
    def __enter__(self):
        try:
            sys.stderr.flush()
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.saved_stderr_fd = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.null_fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, "null_fd", None) is not None:
            try:
                sys.stderr.flush()
                os.dup2(self.saved_stderr_fd, 2)
                os.close(self.saved_stderr_fd)
                os.close(self.null_fd)
            except Exception:
                pass


import torch


# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Automatically load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except Exception:
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
        except Exception:
            pass

# CPU thread allocation (4 cores on Raspberry Pi 4)
N_THREADS = int(os.environ.get("N_THREADS", str(min(4, os.cpu_count() or 4))))


# Auto-detect best compute device (MPS on Apple Silicon, CPU on Raspberry Pi)
_DEFAULT_YOLO_DEVICE = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
_DEFAULT_GPU_LAYERS = -1 if _DEFAULT_YOLO_DEVICE == "mps" else 0

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, "model.gguf"))
HF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
HF_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
CTX_SIZE = int(os.environ.get("CTX_SIZE", "4096"))            # 4096 tokens context window (~25MB KV-cache in Q8_0)
N_BATCH = int(os.environ.get("N_BATCH", "512"))               # Fast prompt evaluation batching
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", str(_DEFAULT_GPU_LAYERS)))

# Speculative Decoding (Disabled by default to avoid numpy stride broadcast bug in streaming multi-turn)
SPECULATIVE_DECODING = os.environ.get("SPECULATIVE_DECODING", "none").lower()
SPECULATIVE_NGRAM_SIZE = int(os.environ.get("SPECULATIVE_NGRAM_SIZE", "2"))
SPECULATIVE_NUM_PRED_TOKENS = int(os.environ.get("SPECULATIVE_NUM_PRED_TOKENS", "8"))

# LLM Sampling Parameters (Tuned for Qwen 2.5 natural fluency and grammatical coherence)
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.environ.get("DEFAULT_TOP_P", "0.9"))
DEFAULT_REPEAT_PENALTY = float(os.environ.get("DEFAULT_REPEAT_PENALTY", "1.05"))
DEFAULT_FREQUENCY_PENALTY = float(os.environ.get("DEFAULT_FREQUENCY_PENALTY", "0.0"))
DEFAULT_PRESENCE_PENALTY = float(os.environ.get("DEFAULT_PRESENCE_PENALTY", "0.0"))

# KV Cache Quantization (Q8_0 for 50% lower KV-cache RAM footprint + Flash Attention)
KV_CACHE_TYPE = os.environ.get("KV_CACHE_TYPE", "q8_0").lower()
FLASH_ATTN = bool(int(os.environ.get("FLASH_ATTN", "1")))

PERSONA_SYSTEM_PROMPT = (
    "Your name is Karma. You are a genuine, warm, witty human friend sharing space with someone. "
    "You perceive the room naturally—seeing objects and hearing spoken conversation. "
    "You speak naturally like a close friend, with real emotion, humor, curiosity, and empathy. "
    "Use conversational language with natural contractions (I'm, that's, don't, let's). "
    "Never act like an AI, virtual assistant, language model, or computer program. "
    "Never mention cameras, sensors, vision models, or technical mechanics."
)

YOLO_MODEL = os.environ.get("YOLO_MODEL", os.path.join(MODELS_DIR, "yolov8n.pt"))
HAND_LANDMARKER_MODEL = os.environ.get("HAND_LANDMARKER_MODEL", os.path.join(MODELS_DIR, "hand_landmarker.task"))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", _DEFAULT_YOLO_DEVICE)
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "320"))         # 320x320 runs 4x faster on CPU than 640x640
YOLO_CONFIDENCE = 0.50
ENABLE_YOLO = bool(int(os.environ.get("ENABLE_YOLO", "1")))
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
VISION_POLL_SECONDS = 0.033                                   # Target ~30 FPS
OBJECT_DEDUP_SECONDS = 3                                      # Interval before repeating same object in memory
# Debug & Logging Mode (enabled via --debug, -d, -v, or DEBUG=1)
DEBUG = bool(int(os.environ.get("DEBUG", "0"))) if os.environ.get("DEBUG", "0").isdigit() else os.environ.get("DEBUG", "").lower() in ("true", "yes", "1")
SHOW_VISION_WINDOW = bool(int(os.environ.get("SHOW_VISION_WINDOW", "1" if DEBUG else "0"))) if os.environ.get("SHOW_VISION_WINDOW", "").isdigit() else os.environ.get("SHOW_VISION_WINDOW", "").lower() in ("true", "yes", "1")
LOG_VISION_TO_CONSOLE = bool(int(os.environ.get("LOG_VISION_TO_CONSOLE", "1" if DEBUG else "0"))) if os.environ.get("LOG_VISION_TO_CONSOLE", "").isdigit() else os.environ.get("LOG_VISION_TO_CONSOLE", "").lower() in ("true", "yes", "1")
FULLSCREEN_FACE = bool(int(os.environ.get("FULLSCREEN_FACE", "1"))) if os.environ.get("FULLSCREEN_FACE", "1").isdigit() else os.environ.get("FULLSCREEN_FACE", "1").lower() in ("true", "yes", "1")

# Groq Cloud LLM (Only active when --groq is explicitly passed)
USE_GROQ = bool(int(os.environ.get("USE_GROQ", "0"))) if os.environ.get("USE_GROQ", "0").isdigit() else os.environ.get("USE_GROQ", "").lower() in ("true", "yes", "1")
_raw_groq = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MODEL = f"openai/{_raw_groq}" if _raw_groq in ("gpt-oss-20b", "gpt-oss-120b", "gpt-oss-safeguard-20b") else _raw_groq


def log_debug(*args, **kwargs) -> None:
    """Print message only when DEBUG mode is active."""
    if DEBUG:
        print(*args, **kwargs)


def apply_cli_args(argv=None) -> None:
    """Applies command-line arguments to global configuration."""
    global DEBUG, SHOW_VISION_WINDOW, LOG_VISION_TO_CONSOLE, FULLSCREEN_FACE, USE_GROQ, GROQ_MODEL
    if argv is None:
        import sys
        argv = sys.argv[1:]

    for arg in argv:
        clean = arg.strip().lower()
        if clean in ("--debug", "-d", "-v", "--verbose"):
            DEBUG = True
            SHOW_VISION_WINDOW = True
            LOG_VISION_TO_CONSOLE = True
        elif clean in ("--camera", "--vision", "-c"):
            SHOW_VISION_WINDOW = True
        elif clean in ("--no-camera", "--hide-camera"):
            SHOW_VISION_WINDOW = False
        elif clean in ("--windowed", "-w", "--no-fullscreen"):
            FULLSCREEN_FACE = False
        elif clean in ("--fullscreen", "-f"):
            FULLSCREEN_FACE = True
        elif clean in ("--groq", "-g"):
            USE_GROQ = True
            GROQ_MODEL = "openai/gpt-oss-20b"
        elif clean.startswith("--groq="):
            USE_GROQ = True
            val = arg.split("=", 1)[1].strip()
            GROQ_MODEL = f"openai/{val}" if val in ("gpt-oss-20b", "gpt-oss-120b", "gpt-oss-safeguard-20b") else val
        elif clean.startswith("--groq-model="):
            USE_GROQ = True
            val = arg.split("=", 1)[1].strip()
            GROQ_MODEL = f"openai/{val}" if val in ("gpt-oss-20b", "gpt-oss-120b", "gpt-oss-safeguard-20b") else val




# Face Recognition & Name Learning
FACE_RECOGNITION_ENABLED = True
FACE_REGISTRY_PATH = os.path.join(BASE_DIR, "faces.json")
FACE_RECOGNITION_TOLERANCE = 0.55                              # Euclidean distance threshold (lower = stricter)
FACE_RECOGNITION_INTERVAL = 0.5                                # Interval between face encodings (seconds)

WHISPER_MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", os.path.join(MODELS_DIR, "whisper-tiny.en"))
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny.en")
SILERO_VAD_MODEL_PATH = os.environ.get("SILERO_VAD_MODEL_PATH", os.path.join(MODELS_DIR, "silero_vad.jit"))
SAMPLE_RATE = 16000
BLOCK_SIZE = 512                                              # 512 samples (32ms) - exact native Silero VAD frame size
MIN_TRANSCRIPT_CHARS = 3                                      # Ignore short noise/blips
VAD_SPEECH_CONFIDENCE = float(os.environ.get("VAD_SPEECH_CONFIDENCE", "0.35"))  # Silero speech threshold
VAD_SILENCE_TIMEOUT = float(os.environ.get("VAD_SILENCE_TIMEOUT", "0.35"))   # 350ms of silence triggers turn
VAD_POST_SPEECH_GRACE_MS = int(os.environ.get("VAD_POST_SPEECH_GRACE_MS", "200")) # Snappy grace period
MIN_SPEECH_DURATION = float(os.environ.get("MIN_SPEECH_DURATION", "0.20"))  # Ignore blips < 200ms
BARGE_IN_ENABLED = False                                      # Set True when using headphones/AirPods
BARGE_IN_VAD_CONFIDENCE = 0.70                                # Silero VAD confidence required during speech
BARGE_IN_ENERGY_MULT = 3.5                                    # Energy gate multiplier fallback

USE_KOKORO_ONNX = bool(int(os.environ.get("USE_KOKORO_ONNX", "0"))) if os.environ.get("USE_KOKORO_ONNX", "0").isdigit() else os.environ.get("USE_KOKORO_ONNX", "").lower() in ("true", "yes", "1")
KOKORO_MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", os.path.join(MODELS_DIR, "kokoro_q4.onnx"))
KOKORO_VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", os.path.join(MODELS_DIR, "voices-v1.0.bin"))
TTS_LANG_CODE = "a"                                           # 'a' = American English, 'b' = British English
TTS_VOICE = os.environ.get("TTS_VOICE", "af_bella")           # Kokoro voice (54 voices available)
TTS_SAMPLE_RATE = 24000
SPEAK_THOUGHTS = False                                        # If True, speaks spontaneous internal thoughts
TTS_STREAMING = True                                          # Stream LLM tokens -> parallel sentence synthesis
PROSODY_SENTENCE_BOUNDARIES = r'[.!?]+'


THINK_INTERVAL_SECONDS = 5                                    # How often background thoughts evaluate
RECENT_WINDOW_SECONDS = 180                                   # Working memory context window for thoughts
VISION_CONTEXT_WINDOW_SECONDS = 8                             # Visual object window for dialogue context
INTERACTION_FACE_WINDOW = 15                                  # Window for pairing face with speech
IDLE_SLEEP_MINUTES = 20                                       # Inactivity minutes before sleep consolidation

MEMORY_DB_PATH = os.path.join(BASE_DIR, "memory.db")
ARCHIVE_DIR = os.path.join(BASE_DIR, "memory_archive")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", os.path.join(MODELS_DIR, "all-MiniLM-L6-v2"))
EMBED_DIM = 384
