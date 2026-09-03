import os
import sys
import warnings
import torch

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


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except Exception:
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
        except Exception:
            pass

N_THREADS = int(os.environ.get("N_THREADS", str(min(4, os.cpu_count() or 4))))

_DEFAULT_YOLO_DEVICE = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
_DEFAULT_GPU_LAYERS = -1 if _DEFAULT_YOLO_DEVICE == "mps" else 0

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, "model.gguf"))
HF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
HF_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
CTX_SIZE = int(os.environ.get("CTX_SIZE", "4096"))
N_BATCH = int(os.environ.get("N_BATCH", "512"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", str(_DEFAULT_GPU_LAYERS)))

SPECULATIVE_DECODING = os.environ.get("SPECULATIVE_DECODING", "none").lower()
SPECULATIVE_NGRAM_SIZE = int(os.environ.get("SPECULATIVE_NGRAM_SIZE", "2"))
SPECULATIVE_NUM_PRED_TOKENS = int(os.environ.get("SPECULATIVE_NUM_PRED_TOKENS", "8"))

DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.environ.get("DEFAULT_TOP_P", "0.9"))
DEFAULT_REPEAT_PENALTY = float(os.environ.get("DEFAULT_REPEAT_PENALTY", "1.05"))
DEFAULT_FREQUENCY_PENALTY = float(os.environ.get("DEFAULT_FREQUENCY_PENALTY", "0.0"))
DEFAULT_PRESENCE_PENALTY = float(os.environ.get("DEFAULT_PRESENCE_PENALTY", "0.0"))

KV_CACHE_TYPE = os.environ.get("KV_CACHE_TYPE", "q8_0").lower()
FLASH_ATTN = _env_bool("FLASH_ATTN", True)

PERSONA_SYSTEM_PROMPT = "You are Karma, a companion in the room having a casual conversation."

YOLO_MODEL = os.environ.get("YOLO_MODEL", os.path.join(MODELS_DIR, "yolov8n.pt"))
HAND_LANDMARKER_MODEL = os.environ.get("HAND_LANDMARKER_MODEL", os.path.join(MODELS_DIR, "hand_landmarker.task"))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", _DEFAULT_YOLO_DEVICE)
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "320"))
YOLO_CONFIDENCE = 0.50
ENABLE_YOLO = _env_bool("ENABLE_YOLO", True)
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
VISION_POLL_SECONDS = 0.033
OBJECT_DEDUP_SECONDS = 3

DEBUG = _env_bool("DEBUG", False)
SHOW_VISION_WINDOW = _env_bool("SHOW_VISION_WINDOW", False)
LOG_VISION_TO_CONSOLE = _env_bool("LOG_VISION_TO_CONSOLE", False)
FULLSCREEN_FACE = _env_bool("FULLSCREEN_FACE", True)

USE_GROQ = _env_bool("USE_GROQ", False)
_raw_groq = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MODEL = f"openai/{_raw_groq}" if _raw_groq in ("gpt-oss-20b", "gpt-oss-120b", "gpt-oss-safeguard-20b") else _raw_groq


def log_debug(*args, **kwargs) -> None:
    if DEBUG:
        print(*args, **kwargs)


def apply_cli_args(argv=None) -> None:
    global DEBUG, SHOW_VISION_WINDOW, LOG_VISION_TO_CONSOLE, FULLSCREEN_FACE, USE_GROQ, GROQ_MODEL
    if argv is None:
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


FACE_RECOGNITION_ENABLED = True
FACE_REGISTRY_PATH = os.path.join(BASE_DIR, "faces.json")
FACE_RECOGNITION_TOLERANCE = 0.55
FACE_RECOGNITION_INTERVAL = 0.5

WHISPER_MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", os.path.join(MODELS_DIR, "whisper-tiny.en"))
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny.en")
SILERO_VAD_MODEL_PATH = os.environ.get("SILERO_VAD_MODEL_PATH", os.path.join(MODELS_DIR, "silero_vad.jit"))
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
MIN_TRANSCRIPT_CHARS = 3
VAD_SPEECH_CONFIDENCE = float(os.environ.get("VAD_SPEECH_CONFIDENCE", "0.35"))
VAD_SILENCE_TIMEOUT = float(os.environ.get("VAD_SILENCE_TIMEOUT", "0.35"))
VAD_POST_SPEECH_GRACE_MS = int(os.environ.get("VAD_POST_SPEECH_GRACE_MS", "200"))
MIN_SPEECH_DURATION = float(os.environ.get("MIN_SPEECH_DURATION", "0.20"))
BARGE_IN_ENABLED = False
BARGE_IN_VAD_CONFIDENCE = 0.70
BARGE_IN_ENERGY_MULT = 3.5

USE_KOKORO_ONNX = _env_bool("USE_KOKORO_ONNX", False)
KOKORO_MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", os.path.join(MODELS_DIR, "kokoro_q4.onnx"))
KOKORO_VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", os.path.join(MODELS_DIR, "voices-v1.0.bin"))
TTS_LANG_CODE = "a"
TTS_VOICE = os.environ.get("TTS_VOICE", "af_bella")
TTS_SAMPLE_RATE = 24000
SPEAK_THOUGHTS = False
TTS_STREAMING = True
PROSODY_SENTENCE_BOUNDARIES = r'[.!?]+'

THINK_INTERVAL_SECONDS = 5
RECENT_WINDOW_SECONDS = 180
VISION_CONTEXT_WINDOW_SECONDS = 8
INTERACTION_FACE_WINDOW = 15
IDLE_SLEEP_MINUTES = 20

MEMORY_DB_PATH = os.path.join(BASE_DIR, "memory.db")
ARCHIVE_DIR = os.path.join(BASE_DIR, "memory_archive")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", os.path.join(MODELS_DIR, "all-MiniLM-L6-v2"))
EMBED_DIM = 384
