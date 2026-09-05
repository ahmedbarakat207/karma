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

PERSONA_SYSTEM_PROMPT = (
    "You are Karma, a witty, chill friend hanging out in the room. "
    "You speak naturally like a real person with your own opinions and tastes. "
    "You love music, especially jazz, lo-fi, and indie rock. "
    "Never give textbook definitions or lecture like an encyclopedia. "
    "Never sound like a customer service bot or AI assistant. "
    "Keep replies brief (1-2 sentences) like a real casual conversation."
)

YOLO_MODEL = os.environ.get("YOLO_MODEL", os.path.join(MODELS_DIR, "yolov8n.pt"))
HAND_LANDMARKER_MODEL = os.environ.get("HAND_LANDMARKER_MODEL", os.path.join(MODELS_DIR, "hand_landmarker.task"))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", _DEFAULT_YOLO_DEVICE)
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "320"))
YOLO_CONFIDENCE = 0.50
ENABLE_YOLO = _env_bool("ENABLE_YOLO", True)
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
VISION_POLL_SECONDS = 0.033
OBJECT_DEDUP_SECONDS = 3

# On-demand VLM verifier (SmolVLM2, CPU, lazy-loaded). YOLO keeps tracking
# every frame; only genuinely novel YOLO labels trigger ONE snapshot.
VLM_ENABLED = _env_bool("VLM_ENABLED", True)
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
VLM_MODEL_DIR = os.environ.get("VLM_MODEL_DIR", os.path.join(MODELS_DIR, "smolvlm2-256m"))
VLM_COOLDOWN_SECONDS = float(os.environ.get("VLM_COOLDOWN_SECONDS", "20.0"))
VLM_MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "128"))
VLM_SNAPSHOT_WIDTH = int(os.environ.get("VLM_SNAPSHOT_WIDTH", "384"))
VLM_CORRECTION_TTL_SECONDS = float(os.environ.get("VLM_CORRECTION_TTL_SECONDS", "60.0"))

DEBUG = _env_bool("DEBUG", False)
SHOW_VISION_WINDOW = _env_bool("SHOW_VISION_WINDOW", False)
LOG_VISION_TO_CONSOLE = _env_bool("LOG_VISION_TO_CONSOLE", False)
FULLSCREEN_FACE = _env_bool("FULLSCREEN_FACE", True)

USE_ELECTRON = _env_bool("USE_ELECTRON", True)
UI_WS_HOST = os.environ.get("UI_WS_HOST", "127.0.0.1")
UI_WS_PORT = int(os.environ.get("UI_WS_PORT", "8765"))

# LAN dashboard: served on all interfaces so phones/laptops on the same
# wifi can reach it. Always password-gated (see server.py), the local
# Electron websocket above stays localhost-only and ungated.
UI_DASH_HOST = os.environ.get("UI_DASH_HOST", "0.0.0.0")
UI_DASH_PORT = int(os.environ.get("UI_DASH_PORT", "8080"))
KARMA_UI_PASSWORD = os.environ.get("KARMA_UI_PASSWORD", "")

# Dashboard shell tab: interactive PTY on the robot, same auth as the
# dashboard. Full shell as the service user — set 0 to disable.
SHELL_ENABLED = _env_bool("SHELL_ENABLED", True)
SHELL_IDLE_SECONDS = int(os.environ.get("SHELL_IDLE_SECONDS", "900"))
SHELL_MAX_SESSIONS = int(os.environ.get("SHELL_MAX_SESSIONS", "3"))

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
        elif clean in ("--no-electron", "--opencv-face"):
            USE_ELECTRON = False
        elif clean in ("--electron", "-e"):
            USE_ELECTRON = True
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

# Differential drive (2x 5840-31ZY worm motors + 2x BTS7960 H-bridges).
# BCM pin numbering. All overridable via env. Mock mode when pigpiod absent.
DRIVE_ENABLED = _env_bool("DRIVE_ENABLED", True)
DRIVE_LEFT_RPWM = int(os.environ.get("DRIVE_LEFT_RPWM", "19"))
DRIVE_LEFT_LPWM = int(os.environ.get("DRIVE_LEFT_LPWM", "26"))
DRIVE_LEFT_R_EN = int(os.environ.get("DRIVE_LEFT_R_EN", "16"))
DRIVE_LEFT_L_EN = int(os.environ.get("DRIVE_LEFT_L_EN", "20"))
DRIVE_RIGHT_RPWM = int(os.environ.get("DRIVE_RIGHT_RPWM", "6"))
DRIVE_RIGHT_LPWM = int(os.environ.get("DRIVE_RIGHT_LPWM", "5"))
DRIVE_RIGHT_R_EN = int(os.environ.get("DRIVE_RIGHT_R_EN", "22"))
DRIVE_RIGHT_L_EN = int(os.environ.get("DRIVE_RIGHT_L_EN", "27"))
DRIVE_PWM_FREQ = int(os.environ.get("DRIVE_PWM_FREQ", "20000"))
# Safety: duty clamp + per-command watchdog (worm gear, 60RPM, 65mm wheels
# => ~0.2 m/s max, so even full duty is walking pace).
DRIVE_MAX_DUTY = float(os.environ.get("DRIVE_MAX_DUTY", "0.6"))
DRIVE_MAX_SECONDS = float(os.environ.get("DRIVE_MAX_SECONDS", "3.0"))
DRIVE_CRUISE_DUTY = float(os.environ.get("DRIVE_CRUISE_DUTY", "0.35"))
DRIVE_TURN_DUTY = float(os.environ.get("DRIVE_TURN_DUTY", "0.4"))
# Explorer autonomy tunables.
EXPLORER_ENABLED = _env_bool("EXPLORER_ENABLED", True)
EXPLORER_TICK_SECONDS = float(os.environ.get("EXPLORER_TICK_SECONDS", "2.0"))
# Area fraction of a centered box that counts as blocked (640x480 frame).
OBSTACLE_AREA_RATIO = float(os.environ.get("OBSTACLE_AREA_RATIO", "0.12"))
OBSTACLE_CENTER_MARGIN = float(os.environ.get("OBSTACLE_CENTER_MARGIN", "0.25"))
OBSTACLE_COOLDOWN_SECONDS = float(os.environ.get("OBSTACLE_COOLDOWN_SECONDS", "3.0"))

WHISPER_MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", os.path.join(MODELS_DIR, "whisper-tiny"))
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "")
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
PROSODY_SENTENCE_BOUNDARIES = r'[.!?،؟]+'

# Nabra (Arabic Neural TTS)
NABRA_MODEL_DIR = os.environ.get("NABRA_MODEL_DIR", os.path.join(MODELS_DIR, "nabra"))
NABRA_REPO_ID = os.environ.get("NABRA_REPO_ID", "oddadmix/Nabra-82M-v0.1")
NABRA_VOICE = os.environ.get("NABRA_VOICE", "af_msa")
NABRA_ENABLED = _env_bool("NABRA_ENABLED", True)
AUTO_LANG_DETECT = _env_bool("AUTO_LANG_DETECT", True)

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
