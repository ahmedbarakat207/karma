"""
Karma Central Configuration.
100% Offline & Local Execution.
All Models Stored Locally Inside the models/ Directory.
"""
import os
import warnings

# Suppress framework & dependency warnings across all modules
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ApplePersistenceIgnoreState"] = "YES"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"
warnings.filterwarnings("ignore")

import torch

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# CPU thread allocation (4 cores on Raspberry Pi 4)
N_THREADS = int(os.environ.get("N_THREADS", str(min(4, os.cpu_count() or 4))))

# Auto-detect best compute device (MPS on Apple Silicon, CPU on Raspberry Pi)
_DEFAULT_YOLO_DEVICE = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
_DEFAULT_GPU_LAYERS = -1 if _DEFAULT_YOLO_DEVICE == "mps" else 0

# ==============================================================================
# 1. Local Language Model (llama-cpp-python)
# ==============================================================================
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf"))
HF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
HF_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
CTX_SIZE = int(os.environ.get("CTX_SIZE", "2048"))            # 2048 gives 3x faster prompt eval & low RAM on ARM
N_BATCH = int(os.environ.get("N_BATCH", "512"))               # Fast prompt batching
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", str(_DEFAULT_GPU_LAYERS)))

# Speculative Decoding (Disabled by default to avoid numpy stride broadcast bug in streaming multi-turn)
SPECULATIVE_DECODING = os.environ.get("SPECULATIVE_DECODING", "none").lower()
SPECULATIVE_NGRAM_SIZE = int(os.environ.get("SPECULATIVE_NGRAM_SIZE", "2"))
SPECULATIVE_NUM_PRED_TOKENS = int(os.environ.get("SPECULATIVE_NUM_PRED_TOKENS", "8"))

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

# ==============================================================================
# 2. Local Vision & Object Understanding
# ==============================================================================
YOLO_MODEL = os.environ.get("YOLO_MODEL", os.path.join(MODELS_DIR, "yolov8n.pt"))
HAND_LANDMARKER_MODEL = os.environ.get("HAND_LANDMARKER_MODEL", os.path.join(MODELS_DIR, "hand_landmarker.task"))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", _DEFAULT_YOLO_DEVICE)
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "320"))         # 320x320 runs 4x faster on CPU than 640x640
YOLO_CONFIDENCE = 0.50
VISION_POLL_SECONDS = 0.033                                   # Target ~30 FPS
OBJECT_DEDUP_SECONDS = 3                                      # Interval before repeating same object in memory
SHOW_VISION_WINDOW = True                                     # Live camera window with bounding boxes & HUD
LOG_VISION_TO_CONSOLE = True                                  # Log detected objects to terminal

# Face Recognition & Name Learning
FACE_RECOGNITION_ENABLED = True
FACE_REGISTRY_PATH = os.path.join(BASE_DIR, "faces.json")
FACE_RECOGNITION_TOLERANCE = 0.55                              # Euclidean distance threshold (lower = stricter)
FACE_RECOGNITION_INTERVAL = 0.5                                # Interval between face encodings (seconds)

# ==============================================================================
# 3. Local Speech-to-Text (STT) & VAD
# ==============================================================================
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

# ==============================================================================
# 4. Local Speech Synthesis (TTS) & Prosody
# ==============================================================================
KOKORO_MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", os.path.join(MODELS_DIR, "kokoro_q4.onnx"))
KOKORO_VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", os.path.join(MODELS_DIR, "voices-v1.0.bin"))
TTS_LANG_CODE = "a"                                           # 'a' = American English, 'b' = British English
TTS_VOICE = os.environ.get("TTS_VOICE", "af_bella")           # Kokoro voice (54 voices available)
TTS_SAMPLE_RATE = 24000
SPEAK_THOUGHTS = False                                        # If True, speaks spontaneous internal thoughts
TTS_STREAMING = True                                          # Stream LLM tokens -> parallel sentence synthesis
PROSODY_SENTENCE_BOUNDARIES = r'[.!?]+'

# ==============================================================================
# 5. Cognition, Memory & Sleep Consolidation
# ==============================================================================
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
