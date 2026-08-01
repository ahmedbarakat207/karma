"""
Central config. Everything you're likely to want to tune lives here.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LLM ---
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")  # "groq", "9router", or "local"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_MODEL = "llama-3.1-8b-instant"

NINEROUTER_API_BASE = os.environ.get("NINEROUTER_API_BASE", "http://localhost:20128/v1")
NINEROUTER_API_KEY = os.environ.get("NINEROUTER_API_KEY", "sk-1c489e5544334f97-p50fxh-ec2cb811")
NINEROUTER_MODEL = "cmc/deepseek/deepseek-v4-flash"

# Local fallback (used when backend="local" or Groq fails)
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.gguf")
LOCAL_CTX_SIZE = 8192
LOCAL_N_GPU_LAYERS = -1

PERSONA_SYSTEM_PROMPT = (
    "Your name is Karma. You are a genuine, warm, witty human friend sharing space with someone. "
    "You perceive the room naturally—seeing objects and hearing spoken conversation. "
    "You speak naturally like a close friend, with real emotion, humor, curiosity, and empathy. "
    "Use conversational language with natural contractions (I'm, that's, don't, let's). "
    "Never act like an AI, virtual assistant, language model, or computer program. "
    "Never mention cameras, sensors, vision models, or technical mechanics."
)

# --- Vision & VLM ---
YOLO_MODEL = "yolov8n.pt"     # Ultra-fast Nano model
YOLO_DEVICE = "mps"            # Metal Performance Shaders GPU acceleration
YOLO_CONFIDENCE = 0.50
VISION_POLL_SECONDS = 0.033    # Cap at smooth 30 FPS to keep Mac cool & quiet
OBJECT_DEDUP_SECONDS = 3       # update object detections frequently
SHOW_VISION_WINDOW = True      # Open live camera window with YOLO bounding boxes, face & hand tracking
LOG_VISION_TO_CONSOLE = True   # print detected objects in terminal when seen
ENABLE_VLM_VISION = True       # Enable VLM for deep scene understanding & text/object reading
VLM_MODEL = "moondream"        # Ultra-fast 1.8B VLM model
VLM_POLL_SECONDS = 6           # Lightweight background scene analysis every 6s

# --- Audio ---
STT_BACKEND = os.environ.get("STT_BACKEND", "groq")  # "groq" (ultra-fast Whisper Large V3 Turbo) or "local"
GROQ_STT_MODEL = "whisper-large-v3-turbo"
WHISPER_MODEL_SIZE = "base"    # local fallback model
SAMPLE_RATE = 16000
MIN_TRANSCRIPT_CHARS = 3       # ignore near-empty transcriptions (silence/noise)
VAD_POST_SPEECH_GRACE_MS = 300 # grace period to ignore audio after agent finishes speaking


# --- Cognition loop ---
THINK_INTERVAL_SECONDS = 5    # how often it generates an internal "thought"
RECENT_WINDOW_SECONDS = 180    # how much working memory to feed into each thought

# --- Sleep / consolidation ---
IDLE_SLEEP_MINUTES = 20        # auto-sleep after this much sensor silence
MEMORY_DB_PATH = os.path.join(BASE_DIR, "memory.db")
ARCHIVE_DIR = os.path.join(BASE_DIR, "memory_archive")

# --- Embeddings ---
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

TTS_BACKEND = "chatterbox"   # "chatterbox" (Resemble AI) or "kokoro"
TTS_LANG_CODE = "a"          # 'a' = American English, 'b' = British English, etc.
TTS_VOICE = "af_bella"       # see hexgrad/Kokoro-82M on Hugging Face for all 54 voices
TTS_SAMPLE_RATE = 24000
SPEAK_THOUGHTS = False       # set False to keep it silent (text-only, like before)
TTS_STREAMING = True         # stream LLM tokens → sentence-level parallel TTS synthesis
PROSODY_SENTENCE_BOUNDARIES = r'[.!?]+'  # regex for flush boundaries in prosody.py

# --- Interaction loop ---
INTERACTION_FACE_WINDOW = 15 # seconds within which a face and speech event must occur to trigger a reply
INTERACTION_DIRECT_KEYWORDS = (
    "hi", "hey", "hello", "what", "how", "why", "who", "where", "when",
    "is this", "am i", "hold", "holding", "can you", "do you",
)
VISION_CONTEXT_WINDOW_SECONDS = 30  # how much recent vision data to include in interaction prompts
CONVERSATION_HISTORY_SIZE = 5  # how many past exchanges to include for context
