#!/usr/bin/env python3
"""Time each Karma pipeline stage. Run: .venv/bin/python /tmp/test_pipeline_timing.py
Skips real Kokoro synth by default (28s on Pi) unless --with-local-tts.
"""
import os, sys, time, subprocess, tempfile
sys.path.insert(0, "/home/karma/karma")
os.chdir("/home/karma/karma")
rows = []
def rec(stage, secs, note=""):
    rows.append((stage, secs, note))
    print(f"{stage:28s} {secs:7.2f}s  {note}")

# key
key = ""
with open(".env") as f:
    for line in f:
        if line.strip().startswith("GROQ_API_KEY="):
            key = line.strip().split("=", 1)[1].strip()
os.environ["GROQ_API_KEY"] = key

# 1. embedder
t0=time.time()
from sentence_transformers import SentenceTransformer
from src import config
emb = SentenceTransformer(getattr(config, "EMBED_MODEL_PATH"))
rec("embedder load (one-time)", time.time()-t0, "36s typical, startup only")
for q in ["hi", "what is your favorite hobby and why do you like jazz music?"]:
    a=time.time(); v=emb.encode(q); b=time.time()
    rec(f"embed encode ({len(q)}ch)", b-a, f"dim {len(v)}")
    if len(q) > 10: shared_vec = v.tolist()

# 2. memory query
from src.memory.store import MemoryStore
s = MemoryStore()
a=time.time(); r=s.query(shared_vec, k=2, kind="memory"); b=time.time()
rec("memory query", b-a, f"{len(r)} hits")

# 3. RAG with shared vec (vs fresh encode)
from src.memory.rag import DocumentRAG
rag = DocumentRAG(store=s, embedder=emb)
a=time.time(); ctx=rag.get_rag_context("what is your favorite hobby?", k=2, query_vec=shared_vec); b=time.time()
rec("rag retrieve (shared vec)", b-a, f"{len(ctx)} chars ctx")
a=time.time(); _v2=emb.encode("what is your favorite hobby?").tolist(); b=time.time()
rec("embed encode (2nd, saved)", b-a, "eliminated by sharing")

# 4. Groq LLM
from src.cognition.engine import GroqEngine
g = GroqEngine()
a=time.time(); txt=g.chat("You are Karma, witty chill friend. 1-2 sentences.", "What is your favorite hobby?", max_tokens=75); b=time.time()
rec("groq chat", b-a, f"{len(txt)} chars")
a=time.time(); first=None; toks=[]
for tok in g.stream_chat("You are Karma, witty chill friend. 1-2 sentences.", "What is your favorite hobby?", max_tokens=75):
    if first is None: first=time.time()
    toks.append(tok)
c=time.time()
rec("groq stream TTFT", (first-a) if first else -1, "")
rec("groq stream total", c-a, f"{len(''.join(toks))} chars")

# 5. Groq STT (espeak file, warm x2)
import numpy as np
wav = "/tmp/timing_speech.wav"
subprocess.run(["espeak-ng","-v","en","-s","175","-w",wav,"hello karma testing one two three"],
               check=True, capture_output=True, timeout=30)
import soundfile as sf
audio, sr = sf.read(wav, dtype="float32")
if len(audio.shape) > 1: audio = audio.mean(axis=1)
if sr != 16000:
    import scipy.signal
    audio = scipy.signal.resample_poly(audio, 16000, sr).astype(np.float32)
from src.audio.pipeline import transcribe_via_groq
for i in range(2):
    a=time.time(); t=transcribe_via_groq(audio); b=time.time()
    rec(f"groq STT run{i} ({'cold' if i==0 else 'warm'})", b-a, repr((t or "")[:40]))

# 6. TTS chunking + decode (no synth)
from src.speech.tts import TTSEngine
long_txt = "Hello there! " * 30  # 390 chars -> must chunk
chunks = TTSEngine._chunk_for_groq(long_txt)
rec("tts groq chunking", 0.0, f"{len(long_txt)}ch -> {len(chunks)} chunks, max {max(len(c) for c in chunks)}ch")
# wav decode speed with synthetic wav
import io, wave
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000)
    wf.writeframes((np.sin(np.arange(48000)/48000*2*np.pi*440)*10000).astype(np.int16).tobytes())
a=time.time(); arr=TTSEngine._wav_bytes_to_float24k(buf.getvalue()); b=time.time()
rec("tts wav decode+resample", b-a, f"{len(arr) if arr is not None else 0} samples @24k")

# 7. TTS cache logic (mocked synth, no 28s wait)
import threading
tts = TTSEngine.__new__(TTSEngine)
tts._tts_cache = {}; tts._synth_lock = threading.Lock()
tts._groq_tts_client = None; tts._groq_tts_warned = False
tts.interrupt_event = None
tts._synthesize_groq = lambda text, speed=1.0: np.zeros(2400, dtype=np.float32)
from src.speech.tts import clean_for_speech
a=time.time()
aud1 = tts._synthesize("Hello cache test", speed=1.0)
mid=time.time()
aud2 = tts._synthesize("Hello cache test", speed=1.0)
c=time.time()
rec("tts cache miss (mock)", mid-a, f"{len(aud1) if aud1 is not None else 0} samples")
rec("tts cache hit (mock)", c-mid, f"{len(aud2) if aud2 is not None else 0} samples")

print("\n--- pipeline estimate (steady-state, warm) ---")
print("voice: VAD 0.35 + STT 0.64 + embed 0.49(shared) + LLM 0.70 + TTS Groq ~1.0 = ~3.2s")
print("voice (local TTS fallback): ... + TTS local 28s = ~30s  <- current reality until Orpheus terms accepted")
print("inject (no VAD/STT/TTS-wait): embed 0.49 + LLM 0.70 = ~1.2s + overhead ~= 2.6s measured")
