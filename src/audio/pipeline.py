
import io
import os
import queue
import time
import warnings
import wave
from typing import Optional, Set
import numpy as np
import sounddevice as sd

warnings.filterwarnings("ignore")

from src import config

SAMPLE_RATE = 16000
BLOCK_SIZE = getattr(config, "BLOCK_SIZE", 512)

HALLUCINATIONS: Set[str] = {
    "thank you.", "thank you", "thanks for watching.", "thanks for watching!",
    "subtitles by", "amara.org", "subscribe", "bye.", "you", "okay.", "so",
    "thank you for watching.", "listening", "i'm sorry.", "i'm sorry", "sorry.", "sorry",
}

_torch = None
_silero_vad_model = None
try:
    import torch as _torch
    vad_path = getattr(config, "SILERO_VAD_MODEL_PATH", "")
    if vad_path and os.path.exists(vad_path):
        _silero_vad_model = _torch.jit.load(vad_path)
        config.log_debug(f"[audio] Silero VAD loaded from {vad_path}!")
    else:
        _silero_vad_model, _ = _torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        config.log_debug("[audio] Silero VAD neural voice activity detector initialized!")
except Exception as e:
    config.log_debug(f"[audio] Silero VAD init note: {e}")



def audio_to_wav_bytes(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    buf.seek(0)
    return buf.read()


def is_valid_transcript(text: Optional[str]) -> bool:
    if not text or len(text) < config.MIN_TRANSCRIPT_CHARS:
        return False
    lower = text.lower().strip().strip(".").strip("!").strip("?")
    if lower in HALLUCINATIONS or text.lower().strip() in HALLUCINATIONS:
        return False
    for bad in ["subtitles by", "amara.org", "thanks for watching", "thank you for watching"]:
        if bad in lower:
            return False
    return True


def _groq_stt_available() -> bool:
    if not getattr(config, "STT_USE_GROQ", True):
        return False
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


# Reused STT client: avoids a fresh TLS handshake per utterance (~2s saved
# on the first call after idle; steady-state then ~0.6s).
_groq_stt_client = None


def _get_groq_stt_client():
    global _groq_stt_client
    if _groq_stt_client is not None:
        return _groq_stt_client
    try:
        from groq import Groq
        timeout = float(getattr(config, "GROQ_STT_TIMEOUT", 8.0))
        try:
            _groq_stt_client = Groq(api_key=os.environ["GROQ_API_KEY"].strip(), timeout=timeout)
        except TypeError:
            _groq_stt_client = Groq(api_key=os.environ["GROQ_API_KEY"].strip())
        return _groq_stt_client
    except Exception as e:
        config.log_debug(f"[audio] Groq STT client note: {e}")
        return None


def transcribe_via_groq(audio_np: np.ndarray) -> Optional[str]:
    """Cloud STT via Groq (whisper-large-v3-turbo). ~0.6s warm, zero local CPU.

    This is the primary path when a key is set — it is the 'lighter STT':
    no faster-whisper load, no Pi CPU burn. Returns None on any failure so
    the caller falls back to local tiny.
    """
    if len(audio_np) == 0 or not _groq_stt_available():
        return None
    try:
        wav = audio_to_wav_bytes(
            np.ascontiguousarray(audio_np.astype(np.float32))
            if audio_np.dtype != np.float32 else audio_np
        )
        model = getattr(config, "GROQ_STT_MODEL", "whisper-large-v3-turbo")
        timeout = float(getattr(config, "GROQ_STT_TIMEOUT", 8.0))
        client = _get_groq_stt_client()
        if client is None:
            return None
        lang = (getattr(config, "WHISPER_LANGUAGE", "") or "").strip() or None
        kwargs: dict = {
            "file": ("speech.wav", wav),
            "model": model,
            "response_format": "text",
            "temperature": 0.0,
        }
        if lang:
            kwargs["language"] = lang
        try:
            res = client.audio.transcriptions.create(**kwargs, timeout=timeout)
        except TypeError:
            res = client.audio.transcriptions.create(**kwargs)
        text = res if isinstance(res, str) else (getattr(res, "text", "") or "")
        text = (text or "").strip()
        if is_valid_transcript(text):
            config.log_debug(f"[audio] Groq STT: '{text[:60]}'")
            return text
    except Exception as e:
        config.log_debug(f"[audio] Groq STT note: {e}")
    return None


def transcribe_local(audio_np: np.ndarray, local_whisper) -> Optional[str]:
    if not local_whisper or len(audio_np) == 0:
        return None
    try:
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        if not audio_np.flags["C_CONTIGUOUS"]:
            audio_np = np.ascontiguousarray(audio_np)

        target_lang = getattr(config, "WHISPER_LANGUAGE", "") or None
        # tiny.en only knows English — force en instead of auto-detect.
        model_hint = str(getattr(config, "WHISPER_MODEL_SIZE", "tiny"))
        if model_hint.endswith(".en") and not target_lang:
            target_lang = "en"
        segments, _ = local_whisper.transcribe(
            audio_np,
            language=target_lang,
            task="transcribe",
            vad_filter=False,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            without_timestamps=True,
            word_timestamps=False,
            condition_on_previous_text=False,
            compression_ratio_threshold=None,
            log_prob_threshold=None,
            no_speech_threshold=None,
            initial_prompt=None,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if is_valid_transcript(text):
            return text
    except Exception as e:
        print(f"[audio] Local STT error: {e}")
    return None


def transcribe_audio(audio_np: np.ndarray, local_whisper) -> Optional[str]:
    if len(audio_np) == 0:
        return None
    # Cloud first (fast + light), local tiny fallback (offline).
    text = transcribe_via_groq(audio_np)
    if text:
        return text
    return transcribe_local(audio_np, local_whisper)


class AudioPipeline:

    def __init__(self, memory, stop_event, speaking_event=None, interrupt_event=None):
        self.memory = memory
        self.stop_event = stop_event
        self.speaking_event = speaking_event
        self.interrupt_event = interrupt_event
        self.local_whisper = None

        # Local fallback STT (offline). Cloud Groq is primary when keyed,
        # so local is best-effort only — never let it crash the audio loop.
        # Lighter default tiny.en (~75MB, English-only, no lang-detect) is
        # ~2x faster than multilingual tiny on Pi 4 CPU.
        try:
            from faster_whisper import WhisperModel
            candidates: list = []
            disk_path = getattr(config, "WHISPER_MODEL_PATH", "")
            if disk_path and os.path.exists(disk_path):
                candidates.append(disk_path)
            # Preferred size from env (.env sets tiny.en), then safe fallbacks.
            for name in (
                getattr(config, "WHISPER_MODEL_SIZE", "tiny.en"),
                getattr(config, "WHISPER_MODEL_SIZE_LIGHT", "tiny.en"),
                "tiny.en",
                "tiny",
            ):
                if name and name not in candidates:
                    candidates.append(name)
            threads = getattr(config, "N_THREADS", 2)
            last_err = None
            for cand in candidates:
                try:
                    config.log_debug(f"[audio] loading local faster-whisper '{cand}' with {threads} threads...")
                    try:
                        self.local_whisper = WhisperModel(
                            cand,
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=threads,
                            num_workers=1,
                        )
                    except Exception:
                        # Some ARM ctranslate2 builds reject int8 — retry default.
                        self.local_whisper = WhisperModel(
                            cand,
                            device="cpu",
                            cpu_threads=threads,
                            num_workers=1,
                        )
                    try:
                        dummy = np.zeros(16000, dtype=np.float32)
                        wlang = getattr(config, "WHISPER_LANGUAGE", "") or None
                        if str(cand).endswith(".en") and not wlang:
                            wlang = "en"
                        self.local_whisper.transcribe(dummy, language=wlang, beam_size=1, without_timestamps=True)
                    except Exception:
                        pass
                    config.log_debug(f"[audio] local faster-whisper ready ({cand})!")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    self.local_whisper = None
                    continue
            if self.local_whisper is None and last_err is not None:
                config.log_debug(f"[audio] local whisper init error: {last_err}")
                if _groq_stt_available():
                    config.log_debug("[audio] continuing with Groq cloud STT only.")
        except Exception as e:
            config.log_debug(f"[audio] local whisper init error: {e}")

    def run(self):
        q: queue.Queue = queue.Queue()
        grace_frames = int(getattr(config, "VAD_POST_SPEECH_GRACE_MS", 200) / 100)
        state = {"was_speaking": False, "grace_frames": 0}

        def callback(indata, frames, time_info, status):
            barge_in = getattr(config, "BARGE_IN_ENABLED", False)
            if not barge_in and self.speaking_event and self.speaking_event.is_set():
                state["was_speaking"] = True
                state["grace_frames"] = grace_frames
                return

            if state["was_speaking"]:
                state["was_speaking"] = False

            if not barge_in and state["grace_frames"] > 0:
                state["grace_frames"] -= 1
                return

            q.put(indata.copy())

        config.log_debug(f"[audio] ready -- streaming microphone audio locally (Barge-in: {getattr(config, 'BARGE_IN_ENABLED', False)})")


        bg_energy = 0.002
        alpha = 0.95
        SPEECH_MULT = 2.2
        SILENCE_TIMEOUT = getattr(config, "VAD_SILENCE_TIMEOUT", 0.35)
        MIN_SPEECH_DURATION = getattr(config, "MIN_SPEECH_DURATION", 0.20)

        pre_buffer_max = int(0.8 * SAMPLE_RATE)
        pre_buffer = np.zeros(0, dtype=np.float32)
        speech_buffer = []
        is_speaking = False
        silence_duration = 0.0
        barge_in_consec = 0

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=BLOCK_SIZE, callback=callback):
                while not self.stop_event.is_set():
                    try:
                        data = q.get(timeout=0.2).flatten()
                    except queue.Empty:
                        continue

                    energy = float(np.abs(data).mean())

                    from src.state import internal_state
                    now = time.time()
                    time_since_speech = now - getattr(internal_state, "last_audio_played_time", 0.0)
                    is_agent_speaking = bool(
                        (self.speaking_event and self.speaking_event.is_set())
                        or getattr(internal_state, "is_playing_audio", False)
                        or (time_since_speech < 0.6)
                    )

                    if not is_agent_speaking and energy > (bg_energy * 10) and energy > 0.05:
                        self.memory.add(kind="conscious_trigger", text="Loud noise detected!", salience=0.9)

                    threshold = max(bg_energy * SPEECH_MULT, 0.003)

                    is_speech = False
                    if _silero_vad_model is not None and _torch is not None:
                        try:
                            audio_tensor = _torch.from_numpy(data)
                            prob = _silero_vad_model(audio_tensor, SAMPLE_RATE).item()
                            req_conf = getattr(config, "BARGE_IN_VAD_CONFIDENCE", 0.70) if is_agent_speaking else getattr(config, "VAD_SPEECH_CONFIDENCE", 0.35)
                            is_speech = (prob >= req_conf)
                        except Exception:
                            gate = (bg_energy * getattr(config, "BARGE_IN_ENERGY_MULT", 3.5)) if is_agent_speaking else threshold
                            is_speech = (energy > gate)
                    else:
                        gate = (bg_energy * getattr(config, "BARGE_IN_ENERGY_MULT", 3.5)) if is_agent_speaking else threshold
                        is_speech = (energy > gate)

                    if is_agent_speaking and not getattr(config, "BARGE_IN_ENABLED", False):
                        is_speech = False
                        if is_speaking:
                            is_speaking = False
                            self.memory.set_user_speaking(False)
                            speech_buffer = []
                            pre_buffer = np.zeros(0, dtype=np.float32)

                    if is_agent_speaking and is_speech and getattr(config, "BARGE_IN_ENABLED", False):
                        barge_in_consec += 1
                        if barge_in_consec >= 4:
                            print("[audio] 🚨 User spoke over agent — stopping speech immediately!")
                            if self.interrupt_event:
                                self.interrupt_event.set()
                            if self.speaking_event:
                                self.speaking_event.clear()
                            is_agent_speaking = False
                            barge_in_consec = 0
                    else:
                        barge_in_consec = 0

                    if not is_speaking:
                        if not is_agent_speaking:
                            bg_energy = alpha * bg_energy + (1 - alpha) * energy
                            pre_buffer = np.concatenate((pre_buffer, data))
                            if len(pre_buffer) > pre_buffer_max:
                                pre_buffer = pre_buffer[-pre_buffer_max:]

                    if is_speech:
                        if not is_speaking:
                            is_speaking = True
                            self.memory.set_user_speaking(True)
                            speech_buffer = [pre_buffer] if len(pre_buffer) > 0 else []
                        speech_buffer.append(data)
                        silence_duration = 0.0
                    else:
                        if is_speaking:
                            speech_buffer.append(data)
                            silence_duration += (BLOCK_SIZE / SAMPLE_RATE)
                            if silence_duration >= SILENCE_TIMEOUT:
                                is_speaking = False
                                self.memory.set_user_speaking(False)
                                full_audio = np.concatenate(speech_buffer)
                                dur = len(full_audio) / SAMPLE_RATE
                                speech_buffer = []

                                if dur >= MIN_SPEECH_DURATION:
                                    text = transcribe_audio(full_audio, self.local_whisper)
                                    if text:
                                        print(f"[audio] heard: '{text}'")
                                        from src.state import internal_state
                                        from src.ui import events as _events
                                        internal_state.set_user_speech(text)
                                        self.memory.add(kind="speech", text=text, counts_as_activity=True)
                                        _events.post("heard", text)
                                        # Early ack so the face shows THINKING
                                        # within ~50ms (cognition picks it up
                                        # ≤0.2s later; LLM+TTS then take secs).
                                        try:
                                            internal_state.set_thinking(True)
                                            from src.ui.server import broadcast_state_threadsafe as _bcast
                                            _bcast()
                                        except Exception:
                                            pass
                                pre_buffer = np.zeros(0, dtype=np.float32)
        except Exception as e:
            config.log_debug(f"[audio] stream error: {e}")



def run_audio(memory, stop_event, speaking_event=None, interrupt_event=None) -> None:
    pipeline = AudioPipeline(memory, stop_event, speaking_event, interrupt_event)
    pipeline.run()
