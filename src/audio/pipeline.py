"""
Audio & Speech-to-Text Subsystem ("The Ears").
100% In-Process Offline Microphone Capture, Neural Silero VAD, and faster-whisper.
"""
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
BLOCK_SIZE = getattr(config, "BLOCK_SIZE", 512)  # 512 samples (32ms) - exact native Silero VAD window size

# Common Whisper hallucinations when transcribing near-silence or background noise
HALLUCINATIONS: Set[str] = {
    "thank you.", "thank you", "thanks for watching.", "thanks for watching!",
    "subtitles by", "amara.org", "subscribe", "bye.", "you", "okay.", "so",
    "thank you for watching.", "listening", "unbelievable.", "foreign",
    "i'm sorry.", "i'm sorry", "sorry.", "sorry", "aachman! no?", "aachman",
    "see my last thing okay so", "aachman! no", "aachman no", "thank you for listening."
}

# In-process PyTorch / Silero VAD neural voice activity detector
_torch = None
_silero_vad_model = None
try:
    import torch as _torch
    vad_path = getattr(config, "SILERO_VAD_MODEL_PATH", "")
    if vad_path and os.path.exists(vad_path):
        _silero_vad_model = _torch.jit.load(vad_path)
        print(f"[audio] Silero VAD loaded from {vad_path}!")
    else:
        _silero_vad_model, _ = _torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        print("[audio] Silero VAD neural voice activity detector initialized!")
except Exception as e:
    print(f"[audio] Silero VAD init note: {e}")


def audio_to_wav_bytes(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Converts normalized float32 numpy audio array to 16-bit PCM WAV bytes."""
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
    """Validate transcription against minimum length and hallucination filters."""
    if not text or len(text) < config.MIN_TRANSCRIPT_CHARS:
        return False
    lower = text.lower().strip().strip(".").strip("!").strip("?")
    if lower in HALLUCINATIONS or text.lower().strip() in HALLUCINATIONS:
        return False
    for bad in ["subtitles by", "amara.org", "thanks for watching", "thank you for watching", "aachman"]:
        if bad in lower:
            return False
    return True


def transcribe_audio(audio_np: np.ndarray, local_whisper) -> Optional[str]:
    """Transcribes audio purely in-process using local faster-whisper with ultra-low latency settings."""
    if not local_whisper or len(audio_np) == 0:
        return None
    try:
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        if not audio_np.flags["C_CONTIGUOUS"]:
            audio_np = np.ascontiguousarray(audio_np)

        segments, _ = local_whisper.transcribe(
            audio_np,
            language="en",
            task="transcribe",
            vad_filter=False,                    # Silero VAD already accurately sliced speech
            beam_size=1,                         # Fast greedy search
            best_of=1,                           # No sampling passes
            temperature=0.0,                     # Deterministic
            without_timestamps=True,             # Skips timestamp token prediction (~30-40% speedup)
            word_timestamps=False,               # Skips alignment DTW (~50% speedup)
            condition_on_previous_text=False,    # Skips prompt conditioning (~20% speedup)
            compression_ratio_threshold=None,   # Disable fallback compression check
            log_prob_threshold=None,            # Disable fallback retry on low confidence
            no_speech_threshold=None,           # Handled by Silero VAD
            initial_prompt=None,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if is_valid_transcript(text):
            return text
    except Exception as e:
        print(f"[audio] Local STT error: {e}")
    return None


class AudioPipeline:
    """Manages local microphone capture, VAD state machine, and in-process transcription."""

    def __init__(self, memory, stop_event, speaking_event=None, interrupt_event=None):
        self.memory = memory
        self.stop_event = stop_event
        self.speaking_event = speaking_event
        self.interrupt_event = interrupt_event
        self.local_whisper = None

        try:
            from faster_whisper import WhisperModel
            whisper_path = getattr(config, "WHISPER_MODEL_PATH", "")
            if not os.path.exists(whisper_path):
                whisper_path = "tiny.en"
            threads = getattr(config, "N_THREADS", 4)
            print(f"[audio] loading local faster-whisper from {whisper_path} with {threads} threads...")
            self.local_whisper = WhisperModel(
                whisper_path,
                device="cpu",
                compute_type="int8",
                cpu_threads=threads,
                num_workers=1,
            )
            # Warm up engine on startup to eliminate cold-start delay
            try:
                dummy = np.zeros(16000, dtype=np.float32)
                self.local_whisper.transcribe(dummy, language="en", beam_size=1, without_timestamps=True)
            except Exception:
                pass
            print("[audio] local faster-whisper ready and warmed up!")
        except Exception as e:
            print(f"[audio] local whisper init error: {e}")

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

        print(f"[audio] ready -- streaming microphone audio locally (Barge-in: {getattr(config, 'BARGE_IN_ENABLED', False)})")

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

                    # Loud sound spike detection (ignored when agent is speaking or in echo cooldown)
                    if not is_agent_speaking and energy > (bg_energy * 10) and energy > 0.05:
                        self.memory.add(kind="conscious_trigger", text="Loud noise detected!", salience=0.9)

                    threshold = max(bg_energy * SPEECH_MULT, 0.003)

                    # Neural Silero VAD or energy gate
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

                    # When agent is speaking and barge-in is disabled, ignore self-audio echo completely
                    if is_agent_speaking and not getattr(config, "BARGE_IN_ENABLED", False):
                        is_speech = False
                        if is_speaking:
                            is_speaking = False
                            self.memory.set_user_speaking(False)
                            speech_buffer = []
                            pre_buffer = np.zeros(0, dtype=np.float32)

                    # Barge-in handling
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
                                        self.memory.add(kind="speech", text=text, counts_as_activity=True)
                                pre_buffer = np.zeros(0, dtype=np.float32)
        except Exception as e:
            print(f"[audio] stream error: {e}")


def run_audio(memory, stop_event, speaking_event=None, interrupt_event=None) -> None:
    """Entrypoint for the background audio thread."""
    pipeline = AudioPipeline(memory, stop_event, speaking_event, interrupt_event)
    pipeline.run()
