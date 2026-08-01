"""
Fully local text-to-speech via Kokoro-82M (Apache-2.0, hexgrad/Kokoro-82M).

Public interface:
  speak(text)            — legacy blocking call (synthesise + play, full reply)
  speak_chunk(text)      — synthesise + play a single sentence (convenience)
  _synthesize(text)      — synthesise → numpy array only (used by prosody.py)
  _play_audio(audio)     — play a pre-synthesised array (used by prosody.py)

Thread-safety notes:
  - _synthesize is guarded by _synth_lock: Kokoro's KPipeline / PyTorch
    forward passes are not safe to call from multiple threads concurrently.
  - _play_audio does NOT touch speaking_event. The prosody.py pipeline owns
    that event for the full duration of a reply so the mic stays muted
    across all sentence chunks, not just one.
  - speak() still manages speaking_event internally for backwards compat.
"""
import re
import threading
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

import config


def clean_for_speech(text):
    if not text:
        return ""
    # Strip stage directions in asterisks, brackets, or parentheses
    text = re.sub(r"\*.*?\*", "", text)
    if getattr(config, "TTS_BACKEND", "kokoro") != "chatterbox":
        text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    # Remove special non-printable symbols & emojis
    if getattr(config, "TTS_BACKEND", "kokoro") == "chatterbox":
        text = re.sub(r"[^\w\s.,!?'\-\[\]]", "", text)
    else:
        text = re.sub(r"[^\w\s.,!?'\-]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TTSEngine:
    def __init__(self, lang_code=config.TTS_LANG_CODE, voice=config.TTS_VOICE,
                 speaking_event=None):
        self.backend = getattr(config, "TTS_BACKEND", "kokoro")
        if self.backend == "chatterbox":
            import torch
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            self.chatterbox = ChatterboxTurboTTS.from_pretrained(device=device)
        else:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=lang_code)

        self.voice = voice
        self.speaking_event = speaking_event
        # Guards concurrent synthesis calls — models are not thread-safe.
        # prosody.py uses a single synth_thread so this is belt-and-suspenders,
        # but it protects against any future call-site changes.
        self._synth_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Low-level primitives (used by prosody.py for streaming pipeline)
    # ------------------------------------------------------------------

    def _synthesize(self, text: str, speed: float = 1.0):
        """
        Synthesise text and return a numpy float32 audio array.
        Does NOT play audio. Serialised by _synth_lock for thread safety.
        Returns None if synthesis fails or text is empty after cleaning.
        """
        spoken = clean_for_speech(text)
        if not spoken:
            return None
        with self._synth_lock:
            try:
                if self.backend == "chatterbox":
                    out = self.chatterbox.generate(spoken)
                    audio = out[0] if isinstance(out, tuple) else out
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu().numpy()
                    return audio.squeeze()
                else:
                    chunks = []
                    for _, _, audio in self.pipeline(spoken, voice=self.voice, speed=speed):
                        if audio is not None and len(audio) > 0:
                            chunks.append(audio)
                    if chunks:
                        return np.concatenate(chunks)
            except Exception as e:
                print(f"[speech] synthesis error: {e}")
        return None

    def _play_audio(self, audio: np.ndarray):
        """
        Play a pre-synthesised numpy audio array.

        Does NOT touch speaking_event — that lifecycle is owned by
        prosody_stream() for the entire reply, keeping the mic muted across
        all sentence chunks.
        """
        if audio is None or len(audio) == 0:
            return
        try:
            sd.stop()
            from state import internal_state
            internal_state.is_playing_audio = True
            sd.play(audio, config.TTS_SAMPLE_RATE)
            sd.wait()
            internal_state.is_playing_audio = False
        except Exception as e:
            from state import internal_state
            internal_state.is_playing_audio = False
            print(f"[speech] playback error: {e}")

    # ------------------------------------------------------------------
    # Convenience wrapper (synthesise + play in one call)
    # ------------------------------------------------------------------

    def speak_chunk(self, text: str, speed: float = 1.0):
        """Synthesise and immediately play a single sentence chunk."""
        audio = self._synthesize(text, speed=speed)
        if audio is not None:
            self._play_audio(audio)

    # ------------------------------------------------------------------
    # Legacy blocking interface (unchanged for backwards compatibility)
    # ------------------------------------------------------------------

    def speak(self, text, speed=1.0):
        """
        Synthesise full text and play it in one blocking call.
        Manages speaking_event internally (legacy path, not used by prosody.py).
        """
        spoken_text = clean_for_speech(text)
        if not spoken_text:
            return

        if self.speaking_event:
            self.speaking_event.set()

        try:
            with self._synth_lock:
                if self.backend == "chatterbox":
                    out = self.chatterbox.generate(spoken_text)
                    audio = out[0] if isinstance(out, tuple) else out
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu().numpy()
                    audio_chunks = [audio.squeeze()]
                else:
                    generator = self.pipeline(spoken_text, voice=self.voice, speed=speed)
                    audio_chunks = []
                    for _, _, audio in generator:
                        if audio is not None and len(audio) > 0:
                            audio_chunks.append(audio)

            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                sd.stop()
                from state import internal_state
                internal_state.is_playing_audio = True
                sd.play(full_audio, config.TTS_SAMPLE_RATE)
                sd.wait()
                internal_state.is_playing_audio = False
        except Exception as e:
            from state import internal_state
            internal_state.is_playing_audio = False
            print(f"[speech] tts error: {e}")
        finally:
            if self.speaking_event:
                import time
                time.sleep(0.2)
                self.speaking_event.clear()
