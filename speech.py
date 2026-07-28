"""
Fully local text-to-speech via Kokoro-82M (Apache-2.0, hexgrad/Kokoro-82M).
Optimized audio stream buffer management to prevent PortAudio creaking and silence.
"""
import re
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

import config


def clean_for_speech(text):
    if not text:
        return ""
    # Strip stage directions in asterisks, brackets, or parentheses
    text = re.sub(r"\*.*?\*", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    # Remove special non-printable symbols & emojis
    text = re.sub(r"[^\w\s.,!?'\-]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TTSEngine:
    def __init__(self, lang_code=config.TTS_LANG_CODE, voice=config.TTS_VOICE,
                 speaking_event=None):
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.speaking_event = speaking_event

    def speak(self, text, speed=1.0):
        spoken_text = clean_for_speech(text)
        if not spoken_text:
            return

        if self.speaking_event:
            self.speaking_event.set()

        try:
            generator = self.pipeline(spoken_text, voice=self.voice, speed=speed)
            audio_chunks = []
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    audio_chunks.append(audio)

            if audio_chunks:
                # Concatenate into one smooth, unfragmented audio buffer
                full_audio = np.concatenate(audio_chunks)
                # Clear lingering PortAudio device state to prevent buffer underflows/creaking
                sd.stop()
                sd.play(full_audio, config.TTS_SAMPLE_RATE)
                sd.wait()
        except Exception as e:
            print(f"[speech] tts error: {e}")
        finally:
            if self.speaking_event:
                import time
                time.sleep(0.2)
                self.speaking_event.clear()
