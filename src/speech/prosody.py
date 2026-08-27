"""
Streaming Prosody Pipeline ("The Expressive Voice").
Streams LLM tokens into a real-time JSON prefix parser, extracts emotion/inflection/text_chunks,
and orchestrates parallel TTS synthesis and sounddevice playback with minimal latency.
"""
import queue
import re
import threading
import time
from typing import Generator, List, Tuple, Optional, Any
import numpy as np
import sounddevice as sd

from src import config
from src.state import internal_state

# Regex matching sentence terminal punctuation for natural clause splitting
_SENTENCE_BOUNDARY_RE = re.compile(getattr(config, "PROSODY_SENTENCE_BOUNDARIES", r'[.!?]+'))

# Speed multiplier lookup table by emotion/inflection
_SPEED_MAP = {
    ("excited", "excited"): 1.15,
    ("excited", "emphatic"): 1.10,
    ("playful", "excited"): 1.10,
    ("playful", None): 1.05,
    ("curious", "question"): 1.05,
    ("curious", None): 1.00,
    ("warm", None): 0.95,
    ("attentive", None): 1.00,
    ("tired", "whisper"): 0.85,
    ("tired", None): 0.88,
    ("sad", "whisper"): 0.85,
    ("sad", None): 0.90,
    ("angry", "emphatic"): 1.12,
    ("scared", "whisper"): 1.15,
    ("surprised", "excited"): 1.12,
}


def _resolve_speed(emotion: Optional[str], inflection: Optional[str]) -> float:
    """Determine synthesis playback rate based on emotional and inflection cues."""
    e = (emotion or "").lower().strip()
    i = (inflection or "").lower().strip()

    if (e, i) in _SPEED_MAP:
        return _SPEED_MAP[(e, i)]
    if (e, None) in _SPEED_MAP:
        return _SPEED_MAP[(e, None)]
    if (None, i) in _SPEED_MAP:
        return _SPEED_MAP[(None, i)]

    return 1.0


def _flush_at_boundary(text: str) -> Tuple[List[str], str]:
    """Split text at sentence terminal boundaries."""
    if not text:
        return [], ""

    matches = list(_SENTENCE_BOUNDARY_RE.finditer(text))
    if not matches:
        return [], text

    last_match = matches[-1]
    split_pos = last_match.end()

    if split_pos < len(text) and text[split_pos - 1] == '.' and text[split_pos].isdigit():
        return [], text

    completed = text[:split_pos].strip()
    tail = text[split_pos:].lstrip()

    chunks = [completed]
    return chunks, tail


class _JSONPrefixParser:
    """Streaming parser that extracts emotion, inflection, and text_chunks from JSON tokens."""

    def __init__(self):
        self.emotion: Optional[str] = None
        self.inflection: Optional[str] = None
        self.text_chunks: List[str] = []

        self._in_object = False
        self._in_string = False
        self._string_buf = ""
        self._current_key: Optional[str] = None
        self._in_chunks_array = False
        self._array_closed = False
        self._escape_next = False

    def feed(self, token: str) -> List[str]:
        """Feed a new string token and return any newly extracted text chunks."""
        if self._array_closed:
            return []
        new_chunks: List[str] = []
        for ch in token:
            extracted = self._feed_char(ch)
            if extracted:
                new_chunks.extend(extracted)
        return new_chunks

    def _feed_char(self, ch: str) -> List[str]:
        results: List[str] = []

        # Wait until outer JSON object begins
        if not self._in_object:
            if ch == '{':
                self._in_object = True
            return results

        if self._escape_next:
            self._escape_next = False
            if self._in_string:
                self._string_buf += ch
            return results

        if ch == '\\' and self._in_string:
            self._escape_next = True
            return results

        if (ch == ']' or ch == '}') and not self._in_string:
            self._array_closed = True
            self._in_chunks_array = False
            if self._string_buf.strip():
                val = self._string_buf.strip()
                if val not in ("emotion", "inflection", "text_chunks"):
                    results.append(val)
                self._string_buf = ""
            return results

        if ch == '"':
            if not self._in_string:
                self._in_string = True
                self._string_buf = ""
            else:
                self._in_string = False
                val = self._string_buf.strip()
                if self._current_key is None and not self._in_chunks_array:
                    self._current_key = val
                elif self._current_key == "emotion":
                    self.emotion = val
                    self._current_key = None
                elif self._current_key == "inflection":
                    self.inflection = val
                    self._current_key = None
                elif self._current_key in ("response", "reply", "message", "text"):
                    if val:
                        self.text_chunks.append(val)
                        results.append(val)
                    self._current_key = None
                elif self._in_chunks_array:
                    if val:
                        self.text_chunks.append(val)
                        results.append(val)
                self._string_buf = ""
            return results

        if ch == '[' and self._current_key in ("text_chunks", "response", "reply", "message", "text") and not self._in_string:
            self._in_chunks_array = True
            self._current_key = None
            return results

        if self._in_string:
            self._string_buf += ch

        return results




def prosody_stream(token_iter: Generator[str, None, None], tts: Any, verbose: bool = True) -> None:
    """
    Main streaming entrypoint.
    LLM Tokens -> JSON Parser -> Parallel Synthesis Queue -> Sounddevice Output.
    """
    parser = _JSONPrefixParser()
    interrupt_event = getattr(tts, "interrupt_event", None)
    speaking_event = getattr(tts, "speaking_event", None)

    if interrupt_event:
        interrupt_event.clear()
    if speaking_event:
        speaking_event.set()

    synth_queue: queue.Queue = queue.Queue()
    audio_queue: queue.Queue = queue.Queue()

    def synth_worker():
        try:
            while True:
                if interrupt_event and interrupt_event.is_set():
                    while not synth_queue.empty():
                        try:
                            synth_queue.get_nowait()
                        except Exception:
                            break
                    break

                try:
                    item = synth_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if item is None or (interrupt_event and interrupt_event.is_set()):
                    break

                text, speed = item
                try:
                    audio = tts._synthesize(text, speed=speed)
                    if audio is not None and not (interrupt_event and interrupt_event.is_set()):
                        audio_queue.put(audio)
                except Exception as e:
                    config.log_debug(f"[prosody] synth error: {e}")
        finally:
            audio_queue.put(None)

    def audio_drain_worker():
        try:
            with sd.OutputStream(samplerate=config.TTS_SAMPLE_RATE, channels=1, dtype="float32") as stream:
                while True:
                    if interrupt_event and interrupt_event.is_set():
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except Exception:
                                break
                        break

                    try:
                        audio = audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    if audio is None or (interrupt_event and interrupt_event.is_set()):
                        break

                    try:
                        internal_state.set_playing_audio(True)
                        audio_arr = np.ascontiguousarray(audio, dtype=np.float32)
                        stream.write(audio_arr)
                    except Exception as e:
                        config.log_debug(f"[prosody] playback error: {e}")
                    finally:
                        if audio_queue.empty():
                            internal_state.set_playing_audio(False)
        except Exception as e:
            # Fallback to standard play if continuous stream cannot open
            config.log_debug(f"[prosody] stream open fallback: {e}")
            while True:
                if interrupt_event and interrupt_event.is_set():
                    break
                try:
                    audio = audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if audio is None:
                    break
                tts._play_audio(audio)
        finally:
            internal_state.set_playing_audio(False)

    synth_t = threading.Thread(target=synth_worker, daemon=True, name="prosody_synth")
    drain_t = threading.Thread(target=audio_drain_worker, daemon=True, name="prosody_drain")
    synth_t.start()
    drain_t.start()


    emotion_logged = False
    speed = 1.0

    try:
        for token in token_iter:
            if interrupt_event and interrupt_event.is_set():
                break

            new_chunks = parser.feed(token)

            if not emotion_logged and (parser.emotion or parser.inflection):
                emotion_logged = True
                internal_state.current_emotion = parser.emotion
                speed = _resolve_speed(parser.emotion, parser.inflection)
                if verbose and getattr(config, "DEBUG", False):
                    e_str = parser.emotion or "neutral"
                    i_str = f"/{parser.inflection}" if parser.inflection else ""
                    print(f"[prosody] feeling: {e_str}{i_str} (speed={speed:.2f}x)")

            for chunk in new_chunks:
                if interrupt_event and interrupt_event.is_set():
                    break
                synth_queue.put((chunk, speed))

    except Exception as e:
        config.log_debug(f"[prosody] token loop error: {e}")

    finally:
        synth_queue.put(None)
        synth_t.join(timeout=3.0)
        drain_t.join(timeout=4.0)
        if speaking_event:
            speaking_event.clear()
        internal_state.current_emotion = None
