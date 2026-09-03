import queue
import re
import threading
from typing import Generator, List, Optional, Any
import numpy as np
import sounddevice as sd

from src import config
from src.state import internal_state

_SENTENCE_RE = re.compile(getattr(config, "PROSODY_SENTENCE_BOUNDARIES", r'[.!?]+'))

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
    e = (emotion or "").lower().strip()
    i = (inflection or "").lower().strip()
    return _SPEED_MAP.get((e, i)) or _SPEED_MAP.get((e, None)) or _SPEED_MAP.get((None, i)) or 1.0


def _flush_at_boundary(text: str):
    if not text:
        return [], ""
    matches = list(_SENTENCE_RE.finditer(text))
    if not matches:
        return [], text
    split_pos = matches[-1].end()
    if split_pos < len(text) and text[split_pos - 1] == '.' and text[split_pos].isdigit():
        return [], text
    return [text[:split_pos].strip()], text[split_pos:].lstrip()


class _JSONPrefixParser:
    def __init__(self):
        self.emotion: Optional[str] = None
        self.inflection: Optional[str] = None
        self.text_chunks: List[str] = []

        self._in_object = False
        self._is_plain_text: Optional[bool] = None
        self._prefix_buf = ""
        self._plain_buf = ""
        self._in_string = False
        self._buf = ""
        self._key: Optional[str] = None
        self._in_array = False
        self._array_closed = False
        self._done = False
        self._escape = False

    def feed(self, token: str) -> List[str]:
        if self._done:
            return []

        if self._is_plain_text is None and not self._in_object:
            self._prefix_buf += token
            stripped = self._prefix_buf.strip()
            if "{" in self._prefix_buf:
                self._is_plain_text = False
                token = self._prefix_buf
                self._prefix_buf = ""
            elif stripped and not "```json".startswith(stripped):
                self._is_plain_text = True
                token = self._prefix_buf
                self._prefix_buf = ""
            else:
                return []

        if self._is_plain_text:
            self._plain_buf += token
            chunks, self._plain_buf = _flush_at_boundary(self._plain_buf)
            if chunks:
                self.text_chunks.extend(chunks)
            return chunks

        out = []
        for ch in token:
            got = self._char(ch)
            if got:
                out.extend(got)
        return out

    def flush(self) -> List[str]:
        if self._is_plain_text and self._plain_buf.strip():
            rem = self._plain_buf.strip()
            self._plain_buf = ""
            self.text_chunks.append(rem)
            return [rem]
        return []

    def _char(self, ch: str) -> List[str]:
        results = []

        if not self._in_object:
            if ch == '{':
                self._in_object = True
            return results

        if self._escape:
            self._escape = False
            if self._in_string:
                self._buf += ch
            return results

        if ch == '\\' and self._in_string:
            self._escape = True
            return results

        if (ch == ']' or ch == '}') and not self._in_string:
            if ch == ']':
                self._array_closed = True
            self._done = True
            self._in_array = False
            if self._buf.strip() and self._buf.strip() not in ("emotion", "inflection", "text_chunks"):
                results.append(self._buf.strip())
            self._buf = ""
            return results

        if ch == '"':
            if not self._in_string:
                self._in_string = True
                self._buf = ""
            else:
                self._in_string = False
                val = self._buf.strip()
                if self._key is None and not self._in_array:
                    self._key = val
                elif self._key == "emotion":
                    self.emotion = val
                    self._key = None
                elif self._key == "inflection":
                    self.inflection = val
                    self._key = None
                elif self._key in ("response", "reply", "message", "text"):
                    if val:
                        self.text_chunks.append(val)
                        results.append(val)
                    self._key = None
                elif self._in_array and val:
                    self.text_chunks.append(val)
                    results.append(val)
                self._buf = ""
            return results

        if ch == '[' and self._key in ("text_chunks", "response", "reply", "message", "text") and not self._in_string:
            self._in_array = True
            self._key = None
            return results

        if self._in_string:
            self._buf += ch

        return results


class CodeFilter:
    def __init__(self):
        self.in_code = False
        self.buf: List[str] = []
        self.lang = "code"

    def filter_chunk(self, chunk: str) -> List[str]:
        if not chunk:
            return []

        if self.in_code:
            if "```" in chunk:
                parts = chunk.split("```", 1)
                self.buf.append(parts[0])
                code = "".join(self.buf).strip()
                if code:
                    internal_state.set_active_code(code, lang=self.lang)
                self.in_code = False
                self.buf = []
                return self.filter_chunk(parts[1])
            self.buf.append(chunk)
            return []

        if "```" not in chunk:
            return [chunk]

        parts = chunk.split("```", 1)
        prefix = parts[0].strip()
        rest = parts[1]

        self.in_code = True
        self.buf = []

        nl = rest.find("\n")
        if nl != -1:
            maybe_lang = rest[:nl].strip()
            if maybe_lang and len(maybe_lang) < 15 and " " not in maybe_lang:
                self.lang = maybe_lang
                rest = rest[nl + 1:]
            else:
                self.lang = "code"
        else:
            self.lang = "code"

        if "```" in rest:
            body, after = rest.split("```", 1)
            self.buf.append(body)
            code = "".join(self.buf).strip()
            if code:
                internal_state.set_active_code(code, lang=self.lang)
            self.in_code = False
            self.buf = []
            out = ([prefix] if prefix else []) + self.filter_chunk(after)
            return out

        self.buf.append(rest)
        return [prefix] if prefix else []


def prosody_stream(token_iter: Generator[str, None, None], tts: Any, verbose: bool = True) -> None:
    parser = _JSONPrefixParser()
    code_filter = CodeFilter()
    interrupt_event = getattr(tts, "interrupt_event", None)
    speaking_event = getattr(tts, "speaking_event", None)

    if interrupt_event:
        interrupt_event.clear()
    if speaking_event:
        speaking_event.set()

    synth_q: queue.Queue = queue.Queue()
    audio_q: queue.Queue = queue.Queue()

    def synth_worker():
        try:
            while True:
                if interrupt_event and interrupt_event.is_set():
                    while not synth_q.empty():
                        try:
                            synth_q.get_nowait()
                        except Exception:
                            break
                    break

                try:
                    item = synth_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                if item is None or (interrupt_event and interrupt_event.is_set()):
                    break

                text, speed = item
                try:
                    audio = tts._synthesize(text, speed=speed)
                    if audio is not None and not (interrupt_event and interrupt_event.is_set()):
                        audio_q.put(audio)
                except Exception as e:
                    config.log_debug(f"[prosody] synth error: {e}")
        finally:
            audio_q.put(None)

    def drain_worker():
        try:
            with sd.OutputStream(samplerate=config.TTS_SAMPLE_RATE, channels=1, dtype="float32") as stream:
                while True:
                    if interrupt_event and interrupt_event.is_set():
                        while not audio_q.empty():
                            try:
                                audio_q.get_nowait()
                            except Exception:
                                break
                        break

                    try:
                        audio = audio_q.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    if audio is None or (interrupt_event and interrupt_event.is_set()):
                        break

                    try:
                        internal_state.set_playing_audio(True)
                        stream.write(np.ascontiguousarray(audio, dtype=np.float32))
                    except Exception as e:
                        config.log_debug(f"[prosody] playback error: {e}")
                    finally:
                        if audio_q.empty():
                            internal_state.set_playing_audio(False)
        except Exception as e:
            config.log_debug(f"[prosody] stream fallback: {e}")
            while True:
                if interrupt_event and interrupt_event.is_set():
                    break
                try:
                    audio = audio_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if audio is None:
                    break
                tts._play_audio(audio)
        finally:
            internal_state.set_playing_audio(False)

    synth_t = threading.Thread(target=synth_worker, daemon=True, name="prosody_synth")
    drain_t = threading.Thread(target=drain_worker, daemon=True, name="prosody_drain")
    synth_t.start()
    drain_t.start()

    emotion_logged = False
    speed = 1.0

    try:
        for token in token_iter:
            if interrupt_event and interrupt_event.is_set():
                break

            chunks = parser.feed(token)

            if not emotion_logged and (parser.emotion or parser.inflection):
                emotion_logged = True
                internal_state.current_emotion = parser.emotion
                speed = _resolve_speed(parser.emotion, parser.inflection)
                if verbose and getattr(config, "DEBUG", False):
                    inf = f"/{parser.inflection}" if parser.inflection else ""
                    print(f"[prosody] {parser.emotion or 'neutral'}{inf} @ {speed:.2f}x")

            for chunk in chunks:
                if interrupt_event and interrupt_event.is_set():
                    break
                for spoken in code_filter.filter_chunk(chunk):
                    if spoken.strip():
                        synth_q.put((spoken.strip(), speed))

    except Exception as e:
        config.log_debug(f"[prosody] token loop error: {e}")
    finally:
        for rem in parser.flush():
            for spoken in code_filter.filter_chunk(rem):
                if spoken.strip():
                    synth_q.put((spoken.strip(), speed))

        if code_filter.in_code and code_filter.buf:
            code = "".join(code_filter.buf).strip()
            if code:
                internal_state.set_active_code(code, lang=code_filter.lang)

        synth_q.put(None)
        synth_t.join(timeout=3.0)
        drain_t.join(timeout=4.0)
        if speaking_event:
            speaking_event.clear()
        internal_state.current_emotion = None
