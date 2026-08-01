"""
Prosody-aware streaming layer between the LLM and TTS engine.

Architecture: two-stage pipeline with guaranteed ordering.

  ┌─ Main loop ─────────────────────────────────────────────────────┐
  │  LLM token stream → JSON prefix parser → sentence text_buffer  │
  │  → flush at first sentence boundary → synth_queue              │
  └─────────────────────────────────────────────────────────────────┘
           ↓ (text, speed) tuples, in order
  ┌─ synth_thread ──────────────────────────────────────────────────┐
  │  synth_queue → Kokoro _synthesize() → audio_queue              │
  └─────────────────────────────────────────────────────────────────┘
           ↓ numpy audio arrays, in order
  ┌─ drain_thread ──────────────────────────────────────────────────┐
  │  audio_queue → _play_audio()                                    │
  └─────────────────────────────────────────────────────────────────┘

Why a single synth_thread:
  - Guarantees playback order without coordination overhead.
  - Kokoro KPipeline is not concurrent-safe.
  - Sentence N+1 is synthesised while sentence N plays — so first-audio
    latency = time to generate the first sentence only, not the full reply.

speaking_event lifecycle is owned entirely by prosody_stream, so the
mic-mute spans the whole reply — not just one chunk at a time.
"""
import re
import queue
import threading

# Sentence-ending boundaries — flush at the FIRST occurrence.
# Lookbehind (?<!\d) prevents matching dots in numbers (2.0, v3.1).
# Lookahead (?=\s|$) prevents matching abbreviation dots with no trailing space.
_SENTENCE_END = re.compile(r'(?<!\d)[.!?]+(?=\s|$)')

# Speed modifiers per inflection type
_INFLECTION_SPEED = {
    "question":  0.95,
    "excited":   1.08,
    "whisper":   0.88,
    "emphatic":  1.00,
    "flat":      1.00,
}

# Speed modifiers per emotion
_EMOTION_SPEED = {
    "curious":     0.97,
    "playful":     1.05,
    "tired":       0.90,
    "sad":         0.88,
    "surprised":   1.05,
    "warm":        0.95,
    "inquisitive": 0.97,
}


def _resolve_speed(emotion: str, inflection: str) -> float:
    """Combine emotion + inflection into a single TTS speed multiplier."""
    speed = 1.0
    speed *= _EMOTION_SPEED.get((emotion or "").lower(), 1.0)
    speed *= _INFLECTION_SPEED.get((inflection or "").lower(), 1.0)
    return max(0.8, min(1.2, speed))


class _JSONPrefixParser:
    """
    Incrementally scans the LLM token stream for the JSON envelope:
      { "emotion": "...", "inflection": "...", "text_chunks": [ ... ] }

    Extracts emotion + inflection as soon as those keys stream in — before
    the first text_chunk arrives — so speed can be set from the first word.

    Character-by-character parsing avoids waiting for the full JSON object.
    """

    def __init__(self):
        self._buf = ""
        self._state = "SCANNING"    # SCANNING → IN_JSON → DONE
        self._depth = 0
        self.emotion = "neutral"
        self.inflection = "flat"

        # text_chunks array tracking
        self._in_array = False       # entered the [ of text_chunks
        self._array_closed = False   # seen the closing ]
        self._in_string = False      # inside a JSON string within the array
        self._string_buf = ""
        self._escape_next = False

        # Guards: stop running regex once each field is found
        self._emotion_found = False
        self._inflection_found = False

    def feed(self, token: str):
        """Feed a raw token. Returns list of text strings ready for TTS."""
        results = []
        for ch in token:
            results.extend(self._process_char(ch))
        return results

    def _process_char(self, ch):
        if self._state == "SCANNING":
            if ch == '{':
                self._state = "IN_JSON"
                self._depth = 1
                self._buf = "{"
            return []

        if self._state == "IN_JSON":
            self._buf += ch
            if ch == '{':
                self._depth += 1
            elif ch == '}':
                self._depth -= 1
                if self._depth == 0:
                    self._state = "DONE"
                    return []
            self._try_extract_prefix()
            return self._scan_text_chunks(ch)

        return []

    def _try_extract_prefix(self):
        """
        Best-effort extraction of emotion/inflection from the partial buffer.
        Uses a key-presence check before running regex to avoid O(n²) work.
        """
        if not self._emotion_found and '"emotion"' in self._buf:
            m = re.search(r'"emotion"\s*:\s*"([^"]*)"', self._buf)
            if m:
                self.emotion = m.group(1)
                self._emotion_found = True

        if not self._inflection_found and '"inflection"' in self._buf:
            m = re.search(r'"inflection"\s*:\s*"([^"]*)"', self._buf)
            if m:
                self.inflection = m.group(1)
                self._inflection_found = True

    def _scan_text_chunks(self, ch):
        """
        Parse string elements from the text_chunks array char-by-char.

        Array entry detection: waits for the literal sequence
        `"text_chunks"` in the buffer followed by `[` in the suffix after
        that key — so a `[` inside earlier string values doesn't trigger it.

        Array close detection: tracks the `]` character (outside a string)
        and stops emitting once the array is closed, preventing any trailing
        JSON content from being sent to Kokoro as speech.
        """
        results = []

        # ── Not yet inside the array ───────────────────────────────────────
        if not self._in_array:
            if '"text_chunks"' in self._buf:
                suffix = self._buf.split('"text_chunks"')[-1]
                if '[' in suffix:
                    self._in_array = True
                    self._in_string = False
                    self._string_buf = ""
            return results

        # ── Array already closed — ignore everything ───────────────────────
        if self._array_closed:
            return results

        # ── Handle escape sequences ────────────────────────────────────────
        if self._escape_next:
            self._escape_next = False
            if self._in_string:
                self._string_buf += '\n' if ch == 'n' else '\t' if ch == 't' else ch
            return results

        if ch == '\\' and self._in_string:
            self._escape_next = True
            return results

        # ── Array close bracket ────────────────────────────────────────────
        if ch == ']' and not self._in_string:
            self._array_closed = True
            # Flush any unterminated string (shouldn't happen in valid JSON)
            if self._in_string and self._string_buf.strip():
                results.append(self._string_buf.strip())
            return results

        # ── String open/close ──────────────────────────────────────────────
        if ch == '"':
            if not self._in_string:
                self._in_string = True
                self._string_buf = ""
            else:
                self._in_string = False
                text = self._string_buf.strip()
                if text:
                    results.append(text)
                self._string_buf = ""
            return results

        if self._in_string:
            self._string_buf += ch

        return results


def prosody_stream(token_iter, tts, verbose=True):
    """
    Main entry point. Drives the prosody-aware streaming pipeline.

    Blocks until all tokens are consumed AND all audio has finished playing.

    Args:
        token_iter: Iterator of raw string tokens from the LLM stream.
        tts:        TTSEngine instance (must expose _synthesize / _play_audio).
        verbose:    Log emotion/inflection when detected.
    """
    parser = _JSONPrefixParser()

    synth_queue = queue.Queue()   # (text, speed) → synth thread
    audio_queue = queue.Queue()   # numpy arrays  → drain thread

    def synth_thread_fn():
        """
        Reads text chunks one-by-one, synthesises, feeds audio queue.

        The try/finally guarantees the poison pill is always forwarded to
        the drain thread — even if _synthesize raises unexpectedly — so
        drain_thread never blocks on audio_queue.get() indefinitely.
        """
        try:
            while True:
                item = synth_queue.get()
                if item is None:        # poison pill
                    break
                text, speed = item
                try:
                    audio = tts._synthesize(text, speed=speed)
                    if audio is not None:
                        audio_queue.put(audio)
                except Exception as e:
                    print(f"[prosody] synthesis error: {e}")
        finally:
            audio_queue.put(None)       # always forward poison pill


    def drain_thread_fn():
        """Reads synthesised audio in order and streams it gaplessly."""
        import sounddevice as sd
        import config
        from state import internal_state
        try:
            with sd.OutputStream(samplerate=config.TTS_SAMPLE_RATE, channels=1, dtype='float32') as stream:
                while True:
                    audio = audio_queue.get()
                    if audio is None:           # poison pill
                        break
                    try:
                        internal_state.is_playing_audio = True
                        # stream.write() blocks until the array is fully sent to the audio hardware,
                        # achieving perfect seamless playback across sentence chunks.
                        stream.write(audio)
                        internal_state.is_playing_audio = False
                    except Exception as e:
                        internal_state.is_playing_audio = False
                        print(f"[prosody] playback error: {e}")
        except Exception as e:
            print(f"[prosody] stream error: {e}")

    synth_t = threading.Thread(target=synth_thread_fn, daemon=True)
    drain_t = threading.Thread(target=drain_thread_fn, daemon=True)
    synth_t.start()
    drain_t.start()

    # Own the speaking_event for the entire reply — not per chunk.
    if tts.speaking_event:
        tts.speaking_event.set()

    try:
        text_buffer = ""
        emotion_logged = False
        speed = 1.0
        is_json_mode = None     # None = undecided, True = JSON, False = plain text
        raw_buffer = ""

        for token in token_iter:
            raw_buffer += token

            # Detect mode from the first non-whitespace character in the stream
            if is_json_mode is None:
                stripped = raw_buffer.lstrip()
                if stripped:
                    is_json_mode = stripped[0] == '{'

            if not is_json_mode:
                # ── Plain-text fallback: skip parser, flush at boundaries ──
                # No need to feed the parser — emotion/inflection stay at
                # defaults (neutral/flat), which is fine for unstructured output.
                text_buffer += token
                text_buffer, flushed = _flush_at_boundary(text_buffer)
                if flushed:
                    s = _resolve_speed(parser.emotion, parser.inflection)
                    synth_queue.put((flushed, s))
                continue

            # ── JSON mode: feed parser to get text_tokens ──────────────────
            text_tokens = parser.feed(token)

            if verbose and not emotion_logged:
                if parser._emotion_found or parser._inflection_found:
                    print(f"[prosody] emotion={parser.emotion} inflection={parser.inflection}")
                    emotion_logged = True
                    from state import internal_state
                    internal_state.current_emotion = parser.emotion

            speed = _resolve_speed(parser.emotion, parser.inflection)

            for text_token in text_tokens:
                text_buffer += text_token + " "
                text_buffer, flushed = _flush_at_boundary(text_buffer)
                if flushed:
                    synth_queue.put((flushed, speed))

        # Flush any text that didn't end with a sentence boundary
        remainder = text_buffer.strip()
        if remainder:
            synth_queue.put((remainder, speed))

    finally:
        # Signal synth thread to stop, wait for both threads to finish.
        # This runs even if token_iter raises — threads are always cleaned up.
        synth_queue.put(None)
        synth_t.join()
        drain_t.join()

        # Clear speaking_event after ALL audio has played
        if tts.speaking_event:
            import time
            time.sleep(0.2)
            tts.speaking_event.clear()

        from state import internal_state
        internal_state.current_emotion = None


def _flush_at_boundary(buf: str):
    """
    Split buf at the FIRST sentence-ending boundary (. ? !).
    Returns (remainder, flushed_sentence), or (buf, None) if no boundary.

    Using re.search (first match) sends sentences to TTS as soon as they
    arrive, rather than waiting for multiple sentences to accumulate.
    """
    match = _SENTENCE_END.search(buf)
    if match is None:
        return buf, None

    end_idx = match.end()
    flushed = buf[:end_idx].strip()
    remainder = buf[end_idx:].strip()
    if not flushed:
        return remainder, None
    return remainder, flushed
