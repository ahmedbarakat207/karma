import os
import sys
import time
import queue
import numpy as np
import pytest

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prosody import _flush_at_boundary, _JSONPrefixParser, _resolve_speed
from audio import is_valid_transcript, audio_to_wav_bytes
from interaction import _extract_plain_text, retrieve_memories
from llm_engine import FallbackEngine
from working_memory import WorkingMemory

# --- PROSODY TESTS ---

def test_flush_sentence():
    buf = "Hello there. How are you?"
    r, f = _flush_at_boundary(buf)
    assert f == "Hello there."
    assert r == "How are you?"

def test_flush_decimal_no_flush():
    buf = "Version 2.0 is out"
    r, f = _flush_at_boundary(buf)
    assert f is None
    assert r == buf

def test_flush_abbreviation_no_flush():
    buf = "e.g.something"
    r, f = _flush_at_boundary(buf)
    assert f is None
    assert r == buf

def test_flush_question():
    buf = "What? No way."
    r, f = _flush_at_boundary(buf)
    assert f == "What?"
    assert r == "No way."

def test_flush_no_boundary():
    buf = "Just a regular sentence without punctuation"
    r, f = _flush_at_boundary(buf)
    assert f is None
    assert r == buf

def test_json_parser_full():
    p = _JSONPrefixParser()
    stream = '{"emotion": "inquisitive", "inflection": "question", "text_chunks": ["Wait.", "You got that working?"]}'
    results = []
    for ch in stream:
        results.extend(p.feed(ch))
    assert p.emotion == "inquisitive"
    assert p.inflection == "question"
    assert results == ["Wait.", "You got that working?"]
    assert p._state == "DONE"

def test_json_parser_early_emotion():
    p = _JSONPrefixParser()
    for ch in '{"emotion": "playful"':
        p.feed(ch)
    assert p.emotion == "playful"
    assert p._emotion_found
    assert not p._inflection_found

def test_json_parser_array_closed():
    p = _JSONPrefixParser()
    full = '{"emotion": "warm", "inflection": "flat", "text_chunks": ["Hello world."]}'
    results = []
    for ch in full + '"trailing junk"':
        results.extend(p.feed(ch))
    assert results == ["Hello world."]
    assert p._array_closed

def test_json_parser_escape():
    p = _JSONPrefixParser()
    stream = '{"emotion": "neutral", "inflection": "flat", "text_chunks": ["He said \\"hello\\" to me."]}'
    results = []
    for ch in stream:
        results.extend(p.feed(ch))
    assert results == ['He said "hello" to me.']

def test_resolve_speed_range():
    # Should not go above 1.2 or below 0.8
    s1 = _resolve_speed("playful", "excited") # 1.05 * 1.08 = 1.134
    assert 1.1 < s1 < 1.2
    
    # "sad" (0.88) * "whisper" (0.88) = 0.7744, should clamp to 0.8
    s2 = _resolve_speed("sad", "whisper")
    assert s2 == 0.8

# --- AUDIO TESTS ---

def test_is_valid_transcript_real():
    assert is_valid_transcript("Hey, how is it going today?")

def test_is_valid_transcript_hallucination():
    assert not is_valid_transcript("Thanks for watching.")
    assert not is_valid_transcript("amara.org")
    assert not is_valid_transcript("aachman! no?")

def test_audio_to_wav_bytes():
    audio_np = np.zeros(16000, dtype=np.float32)
    wav_bytes = audio_to_wav_bytes(audio_np, sample_rate=16000)
    # Check for RIFF and WAVE headers
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes[:12]

# --- INTERACTION TESTS ---

def test_extract_plain_text_json():
    raw = '{"emotion": "neutral", "text_chunks": ["Hello.", "World."]}'
    res = _extract_plain_text(raw)
    assert res == "Hello. World."

def test_extract_plain_text_regex():
    # Malformed JSON (missing closing brace)
    raw = '{"emotion": "neutral", "text_chunks": ["Hello.", "World."]'
    res = _extract_plain_text(raw)
    assert res == "Hello. World."

def test_extract_plain_text_strip():
    # Complete garbage, but has some text
    raw = 'just some raw text here'
    res = _extract_plain_text(raw)
    assert res == 'just some raw text here'

def test_retrieve_memories_empty():
    res = retrieve_memories("hello", store=None, embedder=None)
    assert res == ""

# --- LLM ENGINE TESTS ---

class DummyPrimary:
    def chat(self, *a, **k): return "primary"
    def stream_chat(self, *a, **k): yield "primary"

class DummySecondary:
    def chat(self, *a, **k): return "secondary"
    def stream_chat(self, *a, **k): yield "secondary"

class TransientFailingPrimary:
    def chat(self, *a, **k): raise Exception("Error 429 rate limit")
    def stream_chat(self, *a, **k): raise Exception("Error 429 rate limit")

class PermanentFailingPrimary:
    def chat(self, *a, **k): raise Exception("401 Unauthorized")
    def stream_chat(self, *a, **k): raise Exception("401 Unauthorized")

def test_fallback_transient():
    fb = FallbackEngine(TransientFailingPrimary(), DummySecondary(), "TestPrimary")
    res = fb.chat("sys", "user")
    assert res == "secondary"
    assert not fb._primary_permanent_fail
    assert fb._primary_disabled_until > time.time()

def test_fallback_permanent():
    fb = FallbackEngine(PermanentFailingPrimary(), DummySecondary(), "TestPrimary")
    res = fb.chat("sys", "user")
    assert res == "secondary"
    assert fb._primary_permanent_fail

def test_fallback_recovery():
    fb = FallbackEngine(TransientFailingPrimary(), DummySecondary(), "TestPrimary")
    # Fake that the cooldown happened in the past
    fb._primary_disabled_until = time.time() - 10
    assert fb._primary_available()

# --- WORKING MEMORY TESTS ---

def test_memory_dedup():
    wm = WorkingMemory()
    wm.add(kind="object", text="saw a couch", dedup_seconds=10)
    wm.add(kind="object", text="saw a couch", dedup_seconds=10) # Should dedup
    assert len(wm.all_events()) == 1

def test_memory_mark_handled():
    wm = WorkingMemory()
    wm.add(kind="speech", text="Hello")
    events = wm.all_events()
    ts = events[0]["ts"]
    
    # Initially unhandled
    assert len(wm.unhandled_speech(0)) == 1
    
    wm.mark_handled(ts)
    assert len(wm.unhandled_speech(0)) == 0

def test_memory_conversation_ctx():
    wm = WorkingMemory()
    for i in range(10):
        wm.add_conversation(f"Q{i}", f"A{i}")
    
    ctx = wm.get_conversation_context(n=2)
    assert "User: Q8" in ctx
    assert "You: A8" in ctx
    assert "User: Q9" in ctx
    assert "You: A9" in ctx
    assert "User: Q7" not in ctx

# --- VISION TESTS ---

def test_startle_cooldown():
    # Emulate the cooldown logic from vision.py
    _startle_cooldowns = {}
    STARTLE_COOLDOWN_SECONDS = 10
    
    startle_items = ["couch"]
    now_t = time.time()
    
    cooled = [
        l for l in startle_items
        if now_t - _startle_cooldowns.get(l, 0) >= STARTLE_COOLDOWN_SECONDS
    ]
    
    assert "couch" in cooled
    _startle_cooldowns["couch"] = now_t
    
    # Try again immediately
    cooled2 = [
        l for l in startle_items
        if now_t - _startle_cooldowns.get(l, 0) >= STARTLE_COOLDOWN_SECONDS
    ]
    assert "couch" not in cooled2
