import os
import sys
import threading
import time
import numpy as np
import pytest

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.speech.prosody import _flush_at_boundary, _JSONPrefixParser, _resolve_speed, prosody_stream
from src.audio.pipeline import is_valid_transcript, audio_to_wav_bytes, transcribe_audio
from src.cognition.interaction import _extract_plain_text, retrieve_memories, _extract_name_introduction, _deduplicate_phrase_loops

from src.cognition.engine import LocalEngine, GroqEngine, create_engine, _strip_thinking, _strip_thinking_from_stream

from src.memory.face_registry import FaceRegistry
from src.memory.working import WorkingMemory
from src.state import internal_state
from src.vision.render import FaceRenderer, VisionRenderer




def test_strip_thinking_tags():
    assert _strip_thinking("<think>reasoning here</think>Hello world!") == "Hello world!"
    assert _strip_thinking("<thought>thinking here</thought>Hi there!") == "Hi there!"
    assert _strip_thinking("<think>unfinished reasoning") == ""
    assert _strip_thinking("Direct response without tags") == "Direct response without tags"


def test_strip_thinking_stream():
    tokens = ["<thought>", "internal", " analysis", "</thought>", "Hey", " there", "!"]
    result = "".join(list(_strip_thinking_from_stream(tokens)))
    assert result == "Hey there!"

    think_tokens = ["<think>", "pondering", "</think>", "All", " good", "."]
    result2 = "".join(list(_strip_thinking_from_stream(think_tokens)))
    assert result2 == "All good."



def test_flush_sentence():
    buf = "Hello there. How are you?"
    chunks, tail = _flush_at_boundary(buf)
    assert len(chunks) > 0
    assert chunks[0] == "Hello there. How are you?"
    assert tail == ""


def test_flush_no_boundary():
    buf = "Just a regular sentence without punctuation"
    chunks, tail = _flush_at_boundary(buf)
    assert chunks == []
    assert tail == buf


def test_json_parser_full():
    p = _JSONPrefixParser()
    stream = '{"emotion": "inquisitive", "inflection": "question", "text_chunks": ["Wait.", "You got that working?"]}'
    results = []
    for ch in stream:
        results.extend(p.feed(ch))
    assert p.emotion == "inquisitive"
    assert p.inflection == "question"
    assert results == ["Wait.", "You got that working?"]
    assert p._array_closed


def test_json_parser_early_emotion():
    p = _JSONPrefixParser()
    for ch in '{"emotion": "playful"':
        p.feed(ch)
    assert p.emotion == "playful"
    assert p.inflection is None


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


def test_json_parser_markdown_fence():
    p = _JSONPrefixParser()
    stream = '```json\n{"emotion": "playful", "inflection": "excited", "text_chunks": ["Hey there!"]}\n```'
    results = []
    for ch in stream:
        results.extend(p.feed(ch))
    assert p.emotion == "playful"
    assert results == ["Hey there!"]


def test_resolve_speed_range():
    s1 = _resolve_speed("playful", "excited")
    assert 1.05 <= s1 <= 1.20

    s2 = _resolve_speed("tired", "whisper")
    assert 0.80 <= s2 <= 0.90




def test_is_valid_transcript_real():
    assert is_valid_transcript("Hey, how is it going today?")


def test_is_valid_transcript_hallucination():
    assert not is_valid_transcript("Thanks for watching.")
    assert not is_valid_transcript("amara.org")
    assert not is_valid_transcript("aachman! no?")


def test_audio_to_wav_bytes():
    audio_np = np.zeros(16000, dtype=np.float32)
    wav_bytes = audio_to_wav_bytes(audio_np, sample_rate=16000)
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes[:12]


def test_transcribe_audio_local():
    class MockSegment:
        def __init__(self, text):
            self.text = text

    class MockWhisper:
        def transcribe(self, *a, **k):
            return [MockSegment("Hello from local whisper")], None

    res = transcribe_audio(np.zeros(16000, dtype=np.float32), local_whisper=MockWhisper())
    assert res == "Hello from local whisper"



def test_extract_plain_text_json():
    raw = '{"emotion": "neutral", "text_chunks": ["Hello.", "World."]}'
    res = _extract_plain_text(raw)
    assert res == "Hello. World."


def test_extract_plain_text_regex():
    raw = '{"emotion": "neutral", "text_chunks": ["Hello.", "World."]'
    res = _extract_plain_text(raw)
    assert res == "Hello. World."


def test_extract_plain_text_strip():
    raw = 'just some raw text here'
    res = _extract_plain_text(raw)
    assert res == 'just some raw text here'


def test_extract_plain_text_markdown_fences():
    raw = '```json\n{"emotion": "playful", "text_chunks": ["Hey there!", "How are you?"]}\n```'
    res = _extract_plain_text(raw)
    assert res == "Hey there! How are you?"


def test_deduplicate_phrase_loops():
    text = "You know, I bet that was like, 'Ahhh! Ahhh! Ahhh! Ahhh! Ahhh! Ahhh!'"
    res = _deduplicate_phrase_loops(text)
    assert "Ahhh! Ahhh! Ahhh! Ahhh!" not in res

    text2 = "Oh, wow, hi there! It's like 'Hi, hi, hi, hi, hi, hi, hi'"
    res2 = _deduplicate_phrase_loops(text2)
    assert "hi, hi, hi, hi" not in res2


def test_retrieve_memories_empty():
    res = retrieve_memories("hello", store=None, embedder=None)
    assert res == ""




def test_local_engine_prompt_formatting():
    # Verify ChatML formatting logic
    class FakeLocalEngine(LocalEngine):
        def __init__(self):
            self.stop_tokens = ["<|im_end|>", "<|endoftext|>"]

    engine = FakeLocalEngine()
    formatted = engine._format_prompt("You are Karma.", "Hello!")
    assert "<|im_start|>system\nYou are Karma.<|im_end|>" in formatted
    assert "<|im_start|>user\nHello!<|im_end|>" in formatted
    assert "<|im_start|>assistant" in formatted



def test_memory_dedup():
    wm = WorkingMemory()
    wm.add(kind="object", text="saw a couch", dedup_seconds=10)
    wm.add(kind="object", text="saw a couch", dedup_seconds=10)
    assert len(wm.all_events()) == 1


def test_memory_mark_handled():
    wm = WorkingMemory()
    wm.add(kind="speech", text="Hello")
    events = wm.all_events()
    ts = events[0]["ts"]

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


def test_working_memory_recognized_people():
    wm = WorkingMemory()
    wm.set_recognized_people({"Mike", "Ahmed"})
    assert wm.get_recognized_people() == {"Mike", "Ahmed"}
    wm.set_recognized_people(set())
    assert wm.get_recognized_people() == set()



def test_face_registry_register_and_recognize(tmp_path):
    registry = FaceRegistry(path=str(tmp_path / "test_faces.json"))
    fake_encoding = np.random.randn(128).astype(np.float64)
    registry.register("Mike", fake_encoding)
    assert registry.count() == 1
    assert "Mike" in registry.known_names()

    result = registry.recognize(fake_encoding, tolerance=0.6)
    assert result == "Mike"


def test_face_registry_unknown(tmp_path):
    registry = FaceRegistry(path=str(tmp_path / "test_faces.json"))
    fake_encoding = np.random.randn(128).astype(np.float64)
    registry.register("Ahmed", fake_encoding)

    different = np.random.randn(128).astype(np.float64) * 10
    result = registry.recognize(different, tolerance=0.55)
    assert result is None


def test_face_registry_persistence(tmp_path):
    path = str(tmp_path / "test_faces.json")
    enc = np.random.randn(128).astype(np.float64)
    reg1 = FaceRegistry(path=path)
    reg1.register("Sarah", enc)
    del reg1

    reg2 = FaceRegistry(path=path)
    assert reg2.count() == 1
    assert "Sarah" in reg2.known_names()



def test_extract_name_im():
    assert _extract_name_introduction("Hey, I'm Mike") == "Mike"


def test_extract_name_my_name_is():
    assert _extract_name_introduction("My name is Ahmed") == "Ahmed"


def test_extract_name_call_me():
    assert _extract_name_introduction("Call me Sarah") == "Sarah"


def test_extract_name_none():
    assert _extract_name_introduction("How are you doing?") is None


def test_extract_name_false_positive():
    assert _extract_name_introduction("I'm doing fine") is None
    assert _extract_name_introduction("I'm just here") is None
    assert _extract_name_introduction("I'm going to go") is None



def test_internal_state_mood_decay():
    internal_state.energy = 0.90
    internal_state.curiosity = 0.80
    internal_state.update([])
    expr = internal_state.get_expression(is_talking=False)
    assert "(" in expr and ")" in expr



def test_barge_in_config():
    assert isinstance(getattr(config, "BARGE_IN_ENABLED", False), bool)
    assert getattr(config, "BARGE_IN_VAD_CONFIDENCE", 0.0) > 0.0
    assert getattr(config, "BARGE_IN_ENERGY_MULT", 0.0) > 0.0


def test_barge_in_prosody_interrupt():
    speaking_event = threading.Event()
    interrupt_event = threading.Event()

    class MockTTS:
        def __init__(self):
            self.speaking_event = speaking_event
            self.interrupt_event = interrupt_event
            self.synth_calls = []

        def _synthesize(self, text, speed=1.0):
            self.synth_calls.append(text)
            return np.zeros(1600, dtype=np.float32)

        def _play_audio(self, audio):
            pass

    mock_tts = MockTTS()

    def token_generator():
        yield '{"emotion": "warm", "inflection": "excited", "text_chunks": ["First long sentence here.'
        interrupt_event.set()
        yield '", "Second sentence that should never be spoken."]}'

    start = time.time()
    prosody_stream(token_generator(), mock_tts, verbose=False)
    elapsed = time.time() - start

    assert elapsed < 2.0
    assert speaking_event.is_set() is False



def test_face_renderer_canvas_shape():
    renderer = FaceRenderer(width=1280, height=720)
    frame = renderer.render(is_talking=False, is_user_speaking=False, fps=30.0)
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8


def test_face_renderer_custom_resolution():
    renderer = FaceRenderer(width=1920, height=1080)
    frame = renderer.render(target_shape=(1080, 1920))
    assert frame.shape == (1080, 1920, 3)


def test_face_renderer_all_moods():
    renderer = FaceRenderer(width=640, height=480)
    moods = ["playful", "curious", "excited", "attentive", "tired", "sad", "love", "angry"]
    for m in moods:
        internal_state.mood = m
        internal_state.current_emotion = m
        frame = renderer.render(is_talking=True, is_user_speaking=False)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    internal_state.current_emotion = None


def test_face_renderer_subtitles_overlay():
    renderer = FaceRenderer(width=800, height=600)
    internal_state.set_karma_speech("Hello there, my friend!")
    frame = renderer.render(is_talking=True)
    assert frame is not None
    assert frame.shape == (600, 800, 3)



def test_internal_state_subtitles():
    internal_state.set_user_speech("How are you?")
    sub = internal_state.get_active_subtitle(max_age=5.0)
    assert sub is not None
    assert sub[0] == "You"
    assert sub[1] == "How are you?"

    time.sleep(0.01)
    internal_state.set_karma_speech("I am feeling great!")
    sub2 = internal_state.get_active_subtitle(max_age=5.0)
    assert sub2 is not None
    assert sub2[0] == "Karma"
    assert sub2[1] == "I am feeling great!"


def test_internal_state_gaze_clamping():
    internal_state.set_gaze(2.5, -3.0, is_present=True)
    assert internal_state.gaze_x == 1.0
    assert internal_state.gaze_y == -1.0
    assert internal_state.is_user_present is True

    internal_state.set_gaze(0.2, 0.4, is_present=True)
    assert abs(internal_state.gaze_x - 0.2) < 1e-4
    assert abs(internal_state.gaze_y - 0.4) < 1e-4



def test_apply_cli_args_debug():
    config.apply_cli_args(["--debug"])
    assert config.DEBUG is True
    assert config.SHOW_VISION_WINDOW is True
    assert config.LOG_VISION_TO_CONSOLE is True


def test_apply_cli_args_windowed():
    config.apply_cli_args(["--windowed"])
    assert config.FULLSCREEN_FACE is False


def test_apply_cli_args_fullscreen():
    config.apply_cli_args(["--fullscreen"])
    assert config.FULLSCREEN_FACE is True


def test_apply_cli_args_camera_toggle():
    config.apply_cli_args(["--hide-camera"])
    assert config.SHOW_VISION_WINDOW is False
    config.apply_cli_args(["--camera"])
    assert config.SHOW_VISION_WINDOW is True


def test_apply_cli_args_groq():
    config.USE_GROQ = False
    config.apply_cli_args(["--groq"])
    assert config.USE_GROQ is True
    assert config.GROQ_MODEL == "openai/gpt-oss-20b"


def test_apply_cli_args_groq_custom_model():
    config.apply_cli_args(["--groq=llama-3.3-70b-versatile"])
    assert config.USE_GROQ is True
    assert config.GROQ_MODEL == "llama-3.3-70b-versatile"


def test_create_engine_groq():
    engine = create_engine(use_groq=True, model_name="gpt-oss-20b")
    assert isinstance(engine, GroqEngine)
    assert engine.model == "openai/gpt-oss-20b"



