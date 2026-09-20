#!/usr/bin/env python3
import os
import time
import pytest
from unittest.mock import MagicMock, patch

from src.speech.arabic_g2p import is_arabic, normalize_arabic_text, clean_phonemes, ArabicG2P
from src.speech.tts import clean_for_speech, TTSEngine
from src.speech.prosody import _flush_at_boundary
from src.cognition.interaction import _check_kiosk_intent


def test_arabic_detection():
    # Pure Arabic
    assert is_arabic("أهلاً وسهلاً") is True
    assert is_arabic("عامل إيه يا صاحبي؟") is True
    assert is_arabic("صباح الخير") is True

    # Pure English
    assert is_arabic("Hello, how are you?") is False
    assert is_arabic("Karma robot online.") is False
    assert is_arabic("12345!@#$") is False

    # Mixed text containing Arabic
    assert is_arabic("Hello يا صاحبي") is True
    assert is_arabic("") is False


def test_clean_for_speech_bilingual():
    # English cleaning
    en = clean_for_speech("Hello *world* [note] (ignore) it's 100% fine!")
    assert en == "Hello it's 100 fine!"

    # Arabic cleaning - must preserve Arabic letters, diacritics, and Arabic commas/question marks
    ar = clean_for_speech("أهلاً *بيك*، عامل إيه يا صاحبي؟ (ملاحظة)")
    assert "أهلاً" in ar
    assert "عامل إيه" in ar
    assert "،" in ar
    assert "؟" in ar
    assert "ملاحظة" not in ar
    assert "بيك" not in ar


def test_arabic_g2p_phonemizer():
    g2p = ArabicG2P(diacritize=False)
    # Test Arabic greeting phonemization
    ph = g2p("أهلاً يا صاحبي، عامل إيه؟")
    assert len(ph) > 0
    # Must preserve pharyngeals
    assert "ħ" in ph or "ʕ" in ph


def test_tts_routing_logic():
    from src import config as cfg
    with patch("kokoro.KPipeline") as mock_pipeline_cls, \
         patch.object(cfg, "TTS_ENGINE", "kokoro"):
        mock_pipe = MagicMock()
        mock_pipeline_cls.return_value = mock_pipe

        tts = TTSEngine()

        # Mock English and Arabic synthesis internal methods
        with patch.object(tts, "_synthesize_english", return_value=None) as mock_en, \
             patch.object(tts, "_synthesize_arabic", return_value=None) as mock_ar, \
             patch.object(tts, "_synthesize_groq", return_value=None):

            # English input
            tts._synthesize("Hello friend, how is the weather today?")
            mock_en.assert_called_once()
            mock_ar.assert_not_called()

            mock_en.reset_mock()
            mock_ar.reset_mock()

            # Arabic input
            tts._synthesize("أهلاً يا صاحبي، عامل إيه؟")
            mock_ar.assert_called_once()


def test_arabic_kiosk_intent():
    # Map commands in Arabic
    intent, floor = _check_kiosk_intent("افتح الخريطة لو سمحت")
    assert intent == "map"
    assert floor is None

    # Map floor 2 in Arabic
    intent, floor = _check_kiosk_intent("وريني الخريطة الدور التاني")
    assert intent == "map"
    assert floor == 1

    # Map floor 1 in Arabic
    intent, floor = _check_kiosk_intent("افتح خريطة الدور الاول")
    assert intent == "map"
    assert floor == 0

    # Achievements in Arabic
    intent, floor = _check_kiosk_intent("وريني الانجازات بتاعتنا")
    assert intent == "achievements"

    # Projects/Apps in Arabic
    intent, floor = _check_kiosk_intent("افتح المشاريع")
    assert intent == "apps"

    # Documents in Arabic
    intent, floor = _check_kiosk_intent("افتح الملفات والمستندات")
    assert intent == "docs"

    # Face/Close in Arabic
    intent, floor = _check_kiosk_intent("اقفل القائمة وارجع للوش")
    assert intent == "face"


def test_num2words_fallback():
    from src.speech.tts import _ensure_num2words
    import sys
    _ensure_num2words()
    assert "num2words" in sys.modules
    mod = sys.modules["num2words"]
    fn = getattr(mod, "num2words", None)
    assert callable(fn)
    assert fn(0) == "zero"
    assert fn(42) == "forty-two"
    assert "one hundred" in fn(105) and "five" in fn(105)
    assert "twenty" in fn(2026, to="year")
    assert fn(1, to="ordinal") == "first"
    assert fn(2, to="ordinal") == "second"
    assert fn(3, to="ordinal") == "third"
    assert fn(21, to="ordinal") in ("twenty-first", "twenty first")
    assert "three point" in fn(3.14)


def test_alsa_device_selection(monkeypatch):
    tts = TTSEngine()

    mock_aplay_output = """
card 0: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM vc4-hdmi-hifi-0 []
  Subdevices: 1/1
card 1: vc4hdmi1 [vc4-hdmi-1], device 0: MAI PCM vc4-hdmi-hifi-1 []
  Subdevices: 1/1
card 2: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]
  Subdevices: 1/1
card 3: UACDemo [USB Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
"""
    class MockProc:
        returncode = 0
        stdout = mock_aplay_output

    with patch("sys.platform", "linux"), \
         patch("shutil.which", return_value="/usr/bin/aplay"), \
         patch("subprocess.run", return_value=MockProc):

        # USB audio is highest priority
        devs = tts._find_best_alsa_devices()
        assert len(devs) >= 4
        assert devs[0] == "plughw:CARD=UACDemo,DEV=0"
        assert "plughw:3,0" in devs
        assert "plughw:CARD=Headphones,DEV=0" in devs
        assert "plughw:2,0" in devs
        # Must never include HDMI
        assert not any("hdmi" in d.lower() for d in devs)

        # Configured override wins
        monkeypatch.setattr("src.config.AUDIO_OUTPUT_DEVICE", "plughw:custom,0")
        assert tts._find_best_alsa_devices() == ["plughw:custom,0"]


def test_set_system_volume_max():
    from src.speech.tts import set_system_volume_max
    calls = []

    def mock_run(cmd, *args, **kwargs):
        calls.append(cmd)
        m = MagicMock()
        m.returncode = 0
        return m

    with patch("sys.platform", "linux"), \
         patch("shutil.which", return_value="/usr/bin/amixer"), \
         patch("subprocess.run", side_effect=mock_run):
        set_system_volume_max()
        assert len(calls) > 0
        # Check that 100% and unmute were called
        assert any("100%" in cmd and "unmute" in cmd for cmd in calls)
        assert any("-c" in cmd and "Headphones" in cmd for cmd in calls)


def test_audio_normalization_peak():
    import numpy as np
    from src.speech.tts import _normalize_audio

    # Test normalization boosts quiet audio close to 0.98 peak
    raw = np.array([0.0, 0.1, -0.2, 0.15], dtype=np.float32)
    normalized = _normalize_audio(raw)
    assert normalized is not None
    assert np.isclose(np.max(np.abs(normalized)), 0.98, atol=0.01)


def test_groq_tts_block_persists_and_skips_fast(tmp_path, monkeypatch):
    import src.speech.tts as tts_mod
    from src import config as cfg

    block_file = tmp_path / "groq_block"
    monkeypatch.setattr(tts_mod, "_groq_tts_block_path", lambda: str(block_file))
    monkeypatch.setattr(cfg, "USE_KOKORO_ONNX", False)

    # No block file yet -> engine attempts cloud (client build may fail
    # without key, but must not be pre-blocked).
    eng = TTSEngine()
    assert eng._groq_tts_blocked_until == 0.0

    # Persist a 1h block -> new engine skips Groq with no network wait.
    tts_mod._write_groq_tts_blocked_until(time.time() + 3600.0)
    assert block_file.exists()
    eng2 = TTSEngine()
    assert eng2._groq_tts_blocked_until > time.time()
    t0 = time.time()
    assert eng2._synthesize_groq("Hello friend") is None
    assert (time.time() - t0) < 1.0

    # Expired block is ignored again.
    tts_mod._write_groq_tts_blocked_until(time.time() - 10.0)
    eng3 = TTSEngine()
    assert eng3._groq_tts_blocked_until < time.time()


def test_flush_clause_early_long_sentence():
    # Long sentence, no terminator yet: flush at the last clause boundary
    # so synthesis overlaps continued generation.
    buf = ("Jazz is all about spontaneous improvisation, smooth syncopation, "
           "and late night sessions that never quite seem to end")
    chunks, tail = _flush_at_boundary(buf)
    assert len(chunks) == 1
    assert tail and len(chunks[0]) >= 30
    assert chunks[0].endswith(",")
    # Head + tail reconstruct the buffer (modulo whitespace).
    assert (chunks[0] + " " + tail).split() == buf.split()


def test_flush_short_buffer_with_comma_stays_whole():
    # Short buffers must NOT fragment: every synth call pays seconds of
    # fixed ONNX cost, so tiny fragments would be slower overall.
    buf = "Hi, how are things going"
    chunks, tail = _flush_at_boundary(buf)
    assert chunks == []
    assert tail == buf


def test_flush_clause_number_guard():    # Never split inside a thousands separator like "3,000".
    buf = ("I have 3,000 apples and oranges in the basket, plus a whole lot "
           "more fruit waiting on the kitchen counter for everyone")
    chunks, tail = _flush_at_boundary(buf)
    if chunks:
        assert not chunks[0].rstrip(",").endswith("3")
        assert "3,000" in (chunks[0] + " " + tail)

    # Number comma as the only separator: must stay whole.
    only = ("I have 3,000 apples and many more things stored in the big wooden "
            "box over there yes indeed")
    assert len(only) >= 90
    chunks2, tail2 = _flush_at_boundary(only)
    assert chunks2 == []
    assert tail2 == only


def test_piper_relpath():
    assert TTSEngine._piper_relpath("en_US-lessac-medium") == \
        "en/en_US/lessac/medium/en_US-lessac-medium"
    assert TTSEngine._piper_relpath("ar_JO-kareem-medium") == \
        "ar/ar_JO/kareem/medium/ar_JO-kareem-medium"


def test_resample_to_24k():
    import numpy as np
    from src import config as cfg

    sr_native = 22050
    t = np.arange(sr_native, dtype=np.float32) / sr_native
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = TTSEngine._resample_to_24k(tone, sr_native)
    assert abs(len(out) - cfg.TTS_SAMPLE_RATE) <= 2
    assert float(np.max(np.abs(out))) > 0.1

    same = np.ones(100, dtype=np.float32)
    assert TTSEngine._resample_to_24k(same, cfg.TTS_SAMPLE_RATE) is not same
    assert len(TTSEngine._resample_to_24k(same, cfg.TTS_SAMPLE_RATE)) == 100


def _make_quiet_engine(monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "USE_KOKORO_ONNX", False)
    return TTSEngine()


def test_piper_routing_and_cache(monkeypatch):
    import numpy as np
    from src import config as cfg
    monkeypatch.setattr(cfg, "TTS_ENGINE", "piper")

    tts = _make_quiet_engine(monkeypatch)
    fake = np.ones(2400, dtype=np.float32)
    with patch.object(tts, "_synthesize_groq", return_value=None) as mock_groq, \
         patch.object(tts, "_synthesize_piper", return_value=fake) as mock_piper, \
         patch.object(tts, "_synthesize_english", return_value=None) as mock_en, \
         patch.object(tts, "_synthesize_arabic", return_value=None) as mock_ar:
        out = tts._synthesize("Hello friend, how is the weather today?")
        assert out is not None and len(out) == len(fake)
        mock_piper.assert_called_once()
        mock_en.assert_not_called()
        mock_ar.assert_not_called()
        # Second call is a cache hit: no backend touched at all.
        mock_piper.reset_mock()
        out2 = tts._synthesize("Hello friend, how is the weather today?")
        assert out2 is not None
        mock_piper.assert_not_called()
        mock_groq.assert_called_once()  # only the first call tried cloud


def test_piper_fallback_to_legacy(monkeypatch):
    import numpy as np
    from src import config as cfg
    monkeypatch.setattr(cfg, "TTS_ENGINE", "piper")

    tts = _make_quiet_engine(monkeypatch)
    legacy = np.ones(1200, dtype=np.float32)
    with patch.object(tts, "_synthesize_groq", return_value=None), \
         patch.object(tts, "_synthesize_piper", return_value=None), \
         patch.object(tts, "_synthesize_english", return_value=legacy) as mock_en:
        out = tts._synthesize("Hello friend, how is the weather today?")
        assert out is not None and len(out) == len(legacy)
        mock_en.assert_called_once()


def test_kokoro_mode_skips_piper(monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "TTS_ENGINE", "kokoro")

    tts = _make_quiet_engine(monkeypatch)
    with patch.object(tts, "_synthesize_groq", return_value=None), \
         patch.object(tts, "_synthesize_piper", return_value=None) as mock_piper, \
         patch.object(tts, "_synthesize_english", return_value=None) as mock_en, \
         patch.object(tts, "_synthesize_arabic", return_value=None) as mock_ar:
        tts._synthesize("Hello friend, how is the weather today?")
        mock_en.assert_called_once()
        mock_ar.assert_not_called()
        mock_piper.assert_not_called()

        mock_en.reset_mock()
        tts._synthesize("أهلا يا صاحبي، عامل إيه؟")
        mock_ar.assert_called_once()
        mock_piper.assert_not_called()


def test_piper_init_skips_kokoro_session(monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "TTS_ENGINE", "piper")
    monkeypatch.setattr(cfg, "USE_KOKORO_ONNX", True)

    tts = TTSEngine()
    assert tts.onnx_kokoro is None
    assert tts.onnx_session is None


@pytest.mark.slow
def test_piper_live_synthesis():
    """End-to-end Piper synthesis (needs models/piper voices on disk)."""
    import numpy as np
    from src import config as cfg

    model = os.path.join(cfg.PIPER_MODEL_DIR,
                         TTSEngine._piper_relpath("en_US-lessac-medium") + ".onnx")
    if not os.path.exists(model):
        pytest.skip("Piper voice not downloaded")
    tts = TTSEngine()
    audio = tts._synthesize_piper("Hey! What's up?", speed=1.0)
    assert audio is not None and len(audio) > cfg.TTS_SAMPLE_RATE // 2
    assert float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) > 0.01
    assert tts._synthesize_piper("", speed=1.0) is None


