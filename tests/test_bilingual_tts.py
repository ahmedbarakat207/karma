#!/usr/bin/env python3
import pytest
from unittest.mock import MagicMock, patch

from src.speech.arabic_g2p import is_arabic, normalize_arabic_text, clean_phonemes, ArabicG2P
from src.speech.tts import clean_for_speech, TTSEngine
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
    with patch("kokoro.KPipeline") as mock_pipeline_cls:
        mock_pipe = MagicMock()
        mock_pipeline_cls.return_value = mock_pipe

        tts = TTSEngine()

        # Mock English and Arabic synthesis internal methods
        with patch.object(tts, "_synthesize_english", return_value=None) as mock_en, \
             patch.object(tts, "_synthesize_arabic", return_value=None) as mock_ar:

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
        assert len(devs) >= 2
        assert devs[0] == "plughw:CARD=UACDemo,DEV=0"
        assert devs[1] == "plughw:CARD=Headphones,DEV=0"
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


