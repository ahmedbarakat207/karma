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
