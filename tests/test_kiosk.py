"""
Unit and Integration Tests for Kiosk Touchscreen Menu & MG90S Neck Actuator.
"""
import os
import pytest
import numpy as np

from src.hardware.neck import NeckActuator, ANGLE_FACE_MODE, ANGLE_KIOSK_MODE
from src.ui.kiosk import KioskManager
from src.cognition.interaction import _check_kiosk_intent


def test_neck_actuator_angles_and_tilt():
    actuator = NeckActuator(pin=18)
    assert actuator.current_angle == ANGLE_FACE_MODE

    # Test kiosk tilt (135 deg)
    actuator.tilt_to_kiosk()
    assert actuator.target_angle == ANGLE_KIOSK_MODE

    # Test face mode tilt (90 deg)
    actuator.tilt_to_face()
    assert actuator.target_angle == ANGLE_FACE_MODE

    # Test pulse conversion
    pulse_90 = actuator._angle_to_pulse(90.0)
    pulse_135 = actuator._angle_to_pulse(135.0)
    assert 1400 <= pulse_90 <= 1600
    assert 1900 <= pulse_135 <= 2100


def test_kiosk_views_and_rendering():
    kiosk = KioskManager()
    assert kiosk.active_view == "face"
    assert not kiosk.is_active()

    # Test switching views
    for v in ["map", "docs", "apps", "achievements"]:
        kiosk.open_view(v)
        assert kiosk.active_view == v
        assert kiosk.is_active()

        frame = kiosk.render_kiosk(800, 480)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 800, 3)

    # Test close
    kiosk.close()
    assert kiosk.active_view == "face"
    assert not kiosk.is_active()


def test_kiosk_touch_interaction():
    kiosk = KioskManager()
    kiosk.close()

    # 1. Touch top-right MENU pill button from face mode
    handled = kiosk.handle_touch(750, 25, 800, 480)
    assert handled is True
    assert kiosk.is_active()
    assert kiosk.active_view == "map"

    # 2. Touch DOCS tab in top bar
    # Tabs: [DOCS] width - 640..width - 525 (160..275)
    handled = kiosk.handle_touch(200, 25, 800, 480)
    assert handled is True
    assert kiosk.active_view == "docs"

    # 3. Touch APPS tab in top bar
    # [APPS] width - 385..width - 250 (415..550)
    handled = kiosk.handle_touch(450, 25, 800, 480)
    assert handled is True
    assert kiosk.active_view == "apps"

    # 4. Touch AWARDS tab in top bar
    # [AWARDS] width - 240..width - 105 (560..695)
    handled = kiosk.handle_touch(600, 25, 800, 480)
    assert handled is True
    assert kiosk.active_view == "achievements"

    # 5. Touch EXIT button in top right
    # [EXIT] width - 95..width - 15 (705..785)
    handled = kiosk.handle_touch(750, 25, 800, 480)
    assert handled is True
    assert kiosk.active_view == "face"
    assert not kiosk.is_active()


def test_kiosk_voice_intents():
    assert _check_kiosk_intent("please open the map") == ("map", None)
    assert _check_kiosk_intent("show floor 2") == ("map", 1)
    assert _check_kiosk_intent("show floor 1") == ("map", 0)
    assert _check_kiosk_intent("open achievements") == ("achievements", None)
    assert _check_kiosk_intent("show student projects") == ("apps", None)
    assert _check_kiosk_intent("open documents") == ("docs", None)
    assert _check_kiosk_intent("close menu please") == ("face", None)
    assert _check_kiosk_intent("tell me a funny joke") is None
