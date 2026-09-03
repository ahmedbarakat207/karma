import os
import pytest
import numpy as np

from src.hardware.neck import NeckActuator, ANGLE_FACE_MODE, ANGLE_KIOSK_MODE
from src.ui.kiosk import KioskManager
from src.cognition.interaction import _check_kiosk_intent


def test_neck_actuator_angles_and_tilt():
    actuator = NeckActuator(pin=18)
    assert actuator.current_angle == ANGLE_FACE_MODE

    actuator.tilt_to_kiosk()
    assert actuator.target_angle == ANGLE_KIOSK_MODE

    actuator.tilt_to_face()
    assert actuator.target_angle == ANGLE_FACE_MODE

    pulse_90 = actuator._angle_to_pulse(90.0)
    pulse_135 = actuator._angle_to_pulse(135.0)
    assert 1400 <= pulse_90 <= 1600
    assert 1900 <= pulse_135 <= 2100


def test_kiosk_views_and_rendering():
    kiosk = KioskManager()
    assert kiosk.active_view == "face"
    assert not kiosk.is_active()

    for v in ["map", "docs", "apps", "achievements"]:
        kiosk.open_view(v)
        assert kiosk.active_view == v
        assert kiosk.is_active()

        frame = kiosk.render_kiosk(800, 480)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 800, 3)

    kiosk.close()
    assert kiosk.active_view == "face"
    assert not kiosk.is_active()


def test_kiosk_touch_interaction():
    kiosk = KioskManager()
    kiosk.close()

    handled = kiosk.handle_touch(750, 25, 800, 480)
    assert handled is True
    assert kiosk.is_active()
    assert kiosk.active_view == "map"

    handled = kiosk.handle_touch(200, 25, 800, 480)
    assert handled is True
    assert kiosk.active_view == "docs"

    handled = kiosk.handle_touch(450, 25, 800, 480)
    assert handled is True
    assert kiosk.active_view == "apps"

    handled = kiosk.handle_touch(600, 25, 800, 480)
    assert handled is True
    assert kiosk.active_view == "achievements"

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
