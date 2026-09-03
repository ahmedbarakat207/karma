import os
import threading
import time
import math
from typing import Optional

from src import config

DEFAULT_SERVO_PIN = int(os.environ.get("NECK_SERVO_PIN", "18"))

ANGLE_FACE = 90.0
ANGLE_KIOSK = 135.0
ANGLE_FACE_MODE = ANGLE_FACE
ANGLE_KIOSK_MODE = ANGLE_KIOSK
MIN_ANGLE = 45.0
MAX_ANGLE = 140.0


class NeckActuator:
    def __init__(self, pin: int = DEFAULT_SERVO_PIN):
        self.pin = pin
        self.current_angle = ANGLE_FACE
        self.target_angle = ANGLE_FACE
        self._lock = threading.Lock()
        self._ramp_thread: Optional[threading.Thread] = None
        self._pi = None
        self._mock = False

        self._connect()

    def _connect(self) -> None:
        try:
            import pigpio
            self._pi = pigpio.pi()
            if not self._pi.connected:
                self._pi = None
                self._mock = True
                config.log_debug("[neck] pigpiod not running, mock mode")
            else:
                config.log_debug(f"[neck] connected on GPIO {self.pin}")
                self._write(self.current_angle)
        except Exception as e:
            self._mock = True
            config.log_debug(f"[neck] no servo hardware ({e})")

    def _angle_to_pulse(self, angle: float) -> int:
        clamped = max(MIN_ANGLE, min(MAX_ANGLE, angle))
        return int(500 + (clamped / 180.0) * 2000)

    def _write(self, angle: float) -> None:
        self.current_angle = angle
        if self._pi and not self._mock:
            try:
                self._pi.set_servo_pulsewidth(self.pin, self._angle_to_pulse(angle))
            except Exception as e:
                config.log_debug(f"[neck] write error: {e}")

    def set_angle(self, target: float, speed: float = 60.0) -> None:
        target = max(MIN_ANGLE, min(MAX_ANGLE, target))
        self.target_angle = target

        with self._lock:
            self._ramp_thread = None

            def _ramp():
                start = self.current_angle
                diff = target - start
                if abs(diff) < 0.5:
                    self._write(target)
                    return

                duration = abs(diff) / max(10.0, speed)
                steps = max(10, int(duration * 30))
                dt = duration / steps

                for i in range(1, steps + 1):
                    t = i / steps
                    smooth = (1.0 - math.cos(t * math.pi)) / 2.0
                    self._write(start + diff * smooth)
                    time.sleep(dt)

                self._write(target)

            self._ramp_thread = threading.Thread(target=_ramp, daemon=True, name="neck_ramp")
            self._ramp_thread.start()

    def tilt_to_kiosk(self) -> None:
        config.log_debug("[neck] -> 135° kiosk")
        self.set_angle(ANGLE_KIOSK)

    def tilt_to_face(self) -> None:
        config.log_debug("[neck] -> 90° face")
        self.set_angle(ANGLE_FACE)

    def is_kiosk_angle(self) -> bool:
        return abs(self.current_angle - ANGLE_KIOSK) < 10.0

    def cleanup(self) -> None:
        if self._pi and not self._mock:
            try:
                self._pi.set_servo_pulsewidth(self.pin, 0)
                self._pi.stop()
            except Exception:
                pass


neck_actuator = NeckActuator()
