"""
Karma Neck Actuator Controller.
Controls the TowerPro MG90S Metal-Gear Micro Servo on GPIO 18 (PWM).
Smoothly pitches the head between 90° (conversational eye-contact)
and 135° (touchscreen kiosk interaction).
"""
import os
import threading
import time
from typing import Optional

from src import config

# Servo limits & pin specifications
DEFAULT_SERVO_PIN = int(os.environ.get("NECK_SERVO_PIN", "18"))
ANGLE_FACE_MODE = 90.0      # Forward-facing eye level
ANGLE_KIOSK_MODE = 135.0    # Tilted up for standing touchscreen interaction
MIN_SAFE_ANGLE = 45.0
MAX_SAFE_ANGLE = 140.0


class NeckActuator:
    """
    Thread-safe controller for the MG90S neck pitch servo.
    Supports smooth interpolation to prevent jerky mechanical movement.
    """

    def __init__(self, pin: int = DEFAULT_SERVO_PIN):
        self.pin = pin
        self.current_angle = ANGLE_FACE_MODE
        self.target_angle = ANGLE_FACE_MODE
        self._lock = threading.Lock()
        self._ramp_thread: Optional[threading.Thread] = None
        self._pi = None
        self._is_mock = False

        self._init_hardware()

    def _init_hardware(self) -> None:
        """Attempts connection to pigpio daemon, falling back to mock mode on non-Pi."""
        try:
            import pigpio
            self._pi = pigpio.pi()
            if not self._pi.connected:
                self._pi = None
                self._is_mock = True
                config.log_debug("[neck] pigpiod not running -- running in mock mode.")
            else:
                self._is_mock = False
                config.log_debug(f"[neck] connected to pigpiod on GPIO {self.pin}.")
                self._apply_angle(self.current_angle)
        except Exception as e:
            self._is_mock = True
            config.log_debug(f"[neck] hardware servo unavailable ({e}) -- running in mock mode.")

    def _angle_to_pulse(self, angle: float) -> int:
        """Converts angle (0-180 deg) to servo pulse width in microseconds (500-2500 us)."""
        clamped = max(MIN_SAFE_ANGLE, min(MAX_SAFE_ANGLE, angle))
        return int(500 + (clamped / 180.0) * 2000)

    def _apply_angle(self, angle: float) -> None:
        """Writes pulse width to hardware pin if available."""
        self.current_angle = angle
        if self._pi and not self._is_mock:
            pulse = self._angle_to_pulse(angle)
            try:
                self._pi.set_servo_pulsewidth(self.pin, pulse)
            except Exception as e:
                config.log_debug(f"[neck] error setting servo pulse: {e}")

    def set_pitch_angle(self, target: float, speed_deg_per_sec: float = 60.0) -> None:
        """Smoothly interpolates servo angle to the target angle."""
        target = max(MIN_SAFE_ANGLE, min(MAX_SAFE_ANGLE, target))
        self.target_angle = target

        with self._lock:
            # Stop existing ramp thread if running
            if self._ramp_thread and self._ramp_thread.is_alive():
                self._ramp_thread = None

            def _ramp():
                start = self.current_angle
                diff = target - start
                if abs(diff) < 0.5:
                    self._apply_angle(target)
                    return

                duration = abs(diff) / max(10.0, speed_deg_per_sec)
                steps = max(10, int(duration * 30))  # 30 Hz updates
                step_time = duration / steps

                for i in range(1, steps + 1):
                    # Cosine ease-in-out smoothing
                    progress = i / steps
                    smoothed = (1.0 - math.cos(progress * math.pi)) / 2.0
                    angle = start + diff * smoothed
                    self._apply_angle(angle)
                    time.sleep(step_time)

                self._apply_angle(target)

            import math
            self._ramp_thread = threading.Thread(target=_ramp, daemon=True, name="neck_servo_ramp")
            self._ramp_thread.start()

    def tilt_to_kiosk(self) -> None:
        """Tilts head to 135° for touch interaction."""
        config.log_debug("[neck] Tilting head to 135° for Kiosk Touch Mode.")
        self.set_pitch_angle(ANGLE_KIOSK_MODE)

    def tilt_to_face(self) -> None:
        """Tilts head to 90° for forward conversational eye contact."""
        config.log_debug("[neck] Tilting head to 90° for Conversational Face Mode.")
        self.set_pitch_angle(ANGLE_FACE_MODE)

    def is_kiosk_angle(self) -> bool:
        """Checks if head is currently tilted into kiosk orientation."""
        return abs(self.current_angle - ANGLE_KIOSK_MODE) < 10.0

    def cleanup(self) -> None:
        """Releases servo hardware and disconnects pigpio."""
        if self._pi and not self._is_mock:
            try:
                self._pi.set_servo_pulsewidth(self.pin, 0)
                self._pi.stop()
            except Exception:
                pass


# Global singleton instance
neck_actuator = NeckActuator()
