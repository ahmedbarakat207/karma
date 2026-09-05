"""Differential drive for Karma's mobile base.

Hardware: 2x 5840-31ZY worm DC gear motors (12V, 60RPM, 65mm wheels)
driven by 2x BTS7960 43A H-bridges. Each bridge takes RPWM/LPWM (speed)
plus R_EN/L_EN (enable). Controlled via pigpio; fully mocked when no
daemon/hardware so Mac dev + pytest stay green.

Kinematics: v in [-1,1] forward, omega in [-1,1] turn (positive = left).
left = v - omega*k, right = v + omega*k, clamped to DRIVE_MAX_DUTY.

Safety: every motion has a watchdog auto-stop (DRIVE_MAX_SECONDS),
an estop latch, and duty clamping. Worm gears self-lock on stop.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from src import config


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # radians, dead-reckoned estimate (no encoders)


class _Bridge:
    """One BTS7960 channel pair. Mock when pigpio unavailable."""

    def __init__(self, r_pwm: int, l_pwm: int, r_en: int, l_en: int):
        self.r_pwm = r_pwm
        self.l_pwm = l_pwm
        self.r_en = r_en
        self.l_en = l_en
        self._pi = None
        self._mock = False
        self.last_duty: float = 0.0  # signed, for tests/telemetry
        self._connect()

    def _connect(self) -> None:
        try:
            import pigpio
            pi = pigpio.pi()
            if not pi.connected:
                self._mock = True
                config.log_debug("[drive] pigpiod not running, mock mode")
                return
            self._pi = pi
            freq = getattr(config, "DRIVE_PWM_FREQ", 20000)
            for pin in (self.r_pwm, self.l_pwm, self.r_en, self.l_en):
                try:
                    pi.set_mode(pin, pigpio.OUTPUT)
                except Exception:
                    pass
            try:
                pi.set_PWM_frequency(self.r_pwm, freq)
                pi.set_PWM_frequency(self.l_pwm, freq)
            except Exception:
                pass
            self._enable(True)
            self._raw(0.0)
        except Exception as e:
            self._mock = True
            config.log_debug(f"[drive] no motor hardware ({e}), mock mode")

    def _enable(self, on: bool) -> None:
        if self._pi and not self._mock:
            try:
                import pigpio
                self._pi.write(self.r_en, 1 if on else 0)
                self._pi.write(self.l_en, 1 if on else 0)
            except Exception as e:
                config.log_debug(f"[drive] enable error: {e}")

    def _raw(self, duty: float) -> None:
        """Signed duty in [-1,1]. RPWM=fwd, LPWM=rev."""
        self.last_duty = max(-1.0, min(1.0, duty))
        if self._pi and not self._mock:
            try:
                mag = int(abs(self.last_duty) * 255)
                if self.last_duty >= 0:
                    self._pi.set_PWM_dutycycle(self.r_pwm, mag)
                    self._pi.set_PWM_dutycycle(self.l_pwm, 0)
                else:
                    self._pi.set_PWM_dutycycle(self.r_pwm, 0)
                    self._pi.set_PWM_dutycycle(self.l_pwm, mag)
            except Exception as e:
                config.log_debug(f"[drive] pwm error: {e}")

    def stop(self) -> None:
        self._raw(0.0)

    def cleanup(self) -> None:
        try:
            self._raw(0.0)
            self._enable(False)
        except Exception:
            pass
        if self._pi and not self._mock:
            try:
                self._pi.stop()
            except Exception:
                pass
            self._pi = None


class DriveBase:
    """Thread-safe differential drive with watchdog + estop."""

    # 60RPM * pi*0.065m circumference ≈ 0.20 m/s at full duty.
    V_MAX_MS = 0.20
    # Track width estimate for yaw rate: omega = (vr-vl)/track.
    TRACK_M = 0.25

    def __init__(self):
        self._lock = threading.Lock()
        self.left = _Bridge(
            getattr(config, "DRIVE_LEFT_RPWM", 19),
            getattr(config, "DRIVE_LEFT_LPWM", 26),
            getattr(config, "DRIVE_LEFT_R_EN", 16),
            getattr(config, "DRIVE_LEFT_L_EN", 20),
        )
        self.right = _Bridge(
            getattr(config, "DRIVE_RIGHT_RPWM", 6),
            getattr(config, "DRIVE_RIGHT_LPWM", 5),
            getattr(config, "DRIVE_RIGHT_R_EN", 22),
            getattr(config, "DRIVE_RIGHT_L_EN", 27),
        )
        self.pose = Pose()
        self._estop = False
        self._moving = False
        self._cmd_v = 0.0
        self._cmd_omega = 0.0
        self._stop_timer: Optional[threading.Timer] = None
        self._last_cmd = ""
        self._last_cmd_time = 0.0
        self.mock = self.left._mock and self.right._mock

    # -- state ---------------------------------------------------------
    @property
    def is_mock(self) -> bool:
        return self.mock

    @property
    def is_estopped(self) -> bool:
        with self._lock:
            return self._estop

    @property
    def is_moving(self) -> bool:
        with self._lock:
            return self._moving

    def last_command(self) -> Tuple[str, float]:
        with self._lock:
            return self._last_cmd, self._last_cmd_time

    def describe(self) -> str:
        with self._lock:
            mode = "mock" if self.mock else "hw"
            if self._estop:
                s = "estopped"
            elif self._moving:
                s = f"moving {self._last_cmd}"
            else:
                s = "idle"
            return (
                f"{s} ({mode}), "
                f"pose≈({self.pose.x:.1f},{self.pose.y:.1f},{self.pose.theta:.2f})"
            )

    # -- core ----------------------------------------------------------
    def set_velocity(self, v: float, omega: float, label: str = "",
                     duration: Optional[float] = None) -> bool:
        """Command normalized (v, omega). Returns False if refused (estop/disabled)."""
        if not getattr(config, "DRIVE_ENABLED", True):
            return False
        max_duty = abs(getattr(config, "DRIVE_MAX_DUTY", 0.6))
        max_secs = abs(getattr(config, "DRIVE_MAX_SECONDS", 3.0))
        if duration is None:
            duration = max_secs
        duration = max(0.1, min(max_secs, duration))

        with self._lock:
            if self._estop:
                return False
            k = 0.5
            left = max(-max_duty, min(max_duty, v - omega * k))
            right = max(-max_duty, min(max_duty, v + omega * k))
            self.left._raw(left)
            self.right._raw(right)
            self._integrate_locked(v, omega, 0.0)  # mark intent; pose integrates on stop
            self._cmd_v, self._cmd_omega = v, omega
            self._moving = True
            self._last_cmd = label or f"v={v:.2f} w={omega:.2f}"
            self._last_cmd_time = time.time()
            self._arm_watchdog_locked(duration)
            return True

    def _integrate_locked(self, v: float, omega: float, dt: float) -> None:
        # Called with lock; dt=0 marks intent only.
        if dt <= 0:
            self._pending_v, self._pending_w = v, omega
            self._pending_t0 = time.time()
            return

    def _arm_watchdog_locked(self, duration: float) -> None:
        try:
            if self._stop_timer:
                self._stop_timer.cancel()
        except Exception:
            pass
        t0 = self._last_cmd_time  # generation stamp; stale timers exit early
        v, w = self._cmd_v, self._cmd_omega
        timer = threading.Timer(duration, self._watchdog_stop, args=(t0, v, w))
        timer.daemon = True
        self._stop_timer = timer
        timer.start()

    def _watchdog_stop(self, t0: float, v: float, w: float) -> None:
        with self._lock:
            if t0 != self._last_cmd_time:
                return  # superseded by a newer command; its own timer will stop it
            dt = time.time() - t0
            self._dead_reckon_locked(v, w, min(dt, getattr(config, "DRIVE_MAX_SECONDS", 3.0)))
            self.left._raw(0.0)
            self.right._raw(0.0)
            self._moving = False

    def _dead_reckon_locked(self, v: float, w: float, dt: float) -> None:
        vl = v * self.V_MAX_MS
        vr = v * self.V_MAX_MS
        # Differential yaw from omega command (scaled to rad/s).
        yaw_rate = w * (2 * self.V_MAX_MS / self.TRACK_M) * 0.5
        fwd = (vl + vr) / 2.0
        self.pose.theta += yaw_rate * dt
        self.pose.x += fwd * dt * __import__("math").cos(self.pose.theta)
        self.pose.y += fwd * dt * __import__("math").sin(self.pose.theta)

    # -- convenience ---------------------------------------------------
    def forward(self, speed: Optional[float] = None, duration: float = 1.0) -> bool:
        s = abs(speed if speed is not None else getattr(config, "DRIVE_CRUISE_DUTY", 0.35))
        return self.set_velocity(s, 0.0, label=f"forward {duration:.1f}s", duration=duration)

    def backward(self, speed: Optional[float] = None, duration: float = 1.0) -> bool:
        s = abs(speed if speed is not None else getattr(config, "DRIVE_CRUISE_DUTY", 0.35))
        return self.set_velocity(-s, 0.0, label=f"backward {duration:.1f}s", duration=duration)

    def turn_left(self, speed: Optional[float] = None, duration: float = 0.8) -> bool:
        s = abs(speed if speed is not None else getattr(config, "DRIVE_TURN_DUTY", 0.4))
        return self.set_velocity(0.0, s, label=f"turn_left {duration:.1f}s", duration=duration)

    def turn_right(self, speed: Optional[float] = None, duration: float = 0.8) -> bool:
        s = abs(speed if speed is not None else getattr(config, "DRIVE_TURN_DUTY", 0.4))
        return self.set_velocity(0.0, -s, label=f"turn_right {duration:.1f}s", duration=duration)

    def stop(self) -> None:
        with self._lock:
            try:
                if self._stop_timer:
                    self._stop_timer.cancel()
            except Exception:
                pass
            self._stop_timer = None
            dt = time.time() - self._last_cmd_time if self._moving else 0.0
            if self._moving and 0 < dt < getattr(config, "DRIVE_MAX_SECONDS", 3.0) + 0.5:
                self._dead_reckon_locked(
                    getattr(self, "_cmd_v", 0.0), getattr(self, "_cmd_omega", 0.0), dt
                )
            self.left._raw(0.0)
            self.right._raw(0.0)
            self._moving = False

    def estop(self, reason: str = "") -> None:
        with self._lock:
            self._estop = True
        self.stop()
        config.log_debug(f"[drive] ESTOP {reason}")

    def clear_estop(self) -> None:
        with self._lock:
            self._estop = False

    def reset_pose(self) -> None:
        with self._lock:
            self.pose = Pose()

    def cleanup(self) -> None:
        self.stop()
        self.left.cleanup()
        self.right.cleanup()


drive_base = DriveBase()
