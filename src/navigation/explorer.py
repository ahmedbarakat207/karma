"""Autonomous movement brain: explore, follow, go-to-beacon.

No lidar/encoders on this base (camera + mic only), so navigation is
semantic, not metric: wander with vision e-stop, steer toward faces when
following, and execute timed beacon runs (desk/kitchen/door/charger).
Every move is logged to WorkingMemory as kind="movement" so sleep
consolidation turns it into long-term episodic memory, and the prompt
grounding in interaction.py narrates it in Karma's persona voice.

Pure helpers (is_blocked, choose_explore_action) are unit-tested.
"""

import random
import threading
import time
from typing import Dict, List, Optional, Tuple

from src import config

# (label, conf, (x1, y1, x2, y2))
BBox = Tuple[str, float, Tuple[int, int, int, int]]

# Semantic beacons. Without SLAM these are goto routines (turn + timed
# drive), not coordinates. Matches data/maps floors + RAG movement section.
BEACONS = ("desk", "kitchen", "door", "charger", "window")

OBSTACLE_LABELS = {
    "person", "chair", "couch", "sofa", "table", "dining table", "desk",
    "door", "refrigerator", "bed", "bench", "suitcase", "backpack",
}


def is_blocked(bboxes: List[BBox], frame_w: int = 640, frame_h: int = 480,
               area_ratio: Optional[float] = None,
               center_margin: Optional[float] = None) -> Optional[str]:
    """Return blocking label if a large box sits in the center corridor.

    Worm base has no cliff/force sensors, so this vision e-stop is the
    primary safety net. Returns None when the path looks clear.
    """
    area_ratio = area_ratio if area_ratio is not None else getattr(
        config, "OBSTACLE_AREA_RATIO", 0.12)
    center_margin = center_margin if center_margin is not None else getattr(
        config, "OBSTACLE_CENTER_MARGIN", 0.25)
    frame_area = max(1, frame_w * frame_h)
    cx_mid = frame_w / 2.0
    for label, _conf, (x1, y1, x2, y2) in bboxes or []:
        if label not in OBSTACLE_LABELS:
            continue
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area / frame_area < area_ratio:
            continue
        cx = (x1 + x2) / 2.0
        covers_center = (x1 <= cx_mid <= x2)
        near_center = abs(cx - cx_mid) / frame_w <= center_margin
        if covers_center or near_center:
            return label
    return None


def choose_explore_action(spatial_map: Dict[str, List[str]], blocked: bool,
                          curiosity: float, energy: float,
                          rng: Optional[random.Random] = None) -> Tuple[float, float, str]:
    """Pure wander policy. Returns (v, omega, reason).

    - blocked -> turn in place (alternate by curiosity so we don't loop).
    - person/face on a side + curious -> steer toward them (social explore).
    - tired/low energy -> stay put (save battery, matches mood model).
    - else -> slow forward with slight random weave.
    """
    rng = rng or random
    if blocked:
        side = "left" if curiosity > 0.5 else "right"
        return 0.0, 0.6 if side == "left" else -0.6, f"blocked_turn_{side}"
    if energy < 0.25:
        return 0.0, 0.0, "rest_low_energy"
    left = spatial_map.get("left", [])
    right = spatial_map.get("right", [])
    center = spatial_map.get("center", [])

    def has_person(side_list):
        return any(("person" in s) or ("looking at you" in s) or (s and s[0].isupper()) for s in side_list)

    if curiosity > 0.55:
        if has_person(left) and not has_person(right):
            return 0.25, 0.45, "approach_left_person"
        if has_person(right) and not has_person(left):
            return 0.25, -0.45, "approach_right_person"
        if any("person" in s or "Face" in s for s in center):
            return 0.0, 0.0, "stay_with_person"
    if center and curiosity < 0.3:
        # Cautious: something ahead, sidestep.
        return 0.15, 0.4 if len(left) <= len(right) else -0.4, "cautious_weave"
    weave = rng.uniform(-0.15, 0.15)
    return 0.35, weave, "cruise"


class Explorer:
    """Mode state machine + tick loop. Singleton `explorer` below."""

    IDLE = "idle"
    EXPLORE = "explore"
    FOLLOW = "follow"
    GOTO = "goto"

    def __init__(self):
        self._lock = threading.Lock()
        self.mode = self.IDLE
        self.target = ""
        self._last_tick = 0.0
        self._last_blocked_note = 0.0
        self._turn_parity = False
        self.last_action = "none"

    # -- commands (called from voice intents / dashboard) ----------------
    def start_explore(self) -> str:
        with self._lock:
            self.mode = self.EXPLORE
            self.target = ""
        return "explore"

    def stop(self) -> str:
        with self._lock:
            self.mode = self.IDLE
            self.target = ""
        try:
            from src.hardware.drive import drive_base
            drive_base.stop()
        except Exception:
            pass
        return "stop"

    def start_follow(self, name: str = "") -> str:
        with self._lock:
            self.mode = self.FOLLOW
            self.target = name
        return "follow"

    def goto(self, beacon: str) -> str:
        beacon = (beacon or "").lower()
        if beacon not in BEACONS:
            return ""
        with self._lock:
            self.mode = self.GOTO
            self.target = beacon
        return beacon

    def note_blocked(self, label: str) -> None:
        with self._lock:
            self._last_blocked_note = time.time()
            self.last_action = f"blocked_by_{label}"

    def describe(self) -> str:
        with self._lock:
            return f"mode={self.mode}" + (f" target={self.target}" if self.target else "") \
                + f" last={self.last_action}"

    # -- loop ------------------------------------------------------------
    def tick(self, memory, drive, internal_state) -> Optional[str]:
        """One autonomy step. Returns a memory log line or None. Pure-ish."""
        now = time.time()
        interval = getattr(config, "EXPLORER_TICK_SECONDS", 2.0)
        if now - self._last_tick < interval:
            return None
        self._last_tick = now

        with self._lock:
            mode, target = self.mode, self.target
        if mode == self.IDLE:
            return None
        if getattr(drive, "is_estopped", False):
            return None

        spatial = {}
        try:
            spatial = dict(memory.consciousness.spatial_map or {})
        except Exception:
            pass
        curiosity = getattr(internal_state, "curiosity", 0.6)
        energy = getattr(internal_state, "energy", 0.75)

        if mode == self.FOLLOW:
            return self._tick_follow(memory, drive, internal_state)
        if mode == self.GOTO:
            return self._tick_goto(memory, drive, target)
        # EXPLORE
        blocked = bool(time.time() - self._last_blocked_note < 4.0)
        center = spatial.get("center", [])
        if not blocked and any("person" in s for s in center):
            blocked = False  # social stay handled by policy, not e-stop
        v, w, reason = choose_explore_action(spatial, blocked, curiosity, energy)
        if "blocked_turn" in reason:
            self._turn_parity = not self._turn_parity
            if self._turn_parity:
                w = -w
                reason += "_alt"
        if v == 0 and w == 0:
            drive.stop()
            self.last_action = reason
            return None
        cruise = getattr(config, "DRIVE_CRUISE_DUTY", 0.35)
        drive.set_velocity(v * cruise / 0.35, w, label=f"explore:{reason}",
                           duration=min(2.5, interval + 0.5))
        self.last_action = reason
        seen = ""
        try:
            objs = memory.recent_objects(8) or []
            uniq = sorted(set(objs))[:4]
            seen = f", saw: {', '.join(uniq)}" if uniq else ""
        except Exception:
            pass
        return f"Exploring ({reason}){seen}"

    def _tick_follow(self, memory, drive, internal_state) -> Optional[str]:
        gaze_x = getattr(internal_state, "gaze_x", 0.0)
        present = getattr(internal_state, "is_user_present", False)
        if not present:
            drive.stop()
            self.last_action = "follow_wait"
            return None
        # gaze_x in [-1,1]; steer proportionally, advance slowly.
        w = max(-0.6, min(0.6, -gaze_x * 0.8))
        drive.set_velocity(0.3, w, label="follow:approach", duration=2.0)
        self.last_action = "follow_approach"
        return "Following person"

    def _tick_goto(self, memory, drive, target: str) -> Optional[str]:
        # Timed semantic run: forward 2s, then done. Obstacle e-stop in
        # vision loop can interrupt; arrival announced via memory event.
        drive.set_velocity(0.35, 0.0, label=f"goto:{target}", duration=2.0)
        with self._lock:
            self.mode = self.IDLE
            self.last_action = f"goto_{target}"
            self.target = ""
        try:
            memory.consciousness.self_model["location"] = target
        except Exception:
            pass
        return f"Going to {target}"


explorer = Explorer()


def run_explorer(memory, stop_event, drive=None, internal_state=None) -> None:
    """Background thread: ticks explorer, logs movement to memory."""
    if drive is None:
        try:
            from src.hardware.drive import drive_base as _d
            drive = _d
        except Exception:
            return
    if internal_state is None:
        try:
            from src.state import internal_state as _s
            internal_state = _s
        except Exception:
            return
    if not getattr(config, "EXPLORER_ENABLED", True):
        return
    while not stop_event.is_set():
        try:
            line = explorer.tick(memory, drive, internal_state)
            if line:
                try:
                    memory.add(kind="movement", text=line,
                               counts_as_activity=True, salience=0.3)
                except Exception:
                    pass
        except Exception as e:
            config.log_debug(f"[explorer] tick error: {e}")
        stop_event.wait(getattr(config, "EXPLORER_TICK_SECONDS", 2.0))
