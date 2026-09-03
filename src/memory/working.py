
import threading
import time
from typing import List, Dict, Any, Optional, Set, Tuple


class ConsciousnessState:
    def __init__(self):
        self.current_focus: Optional[str] = None
        self.spatial_map: Dict[str, List[str]] = {"left": [], "center": [], "right": []}
        self.self_model: Dict[str, str] = {
            "location": "desk",
            "orientation": "facing_user",
            "time_of_day": "afternoon"
        }
        self.prediction_error: float = 0.0

    def update(self, vision_objects: List[Tuple[str, Tuple[float, float]]], speech_text: Optional[str] = None):
        self.spatial_map = self._bind_objects_to_space(vision_objects)
        if speech_text:
            self.current_focus = speech_text

    def _bind_objects_to_space(self, vision_objects: List[Tuple[str, Tuple[float, float]]]) -> Dict[str, List[str]]:
        spatial = {"left": [], "center": [], "right": []}
        if vision_objects:
            for label, (cx, cy) in vision_objects:
                if cx < 200:
                    spatial["left"].append(label)
                elif cx > 440:
                    spatial["right"].append(label)
                else:
                    spatial["center"].append(label)
        return spatial


class WorkingMemory:

    def __init__(self):
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        self._recent_keys: Dict[Tuple[str, str], float] = {}
        self._handled_ts: float = 0.0
        self._conversation: List[Dict[str, str]] = []
        self.last_activity_ts: float = time.time()
        self.consciousness = ConsciousnessState()
        self._user_is_speaking: bool = False
        self._latest_face_frame = None
        self._recognized_people: Set[str] = set()

    def add(self, kind: str, text: str, dedup_seconds: float = 0.0,
            counts_as_activity: bool = True, salience: float = 0.0) -> None:
        now = time.time()
        key = (kind, text)

        if dedup_seconds > 0:
            last = self._recent_keys.get(key)
            if last and (now - last) < dedup_seconds:
                return
        self._recent_keys[key] = now

        with self._lock:
            self._events.append({
                "ts": now,
                "kind": kind,
                "text": text,
                "salience": salience
            })
            if counts_as_activity:
                self.last_activity_ts = now
            if kind == "conscious_trigger" and salience > self.consciousness.prediction_error:
                self.consciousness.prediction_error = salience

    def recent_text(self, window_seconds: float) -> Optional[str]:
        cutoff = time.time() - window_seconds
        with self._lock:
            events = [e for e in self._events if e["ts"] >= cutoff]
        if not events:
            return None
        lines = []
        for e in events:
            t = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            lines.append(f"[{t}] ({e['kind']}) {e['text']}")
        return "\n".join(lines)

    def all_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._events) == 0

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._recent_keys.clear()
            self._handled_ts = time.time()

    def mark_handled(self, ts: float) -> None:
        with self._lock:
            self._handled_ts = max(self._handled_ts, ts)

    def unhandled_speech(self, since_ts: float = 0.0) -> List[Dict[str, Any]]:
        with self._lock:
            threshold = max(since_ts, self._handled_ts)
            return [
                e for e in self._events
                if e["kind"] == "speech" and e["ts"] > threshold
            ]

    def recent_objects(self, window_seconds: float) -> List[str]:
        cutoff = time.time() - window_seconds
        with self._lock:
            return [
                e["text"] for e in self._events
                if e["kind"] == "object" and e["ts"] >= cutoff
            ]

    def add_conversation(self, speech: str, reply: str) -> None:
        with self._lock:
            self._conversation.append({"speech": speech, "reply": reply})
            if len(self._conversation) > 10:
                self._conversation = self._conversation[-10:]

    def get_conversation_context(self, n: int = 5) -> str:
        with self._lock:
            recent = self._conversation[-n:]
        if not recent:
            return ""
        lines = []
        for ex in recent:
            lines.append(f"User: {ex['speech']}")
            lines.append(f"You: {ex['reply']}")
        return "\n".join(lines)

    def get_high_salience_events(self) -> Optional[str]:
        cutoff = time.time() - 10
        with self._lock:
            urgent = [
                e["text"] for e in self._events
                if e["kind"] == "conscious_trigger" and e["ts"] >= cutoff and e.get("salience", 0.0) >= 0.7
            ]
            self.consciousness.prediction_error = 0.0
        return ", ".join(urgent) if urgent else None

    def get_workspace(self) -> ConsciousnessState:
        return self.consciousness

    def set_user_speaking(self, is_speaking: bool) -> None:
        with self._lock:
            self._user_is_speaking = is_speaking

    def is_user_speaking(self) -> bool:
        with self._lock:
            return self._user_is_speaking

    def set_face_frame(self, frame) -> None:
        with self._lock:
            self._latest_face_frame = frame

    def get_face_frame(self):
        with self._lock:
            return self._latest_face_frame

    def set_recognized_people(self, names: Set[str]) -> None:
        with self._lock:
            self._recognized_people = set(names)

    def get_recognized_people(self) -> Set[str]:
        with self._lock:
            return set(self._recognized_people)
