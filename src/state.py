import threading
import time
from typing import List, Dict, Any, Optional, Tuple


class InternalState:
    def __init__(self):
        self._lock = threading.Lock()
        self.energy: float = 0.75
        self.curiosity: float = 0.60
        self.last_activity: float = time.time()
        self.mood: str = "playful"
        self.current_emotion: Optional[str] = None
        self.is_playing_audio: bool = False
        self.last_audio_played_time: float = 0.0

        self.last_user_speech: Optional[str] = None
        self.last_user_speech_time: float = 0.0
        self.last_karma_speech: Optional[str] = None
        self.last_karma_speech_time: float = 0.0

        self.active_code_snippet: Optional[str] = None
        self.active_code_lang: Optional[str] = None
        self.active_code_time: float = 0.0

        self.gaze_x: float = 0.0
        self.gaze_y: float = 0.0
        self.is_user_present: bool = False

    def set_active_code(self, code: Optional[str], lang: Optional[str] = None) -> None:
        with self._lock:
            self.active_code_snippet = code.strip() if code else None
            self.active_code_lang = (lang or "code").lower().strip()
            self.active_code_time = time.time() if code else 0.0

    def clear_active_code(self) -> None:
        with self._lock:
            self.active_code_snippet = None
            self.active_code_lang = None
            self.active_code_time = 0.0

    def get_active_code(self, max_age: float = 60.0) -> Optional[Tuple[str, str]]:
        with self._lock:
            if not self.active_code_snippet:
                return None
            if time.time() - self.active_code_time > max_age:
                self.active_code_snippet = None
                return None
            return (self.active_code_snippet, self.active_code_lang or "code")

    def set_user_speech(self, text: str) -> None:
        with self._lock:
            self.last_user_speech = text
            self.last_user_speech_time = time.time()

    def set_karma_speech(self, text: str) -> None:
        with self._lock:
            self.last_karma_speech = text
            self.last_karma_speech_time = time.time()

    def set_gaze(self, x: float, y: float, is_present: bool = True) -> None:
        with self._lock:
            self.gaze_x = max(-1.0, min(1.0, x))
            self.gaze_y = max(-1.0, min(1.0, y))
            self.is_user_present = is_present

    def get_active_subtitle(self, max_age: float = 7.0) -> Optional[Tuple[str, str]]:
        with self._lock:
            now = time.time()
            u_age = now - self.last_user_speech_time
            k_age = now - self.last_karma_speech_time

            if k_age <= max_age and self.last_karma_speech and k_age <= u_age:
                return ("Karma", self.last_karma_speech)
            elif u_age <= max_age and self.last_user_speech:
                return ("You", self.last_user_speech)
            return None

    def set_playing_audio(self, val: bool) -> None:
        with self._lock:
            self.is_playing_audio = val
            if not val:
                self.last_audio_played_time = time.time()

    def set_camera_frame(self, jpeg: Optional[bytes]) -> None:
        with self._lock:
            self.last_camera_jpeg = jpeg
            self.last_camera_jpeg_time = time.time() if jpeg else 0.0

    def get_camera_frame(self, max_age: float = 5.0) -> Optional[bytes]:
        with self._lock:
            if not getattr(self, "last_camera_jpeg", None):
                return None
            if time.time() - getattr(self, "last_camera_jpeg_time", 0.0) > max_age:
                return None
            return self.last_camera_jpeg

    def update(self, events: Optional[List[Dict[str, Any]]] = None) -> None:
        with self._lock:
            now = time.time()
            elapsed = max(0.1, now - self.last_activity)
            self.last_activity = now

            self.energy -= 0.005 * (elapsed / 5.0)
            self.curiosity -= 0.008 * (elapsed / 5.0)

            if events:
                for e in events[-8:]:
                    text = e.get("text", "").lower()
                    kind = e.get("kind", "")

                    if kind == "conscious_trigger":
                        self.energy = min(1.0, self.energy + 0.25)
                        self.curiosity = min(1.0, self.curiosity + 0.30)
                    elif kind == "object" and ("face" in text or "looking at you" in text):
                        self.energy = min(1.0, self.energy + 0.15)
                        self.curiosity = min(1.0, self.curiosity + 0.10)
                    elif kind == "speech":
                        self.energy = min(1.0, self.energy + 0.12)
                        self.curiosity = min(1.0, self.curiosity + 0.15)

            self.energy = max(0.0, min(1.0, self.energy))
            self.curiosity = max(0.0, min(1.0, self.curiosity))

            if self.energy > 0.70:
                self.mood = "playful"
            elif self.curiosity > 0.65:
                self.mood = "curious"
            elif self.energy < 0.30:
                self.mood = "tired"
            else:
                self.mood = "attentive"

    def get_prompt_description(self) -> str:
        with self._lock:
            return (
                f"Current Mood: {self.mood.upper()} | "
                f"Energy Level: {int(self.energy * 100)}% | "
                f"Curiosity Level: {int(self.curiosity * 100)}%"
            )

    def get_expression(self, is_talking: bool = False) -> str:
        with self._lock:
            mood = (self.current_emotion or self.mood or "neutral").lower()

        if is_talking:
            mouth = ["o", "-", "O", "_"][int(time.time() * 10) % 4]
        else:
            mouth = "_"

        faces = {
            ("playful", "happy", "warm", "joyful", "delighted", "amused"): f"(^{mouth}^)",
            ("excited", "amazed", "thrilled", "ecstatic"): f"(*{mouth}*)",
            ("curious", "inquisitive", "confused", "wondering", "puzzled"): f"(O{mouth}o)",
            ("surprised", "shocked", "woah", "astonished"): f"(O{mouth}O)",
            ("tired", "sleepy", "exhausted", "bored", "drowsy"): f"(-{mouth}-)",
            ("sad", "reflective", "depressed", "melancholy", "heartbroken"): f"(u{mouth}u)",
            ("attentive", "focused", "serious", "listening"): f"(ò{mouth}ó)",
            ("angry", "frustrated", "annoyed", "mad", "upset"): f"(>{mouth}<)",
            ("crying", "emotional", "tears"): f"(T{mouth}T)",
            ("cool", "smug", "confident", "chill"): f"(B{mouth}B)",
            ("silly", "goofy", "wacky", "crazy", "derp"): f"(9{mouth}9)",
            ("love", "affectionate", "caring", "sweet"): f"(♥{mouth}♥)",
            ("scared", "fearful", "nervous", "anxious", "worried"): f"(~{mouth}~)",
            ("dizzy", "overwhelmed", "confounded"): f"(@{mouth}@)",
            ("dead", "offline", "broken", "crashed"): f"(x{mouth}x)",
        }

        for keys, expr in faces.items():
            if mood in keys:
                return expr
        return f"(o{mouth}o)"


internal_state = InternalState()
