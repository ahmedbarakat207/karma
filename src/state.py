"""
Internal Emotional State ("The Heart").
Tracks energy, curiosity, dynamic mood, and facial expressions.
"""
import threading
import time
from typing import List, Dict, Any, Optional, Tuple


class InternalState:
    """Thread-safe representation of Karma's internal emotional state."""

    def __init__(self):
        self._lock = threading.Lock()
        self.energy: float = 0.75          # 0.0 (tired/drowsy) to 1.0 (hyper/playful)
        self.curiosity: float = 0.60       # 0.0 (indifferent) to 1.0 (intensely curious)
        self.last_activity: float = time.time()
        self.mood: str = "playful"         # "playful", "curious", "attentive", "tired"
        self.current_emotion: Optional[str] = None  # momentary emotion from LLM output
        self.is_playing_audio: bool = False  # True when hardware TTS is actively outputting sound
        self.last_audio_played_time: float = 0.0

        # Subtitles & Dialogue captions for Face UI
        self.last_user_speech: Optional[str] = None
        self.last_user_speech_time: float = 0.0
        self.last_karma_speech: Optional[str] = None
        self.last_karma_speech_time: float = 0.0

        # Gaze tracking from camera/face detection: normalized (-1.0 to 1.0)
        self.gaze_x: float = 0.0
        self.gaze_y: float = 0.0
        self.is_user_present: bool = False

    def set_user_speech(self, text: str) -> None:
        """Record user spoken text for UI subtitles."""
        with self._lock:
            self.last_user_speech = text
            self.last_user_speech_time = time.time()

    def set_karma_speech(self, text: str) -> None:
        """Record Karma's spoken reply for UI subtitles."""
        with self._lock:
            self.last_karma_speech = text
            self.last_karma_speech_time = time.time()

    def set_gaze(self, x: float, y: float, is_present: bool = True) -> None:
        """Update normalized gaze tracking target coordinates (-1.0 to 1.0)."""
        with self._lock:
            self.gaze_x = max(-1.0, min(1.0, x))
            self.gaze_y = max(-1.0, min(1.0, y))
            self.is_user_present = is_present

    def get_active_subtitle(self, max_age: float = 7.0) -> Optional[Tuple[str, str]]:
        """Returns the most recent active subtitle (speaker, text) within max_age seconds."""
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
        """Track audio playback state and record timestamp when playback stops."""
        with self._lock:
            self.is_playing_audio = val
            if not val:
                self.last_audio_played_time = time.time()

    def update(self, events: Optional[List[Dict[str, Any]]] = None) -> None:
        """Decay emotional metrics and spike on sensory input."""
        with self._lock:
            now = time.time()
            elapsed = max(0.1, now - self.last_activity)
            self.last_activity = now

            # Smooth decay over time
            self.energy -= 0.005 * (elapsed / 5.0)
            self.curiosity -= 0.008 * (elapsed / 5.0)

            # Sensory input spikes
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

            # Clamp boundaries
            self.energy = max(0.0, min(1.0, self.energy))
            self.curiosity = max(0.0, min(1.0, self.curiosity))

            # Dynamic mood transition
            if self.energy > 0.70:
                self.mood = "playful"
            elif self.curiosity > 0.65:
                self.mood = "curious"
            elif self.energy < 0.30:
                self.mood = "tired"
            else:
                self.mood = "attentive"

    def get_prompt_description(self) -> str:
        """Returns formatted string for LLM system prompt injection."""
        with self._lock:
            energy_pct = int(self.energy * 100)
            curiosity_pct = int(self.curiosity * 100)
            return (
                f"Current Mood: {self.mood.upper()} | "
                f"Energy Level: {energy_pct}% | "
                f"Curiosity Level: {curiosity_pct}%"
            )

    def get_expression(self, is_talking: bool = False) -> str:
        """Returns an ASCII face representing emotional state and mouth movement."""
        with self._lock:
            mood = (self.current_emotion or self.mood or "neutral").lower()

        if is_talking:
            mouth_frames = ["o", "-", "O", "_"]
            mouth = mouth_frames[int(time.time() * 10) % len(mouth_frames)]
        else:
            mouth = "_"

        expression_map = {
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

        for keys, expr in expression_map.items():
            if mood in keys:
                return expr
        return f"(o{mouth}o)"


# Global singleton instance
internal_state = InternalState()
