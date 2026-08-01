"""
Internal Emotional State ("The Heart").
Tracks energy, curiosity, and mood, which decay over time and spike based on sensory events.
"""
import time
import random


class InternalState:
    def __init__(self):
        self.energy = 0.75      # 0.0 (tired/drowsy) to 1.0 (hyper/playful)
        self.curiosity = 0.6    # 0.0 (indifferent) to 1.0 (intensely curious)
        self.last_activity = time.time()
        self.mood = "playful"   # playful, curious, attentive, reflective, tired
        self.current_emotion = None  # momentary emotion from LLM output
        self.is_playing_audio = False # accurate flag for when TTS is actually outputting sound

    def update(self, events):
        now = time.time()
        elapsed = max(0.1, now - self.last_activity)
        self.last_activity = now

        # Decay over time (gets gradually tired/bored if no activity)
        self.energy -= 0.005 * (elapsed / 5.0)
        self.curiosity -= 0.008 * (elapsed / 5.0)

        # Spike on sensory input from recent events
        if events:
            for e in events[-8:]:
                text = e.get("text", "").lower()
                kind = e.get("kind", "")

                if kind == "urgent_observation":
                    self.energy = min(1.0, self.energy + 0.25)
                    self.curiosity = min(1.0, self.curiosity + 0.30)
                elif kind == "object" and ("face" in text or "looking at you" in text):
                    self.energy = min(1.0, self.energy + 0.15)
                    self.curiosity = min(1.0, self.curiosity + 0.10)
                elif kind == "speech":
                    self.energy = min(1.0, self.energy + 0.12)
                    self.curiosity = min(1.0, self.curiosity + 0.15)

        # Clamp values
        self.energy = max(0.0, min(1.0, self.energy))
        self.curiosity = max(0.0, min(1.0, self.curiosity))

        # Determine dynamic mood
        if self.energy > 0.70:
            self.mood = "playful"
        elif self.curiosity > 0.65:
            self.mood = "curious"
        elif self.energy < 0.30:
            self.mood = "tired"
        else:
            self.mood = "attentive"

    def get_prompt_description(self):
        energy_pct = int(self.energy * 100)
        curiosity_pct = int(self.curiosity * 100)
        return (
            f"Current Mood: {self.mood.upper()} | "
            f"Energy Level: {energy_pct}% | "
            f"Curiosity Level: {curiosity_pct}%"
        )

    def get_expression(self, is_talking=False):
        """Returns an ASCII face based on current emotion/mood and talking state."""
        mood = self.current_emotion if self.current_emotion else self.mood
        if not mood:
            mood = "neutral"
        mood = mood.lower()

        if is_talking:
            # Flap the mouth dynamically based on time (approx 10 frames per sec)
            mouth_frames = ["o", "-", "O", "_"]
            mouth = mouth_frames[int(time.time() * 10) % len(mouth_frames)]
        else:
            mouth = "_"

        if mood in ("playful", "happy", "warm", "joyful", "delighted", "amused"):
            return f"(^{mouth}^)"
        elif mood in ("excited", "amazed", "thrilled", "ecstatic"):
            return f"(*{mouth}*)"
        elif mood in ("curious", "inquisitive", "confused", "wondering", "puzzled"):
            return f"(O{mouth}o)"
        elif mood in ("surprised", "shocked", "woah", "astonished"):
            return f"(O{mouth}O)"
        elif mood in ("tired", "sleepy", "exhausted", "bored", "drowsy"):
            return f"(-{mouth}-)"
        elif mood in ("sad", "reflective", "depressed", "melancholy", "heartbroken"):
            return f"(u{mouth}u)"
        elif mood in ("attentive", "focused", "serious", "listening"):
            return f"(ò{mouth}ó)"
        elif mood in ("angry", "frustrated", "annoyed", "mad", "upset"):
            return f"(>{mouth}<)"
        elif mood in ("crying", "emotional", "tears"):
            return f"(T{mouth}T)"
        elif mood in ("cool", "smug", "confident", "chill"):
            return f"(B{mouth}B)"
        elif mood in ("silly", "goofy", "wacky", "crazy", "derp"):
            return f"(9{mouth}9)"
        elif mood in ("love", "affectionate", "caring", "sweet"):
            return f"(♥{mouth}♥)"
        elif mood in ("scared", "fearful", "nervous", "anxious", "worried"):
            return f"(~{mouth}~)"
        elif mood in ("dizzy", "overwhelmed", "confounded"):
            return f"(@{mouth}@)"
        elif mood in ("dead", "offline", "broken", "crashed"):
            return f"(x{mouth}x)"
        else:
            return f"(o{mouth}o)"

# Global internal state instance

internal_state = InternalState()
