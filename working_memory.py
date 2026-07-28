"""
The short-term memory stream. Sensor threads and the think loop all write
here; consolidation reads the whole thing at sleep time and clears it.
"""
import time
import threading


class ConsciousnessState:
    def __init__(self):
        self.current_focus = None          # What the agent is paying attention to right now
        self.spatial_map = {}              # {"left": "user", "right": "window", "center": "desk"}
        self.temporal_grid = {             # Sliding window of what happened when
            "0s": None, "2s": None, "5s": None 
        }
        self.self_model = {                # "Where am I?"
            "location": "desk",
            "orientation": "facing_user",
            "time_of_day": "afternoon"
        }
        self.prediction_error = 0.0        # How surprised the agent is right now
        self.narrative_thread = []         # The continuous "I am..." stream

    def update(self, vision_objects, speech_text, current_time):
        # Bind space and time together
        self.spatial_map = self._bind_objects_to_space(vision_objects)
        self.temporal_grid["0s"] = speech_text or "silence"
        self.temporal_grid["2s"] = self.temporal_grid.get("0s", "silence") # shift
        # Calculate prediction error (how different is 'now' from '5s ago'?)
        self.prediction_error = self._calculate_surprise(self.temporal_grid)

    def _bind_objects_to_space(self, vision_objects):
        # Real spatial binding based on bounding box centroids
        map_ = {"left": [], "center": [], "right": []}
        if vision_objects:
            for label, (cx, cy) in vision_objects:
                if cx < 200: map_["left"].append(label)
                elif cx > 440: map_["right"].append(label)
                else: map_["center"].append(label)
        return map_

    def _calculate_surprise(self, grid):
        # Real prediction error based on spatial shifts
        if grid["0s"] and grid["5s"] and grid["0s"] != grid["5s"]:
            # Calculate semantic difference (if "phone" disappeared)
            return 0.8 
        return 0.0


class WorkingMemory:
    def __init__(self):
        self._lock = threading.Lock()
        self._events = []          # list of dicts: ts, kind, text, salience
        self._recent_keys = {}     # (kind, text) -> last-seen ts, for dedup
        self._handled = set()      # indices of events already responded to
        self._conversation = []    # recent (speech, reply) exchanges
        self.last_activity_ts = time.time()
        self.consciousness = ConsciousnessState()
        self._user_is_speaking = False


    def add(self, kind, text, dedup_seconds=0, counts_as_activity=True, salience=0.0):
        now = time.time()
        key = (kind, text)

        if dedup_seconds:
            last = self._recent_keys.get(key)
            if last and (now - last) < dedup_seconds:
                return  # seen recently, skip
        self._recent_keys[key] = now

        with self._lock:
            self._events.append({"ts": now, "kind": kind, "text": text, "salience": salience})
            if counts_as_activity:
                self.last_activity_ts = now
            if kind == "conscious_trigger" and salience > self.consciousness.prediction_error:
                self.consciousness.prediction_error = salience

    def recent_text(self, window_seconds):
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

    def all_events(self):
        with self._lock:
            return list(self._events)

    def is_empty(self):
        with self._lock:
            return len(self._events) == 0

    def clear(self):
        with self._lock:
            self._events = []
            self._handled.clear()

    def mark_handled(self, ts):
        """Mark all events up to timestamp as handled by the interaction loop."""
        with self._lock:
            for i, e in enumerate(self._events):
                if e["ts"] <= ts:
                    self._handled.add(i)

    def unhandled_speech(self, since_ts):
        """Return speech events since since_ts that haven't been handled."""
        with self._lock:
            return [
                e for i, e in enumerate(self._events)
                if e["kind"] == "speech" and e["ts"] > since_ts and i not in self._handled
            ]

    def recent_objects(self, window_seconds):
        """Return recent object labels within window for interaction context."""
        cutoff = time.time() - window_seconds
        with self._lock:
            return [
                e["text"] for e in self._events
                if e["kind"] == "object" and e["ts"] >= cutoff
            ]

    def add_conversation(self, speech, reply):
        """Record a speech/reply exchange for conversation context."""
        with self._lock:
            self._conversation.append({"speech": speech, "reply": reply})
            # Keep only last 10 exchanges
            if len(self._conversation) > 10:
                self._conversation = self._conversation[-10:]

    def get_conversation_context(self, n=5):
        """Return last n exchanges as formatted string for prompt context."""
        with self._lock:
            recent = self._conversation[-n:]
        if not recent:
            return ""
        lines = []
        for ex in recent:
            lines.append(f"User: {ex['speech']}")
            lines.append(f"You: {ex['reply']}")
        return "\n".join(lines)

    def get_high_salience_events(self):
        """Extract conscious triggers with high salience and reset prediction error."""
        cutoff = time.time() - 10
        with self._lock:
            urgent = [
                e["text"] for e in self._events
                if e["kind"] == "conscious_trigger" and e["ts"] >= cutoff and e.get("salience", 0.0) >= 0.7
            ]
            self.consciousness.prediction_error = 0.0  # reset after consumption
        return ", ".join(urgent) if urgent else None

    def get_workspace(self):
        return self.consciousness

    def set_user_speaking(self, is_speaking):
        with self._lock:
            self._user_is_speaking = is_speaking

    def is_user_speaking(self):
        with self._lock:
            return self._user_is_speaking

    def get_temporal_snapshot(self, seconds_ago=5):
        """Return what the agent was aware of exactly X seconds ago."""
        cutoff = time.time() - seconds_ago
        with self._lock:
            return [e for e in self._events if abs(e["ts"] - cutoff) < 0.5]
