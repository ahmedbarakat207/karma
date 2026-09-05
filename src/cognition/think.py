import random
import time

from src import config
from src.cognition.interaction import _extract_plain_text, retrieve_memories
from src.speech.prosody import prosody_stream
from src.ui import events as _events

_THINK_PROMPT = """You are Karma, observing the room. Produce a brief casual remark, or [silence] if nothing notable is happening.

Respond with JSON:
{
  "emotion": "curious|playful|warm|excited|tired|neutral",
  "inflection": "flat|question|excited|whisper",
  "text_chunks": ["your thought"]
}
Or [silence]."""


def think_immediately(memory, engine, tts, store, embedder, urgency: str = "HIGH", speaking_event=None) -> None:
    workspace = memory.get_workspace()
    urgent_triggers = memory.get_high_salience_events()

    past_ctx = retrieve_memories(urgent_triggers or "room", store, embedder, k=2) or "none"
    workspace.self_model['time_of_day'] = time.strftime('%I:%M %p')

    prompt = (
        f"Context:\n"
        f"- Time: {workspace.self_model['time_of_day']}\n"
        f"- Location: {workspace.self_model['location']}\n"
        f"- Event: {urgent_triggers or 'environmental change'}\n"
        f"- Memories: {past_ctx}\n\n"
        f"Spontaneous thought:"
    )

    try:
        should_speak = (
            tts is not None
            and getattr(config, "SPEAK_THOUGHTS", False)
            and not memory.is_user_speaking()
            and (speaking_event is None or not speaking_event.is_set())
            and random.random() < 0.8
        )

        if should_speak and hasattr(engine, "stream_chat"):
            collected = []
            def collecting_stream():
                for tok in engine.stream_chat(_THINK_PROMPT, prompt, max_tokens=150):
                    collected.append(tok)
                    yield tok
            prosody_stream(collecting_stream(), tts)
            raw = "".join(collected).strip()
        else:
            raw = engine.chat(_THINK_PROMPT, prompt, max_tokens=150)

        thought = _extract_plain_text(raw.strip())
        if not thought or thought.lower() == "[silence]" or ("silence" in thought.lower() and len(thought) < 12):
            _events.post("thought", "[silence]", {"urgency": urgency})
            return

        config.log_debug(f"[thought] (URGENT) {thought}")
        _events.post("thought", thought, {"urgency": urgency})
        memory.add(kind="thought", text=thought, salience=0.2)

    except Exception as e:
        config.log_debug(f"[think] urgent thought error: {e}")


_last_thought_text = ""
_last_thought_time = 0.0


def think_quietly(memory, engine, store, embedder) -> None:
    global _last_thought_text, _last_thought_time
    now = time.time()
    if now - _last_thought_time < 6.0:
        return

    workspace = memory.get_workspace()
    recent = memory.recent_text(config.RECENT_WINDOW_SECONDS)
    if not recent:
        return

    workspace.self_model['time_of_day'] = time.strftime('%I:%M %p')

    prompt = (
        f"Context:\n"
        f"- Time: {workspace.self_model['time_of_day']}\n"
        f"- Recent: {recent}\n\n"
        f"Brief thought or [silence]:"
    )

    try:
        raw = engine.chat(_THINK_PROMPT, prompt, max_tokens=150)
        thought = _extract_plain_text(raw.strip())
        if not thought or thought.lower() == "[silence]" or ("silence" in thought.lower() and len(thought) < 12):
            _events.post("thought", "[silence]", {"urgency": "idle"})
            return
        if thought.strip().lower() == _last_thought_text.strip().lower():
            return

        _last_thought_text = thought
        _last_thought_time = now
        config.log_debug(f"[thought] (IDLE) {thought}")
        _events.post("thought", thought, {"urgency": "idle"})
        memory.add(kind="thought", text=thought, salience=0.1, counts_as_activity=False)
    except Exception:
        pass
