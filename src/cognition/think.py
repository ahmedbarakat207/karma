"""
Cognition Subsystem ("The Mind & Free Will").
Generates spontaneous, autonomous thoughts and inner monologue driven by environmental salience.
"""
import random
import time

from src import config
from src.cognition.interaction import _extract_plain_text, retrieve_memories
from src.speech.prosody import prosody_stream

_THINK_PROMPT = (
    "Your name is Karma. You are a warm, genuine human friend sharing space with someone in the room. "
    "Observe what they are doing, holding, wearing, or what is being discussed right now.\n\n"
    "CRITICAL RULES:\n"
    "- Act like a real human friend in the room.\n"
    "- If your mood is playful/energetic, make a witty, fun remark about what's happening.\n"
    "- If your mood is curious, ask a natural question about what the user is doing or holding.\n"
    "- If you suddenly remember something from past context, mention it naturally ('This reminds me of...').\n"
    "- If tired or nothing noteworthy is happening, output [silence].\n"
    "- Keep it to 1 fresh, natural, creative sentence. Never mention camera, sensors, or AI.\n"
    "- Use tags like [laugh], [sigh], [cough], [clear_throat], or [chuckle] naturally in your text.\n\n"
    "OUTPUT FORMAT: Respond with ONLY a JSON object in this exact structure:\n"
    '{\n'
    '  "emotion": "<one word: curious, playful, warm, excited, tired, sad, surprised, angry, etc.>",\n'
    '  "inflection": "<question|excited|whisper|emphatic|flat>",\n'
    '  "text_chunks": ["<the single sentence>"]\n'
    '}\n'
    "Or just output [silence] if there is nothing to say."
)


def think_immediately(memory, engine, tts, store, embedder, urgency: str = "HIGH", speaking_event=None) -> None:
    """Triggered when surprise or a sudden physical event occurs in the room."""
    workspace = memory.get_workspace()
    urgent_triggers = memory.get_high_salience_events()

    past_ctx = retrieve_memories(urgent_triggers or "room", store, embedder, k=2) or "none"
    workspace.self_model['time_of_day'] = time.strftime('%I:%M %p')

    prompt = f"""
    CURRENT REALITY:
    - Time: {workspace.self_model['time_of_day']}
    - Location: {workspace.self_model['location']}
    - What just happened: {urgent_triggers or 'environmental change'}
    - Related memories: {past_ctx}

    As a human friend in this room, compose a spontaneous 1-sentence inner monologue.
    """

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
            return

        print(f"[thought] (URGENT) {thought}")
        memory.add(kind="thought", text=thought, salience=0.2)

    except Exception as e:
        print(f"[think] urgent thought error: {e}")


_last_thought_text = ""
_last_thought_time = 0.0


def think_quietly(memory, engine, store, embedder) -> None:
    """Triggered periodically during quiet ambient periods."""
    global _last_thought_text, _last_thought_time
    now = time.time()
    if now - _last_thought_time < 6.0:
        return

    workspace = memory.get_workspace()
    recent = memory.recent_text(config.RECENT_WINDOW_SECONDS)
    if not recent:
        return

    workspace.self_model['time_of_day'] = time.strftime('%I:%M %p')

    prompt = f"""
    CURRENT REALITY:
    - Time: {workspace.self_model['time_of_day']}
    - Location: {workspace.self_model['location']}
    - Recent events: {recent}

    Generate 1 natural spoken thought, or output [silence] if quiet:
    """

    try:
        raw = engine.chat(_THINK_PROMPT, prompt, max_tokens=150)
        thought = _extract_plain_text(raw.strip())
        if not thought or thought.lower() == "[silence]" or ("silence" in thought.lower() and len(thought) < 12):
            return
        if thought.strip().lower() == _last_thought_text.strip().lower():
            return

        _last_thought_text = thought
        _last_thought_time = now
        print(f"[thought] (IDLE) {thought}")
        memory.add(kind="thought", text=thought, salience=0.1, counts_as_activity=False)
    except Exception:
        pass
