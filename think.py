"""
The cognition loop ("The Mind & Free Will").
Refactored into salience-driven immediate and quiet thought functions.
"""
import time
import random
import config
from state import internal_state
from prosody import prosody_stream
from interaction import _extract_plain_text

_THINK_PROMPT = (
    "You are a warm, genuine human friend sharing space with someone in the room. "
    "Observe what they are doing, holding, wearing, or what is being discussed right now.\n\n"
    "CRITICAL RULES:\n"
    "- Act like a real human friend in the room.\n"
    "- If your mood is playful/energetic, make a witty, fun remark about what's happening.\n"
    "- If your mood is curious, ask a natural question about what the user is doing or holding.\n"
    "- If you suddenly remember something from past context, mention it naturally ('This reminds me of...').\n"
    "- If tired or nothing noteworthy is happening, output [silence].\n"
    "- Keep it to 1 fresh, natural, creative sentence. Never mention camera, sensors, or AI.\n\n"
    "OUTPUT FORMAT: Respond with ONLY a JSON object in this exact structure:\n"
    '{\n'
    '  \"emotion\": \"<curious|playful|warm|excited|tired|sad|surprised|neutral>\",\n'
    '  \"inflection\": \"<question|excited|whisper|emphatic|flat>\",\n'
    '  \"text_chunks\": [\"<the single sentence>\"]\n'
    '}\n'
    "Or just output [silence] if there is nothing to say."
)


def recall_memory_context(store, embedder, triggers_text):
    if not store or not embedder or not triggers_text:
        return "none"
    try:
        emb = embedder.encode(triggers_text[:300]).tolist()
        recalled = store.query(emb, k=2)
        if recalled and recalled[0]["distance"] < 0.75:
            return recalled[0]["text"]
    except Exception:
        pass
    return "none"

def think_immediately(memory, engine, tts, store, embedder, urgency="HIGH", speaking_event=None):
    workspace = memory.get_workspace()
    urgent_triggers = memory.get_high_salience_events()
    
    past_ctx = recall_memory_context(store, embedder, urgent_triggers or "room")
    time_of_day = time.strftime('%I:%M %p')
    workspace.self_model['time_of_day'] = time_of_day
    
    prompt = f"""
    CURRENT REALITY:
    - Time: {workspace.self_model['time_of_day']}
    - Where I am: {workspace.self_model['location']}
    - What just changed: {urgent_triggers or 'nothing specific'}
    - What I remember about this place: {past_ctx}
    
    As a human friend in this room, compose a continuous 1-sentence "inner monologue" 
    about what it feels like to be here right now. Acknowledge your own presence.
    Example: "I'm watching them pick that up, feels like they're about to ask me something."
    """
    
    try:
        # Decide up-front whether to speak (room must be quiet, user not talking)
        should_speak = (
            tts is not None
            and not memory.is_user_speaking()
            and (speaking_event is None or not speaking_event.is_set())
            and random.random() < 0.8
        )

        if should_speak:
            # Streaming prosody-aware path
            collected = []
            def collecting_stream():
                for tok in engine.stream_chat(_THINK_PROMPT, prompt, max_tokens=80):
                    collected.append(tok)
                    yield tok
            prosody_stream(collecting_stream(), tts)
            raw = "".join(collected).strip()
        else:
            # Silent path: generate but don't speak
            raw = engine.chat(_THINK_PROMPT, prompt, max_tokens=80)
            if not raw:
                return

        from interaction import _extract_plain_text
        thought = _extract_plain_text(raw.strip())

        if not thought:
            return
        if thought.lower() == "[silence]" or ("silence" in thought.lower() and len(thought) < 12):
            return

        print(f"[thought] (URGENT) {thought}")
        memory.add(kind="self_awareness", text=thought)

    except Exception as e:
        print(f"[think] urgent generation error: {e}")


def think_quietly(memory, engine, store, embedder):
    workspace = memory.get_workspace()
    recent = memory.recent_text(config.RECENT_WINDOW_SECONDS)
    if not recent:
        return

    time_of_day = time.strftime('%I:%M %p')
    workspace.self_model['time_of_day'] = time_of_day

    prompt = f"""
    CURRENT REALITY:
    - Time: {workspace.self_model['time_of_day']}
    - Where I am: {workspace.self_model['location']}
    - Recent events: {recent}

    Generate 1 natural spoken thought, or output [silence] if quiet:
    """
    try:
        raw = engine.chat(_THINK_PROMPT, prompt, max_tokens=80)
        if not raw:
            return
        # _THINK_PROMPT now requests JSON — extract plain text before logging
        thought = _extract_plain_text(raw.strip())
        if not thought:
            return
        if thought.lower() == "[silence]" or ("silence" in thought.lower() and len(thought) < 12):
            return
        print(f"[thought] (IDLE) {thought}")
        memory.add(kind="self_awareness", text=thought, counts_as_activity=False)
    except Exception:
        pass

