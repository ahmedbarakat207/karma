"""
The cognition loop ("The Mind & Free Will").
Refactored into salience-driven immediate and quiet thought functions.
"""
import time
import random
import config
from state import internal_state

_THINK_PROMPT = (
    "You are a warm, genuine human friend sharing space with someone in the room. "
    "Observe what they are doing, holding, wearing, or what is being discussed right now.\n\n"
    "CRITICAL RULES:\n"
    "- Act like a real human friend in the room.\n"
    "- If your mood is playful/energetic, make a witty, fun remark about what's happening.\n"
    "- If your mood is curious, ask a natural question about what the user is doing or holding.\n"
    "- If you suddenly remember something from past context, mention it naturally ('This reminds me of...').\n"
    "- If tired or nothing noteworthy is happening, output [silence].\n"
    "- Keep it to 1 fresh, natural, creative sentence. Never mention camera, sensors, or AI."
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
        thought = engine.chat(_THINK_PROMPT, prompt, max_tokens=80)
        if thought:
            thought = thought.strip().strip('"').strip("'")
            if thought.lower() == "[silence]" or "silence" in thought.lower() and len(thought) < 12:
                return
            
            print(f"[thought] (URGENT) {thought}")
            memory.add(kind="self_awareness", text=thought)
            
            # Speak it ONLY if the room is quiet and the user isn't talking
            if not memory.is_user_speaking() and (speaking_event is None or not speaking_event.is_set()):
                # Higher chance to speak if it's urgent
                if random.random() < 0.8:
                    if tts:
                        tts.speak(thought)
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
        thought = engine.chat(_THINK_PROMPT, prompt, max_tokens=80)
        if thought:
            thought = thought.strip().strip('"').strip("'")
            if thought.lower() == "[silence]" or "silence" in thought.lower() and len(thought) < 12:
                return
            print(f"[thought] (IDLE) {thought}")
            memory.add(kind="self_awareness", text=thought, counts_as_activity=False)
    except Exception as e:
        pass
