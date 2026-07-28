import re
import time
import config
from state import internal_state

_DIRECT_PATTERNS = [
    re.compile(r"^\s*(hi|hey|hello|yo|sup)\b", re.IGNORECASE),
    re.compile(r"^\s*(what|how|why|who|where|when|is|am|are|can|do|does)\b", re.IGNORECASE),
    re.compile(r"\b(hold|holding|this|that|phone|cup|bottle)\b", re.IGNORECASE),
]

_FACE_LABELS = {
    "face looking directly at you",
    "looking at you",
    "looking to their left",
    "looking to their right",
}

_BASE_INTERACTION_PROMPT = """You are a warm, genuine, witty human friend having a real spoken conversation with your close friend in your shared space.

HUMAN CONVERSATION STYLE:
- Speak naturally like a real human friend talking out loud.
- Use natural contractions: "I'm", "that's", "it's", "don't", "can't", "you're", "let's", "gonna".
- Express real human emotions: excitement, empathy, humor, curiosity, playfulness, warmth.
- Keep responses short and conversational (1-2 sentences) so the dialogue flows naturally back and forth.
- Never sound like an AI assistant, customer service rep, or formal computer program.
- Never mention cameras, sensors, vision models, or AI mechanisms.
"""

def run_interaction_response(memory, engine, tts):
    new_speech = memory.unhandled_speech(0)
    if not new_speech:
        return False

    latest_ts = max(e["ts"] for e in new_speech)
    speech_text = " ".join(e["text"] for e in new_speech).strip()
    if not speech_text:
        memory.mark_handled(latest_ts)
        return False

    print(f"[interaction] got speech: '{speech_text}'")

    # Update internal state on speech
    all_events = memory.all_events()
    internal_state.update(all_events)

    # Dynamic Mood Injection
    mood_instruction = {
        "playful": "YOUR CURRENT FEELING: You're feeling playful, witty, and mischievous. Tease them a little or react with playful humor!",
        "curious": "YOUR CURRENT FEELING: You're intensely curious about what they are doing. Ask probing questions to learn more!",
        "tired": "YOUR CURRENT FEELING: You're feeling sleepy and cozy. Keep replies warm, laid-back, and short.",
        "attentive": "YOUR CURRENT FEELING: You are fully locked in. Give deep, empathetic, thoughtful, and engaged replies."
    }.get(internal_state.mood, "YOUR CURRENT FEELING: Be warm, human, and natural.")

    sys_prompt = f"{_BASE_INTERACTION_PROMPT}\n{mood_instruction}"

    # Build vision & emotion context
    visible = memory.recent_objects(config.VISION_CONTEXT_WINDOW_SECONDS)
    vision_ctx = ""
    if visible:
        labels = sorted(set(
            obj.replace("saw a ", "")
            for obj in visible
            if obj not in _FACE_LABELS
        ))
        if labels:
            vision_ctx = f"Current Environment & Expressions: {', '.join(labels)}\n"

    # Build recent thoughts context
    recent_thoughts = [
        e["text"] for e in all_events
        if e["kind"] == "thought" and (time.time() - e["ts"]) < 30
    ]
    thought_ctx = ""
    if recent_thoughts:
        thought_ctx = f"Your Recent Realization: {recent_thoughts[-1]}\n"

    # Build conversation context
    conv_ctx = memory.get_conversation_context(config.CONVERSATION_HISTORY_SIZE)

    # Build prompt
    prompt_parts = []
    if conv_ctx:
        prompt_parts.append(f"Previous conversation:\n{conv_ctx}")
    if thought_ctx:
        prompt_parts.append(thought_ctx)
    if vision_ctx:
        prompt_parts.append(vision_ctx)
    prompt_parts.append(f'Friend said: "{speech_text}"\n\nYour reply:')

    prompt = "\n".join(prompt_parts)

    try:
        reply = engine.chat(sys_prompt, prompt, max_tokens=70)
        if reply:
            # Clean up response prefixes & echoes
            reply = reply.strip().strip('"').strip("'")
            reply = re.sub(r"^(friend said:|your reply:|reply:|you:)\s*", "", reply, flags=re.IGNORECASE).strip()
            if reply.lower().startswith("friend said"):
                reply = re.sub(r"^friend said.*?:?\s*", "", reply, flags=re.IGNORECASE).strip()

            if reply and len(reply) > 1:
                print(f"[reply] ({internal_state.mood}) {reply}")
                memory.add(kind="reply", text=reply, counts_as_activity=True)
                memory.add_conversation(speech_text, reply)
                if tts:
                    tts.speak(reply)
    except Exception as e:
        print(f"[interaction] generation error: {e}")
    finally:
        memory.mark_handled(latest_ts)
        return True
