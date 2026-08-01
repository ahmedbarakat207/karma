import re
import time
import config
from state import internal_state
from prosody import prosody_stream

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

OUTPUT FORMAT — CRITICAL:
You MUST respond with ONLY a JSON object in exactly this structure. No other text before or after.
The key ORDER matters: emotion and inflection MUST come before text_chunks so the speech engine
can prime its vocal style before the first word is synthesised.

{
  "emotion": "<one word: curious | playful | warm | excited | tired | sad | surprised | inquisitive | neutral>",
  "inflection": "<one word: question | excited | whisper | emphatic | flat>",
  "text_chunks": [
    "<first clause ending at a natural pause or sentence boundary>",
    "<second clause>",
    "..."
  ]
}

Split text_chunks at sentence boundaries (periods, question marks, exclamation marks).
Do NOT split inside a clause — each chunk should be a complete grammatical thought.
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
    prompt_parts.append(f'Friend said: "{speech_text}"\n\nYour reply (JSON only):')

    prompt = "\n".join(prompt_parts)

    try:
        reply_text = ""  # always bound; prevents UnboundLocalError masking real exceptions
        if tts and hasattr(engine, "stream_chat"):
            # --- Streaming path: prosody-aware, sentence-level parallel synthesis ---
            token_stream = engine.stream_chat(sys_prompt, prompt, max_tokens=120)
            # Collect the full reply for memory logging while prosody_stream plays it
            collected_tokens = []

            def collecting_stream():
                for tok in token_stream:
                    collected_tokens.append(tok)
                    yield tok

            prosody_stream(collecting_stream(), tts)
            raw_reply = "".join(collected_tokens).strip()

            # Extract plain text from the JSON for memory logging
            reply_text = _extract_plain_text(raw_reply)
        else:
            # --- Legacy blocking path (no TTS or no streaming support) ---
            raw_reply = engine.chat(sys_prompt, prompt, max_tokens=120)
            raw_reply = raw_reply.strip().strip('"').strip("'")
            raw_reply = re.sub(r"^(friend said:|your reply:|reply:|you:)\s*", "", raw_reply, flags=re.IGNORECASE).strip()
            reply_text = raw_reply
            if tts and reply_text and len(reply_text) > 1:
                tts.speak(reply_text)

        if reply_text and len(reply_text) > 1:
            print(f"[reply] ({internal_state.mood}) {reply_text}")
            memory.add(kind="reply", text=reply_text, counts_as_activity=True)
            memory.add_conversation(speech_text, reply_text)

    except Exception as e:
        print(f"[interaction] generation error: {e}")
    finally:
        memory.mark_handled(latest_ts)
    return True


def _extract_plain_text(raw: str) -> str:
    """
    Extract a plain-text summary of the reply from the JSON envelope for memory logging.
    Falls back gracefully if the JSON is malformed or absent.
    """
    import json
    # Try parsing as JSON first
    try:
        # Find the JSON object in the raw string
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(raw[start:end])
            chunks = obj.get("text_chunks", [])
            if chunks:
                return " ".join(chunks).strip()
    except Exception:
        pass

    # Fallback: extract text_chunks array content with regex
    chunks_match = re.search(r'"text_chunks"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if chunks_match:
        strings = re.findall(r'"((?:[^"\\]|\\.)*)"', chunks_match.group(1))
        if strings:
            return " ".join(strings).strip()

    # Last resort: strip all JSON syntax and return plain text
    plain = re.sub(r'[{}\[\]"]', "", raw)
    plain = re.sub(r'"?(emotion|inflection|text_chunks)"\s*:\s*"?[^,\n]*"?,?\n?', "", plain)
    return plain.strip()
