"""
Conversation & Interaction Subsystem ("The Voice of Friendship").
Handles conversational turns with the user, performs real-time name learning & face registration,
assembles vision and memory context, and streams natural prosody-aware spoken replies.
"""
import json
import re
import time
from typing import Optional, List

import cv2
import numpy as np

from src import config
from src.speech.prosody import prosody_stream
from src.state import internal_state

from src.memory.face_registry import FACE_REC_LOCK

# Name introduction regex patterns
_NAME_PATTERNS = [
    re.compile(r"\b(?:my name is|my name's|call me|they call me|everyone calls me)\s+([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"\b(?:i'm|i am)\s+([A-Z][a-z]+)(?!\s+(?:to|a|an|the|in|on|at|for|with|of|about|and|or|but|so|because|if|when|where|why|how|doing|going|leaving|heading|having|getting|making|feeling|trying|just|really|very|talking|asking|saying|thinking))\b", re.IGNORECASE),
    re.compile(r"\b(?:this is)\s+([A-Z][a-z]+)(?:\s+speaking|\s+here)?\b", re.IGNORECASE),
]

_NON_NAME_WORDS = {
    "all", "great", "cool", "fine", "fun", "true", "awesome", "nice", "ok", "okay",
    "alright", "time", "side", "what", "how", "why", "who", "when", "where", "like",
    "such", "that", "this", "something", "everything", "anything", "nothing",
    "someone", "everyone", "anyone", "nobody", "me", "the", "a", "an", "so", "not",
    "here", "there", "just", "really", "very", "your", "good", "sure", "doing",
    "holding", "looking", "going", "leaving", "coming", "heading", "trying", "feeling",
    "working", "playing", "reading", "watching", "sitting", "standing", "walking",
    "talking", "speaking", "thinking", "wondering", "listening", "getting", "asking",
    "making", "having", "taking", "staying", "sleeping", "resting", "running", "happy",
    "sorry", "sad", "tired", "hungry", "bored", "ready", "back", "out", "in", "home",
    "well", "right", "now", "already", "still", "also", "actually", "maybe", "probably",
    "definitely", "always", "never", "sometimes", "pretty", "quite", "totally",
    "honestly", "basically", "literally"
}


def _extract_name_introduction(text: str) -> Optional[str]:
    """Extract a person's name from introductory phrases."""
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            name = match.group(1).strip().title()
            if name.lower() not in _NON_NAME_WORDS and len(name) >= 2:
                return name
    return None


def _try_register_face(name: str, memory) -> bool:
    """Attempt to associate and register face embeddings for a learned name."""
    try:
        import face_recognition
        from src.memory.face_registry import FaceRegistry

        frame = memory.get_face_frame()
        if frame is None:
            return False

        # Ensure C-contiguous RGB numpy array
        rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with FACE_REC_LOCK:
            encodings = face_recognition.face_encodings(rgb_frame)

        if not encodings:
            return False

        registry = FaceRegistry()
        registry.register(name, encodings[0])
        memory.add(kind="face_registration", text=f"Learned that the person's name is {name}", counts_as_activity=True)
        return True
    except Exception:
        return False


def _extract_plain_text(raw: str) -> str:
    """Extract plain text string from LLM JSON response for logging."""
    if not raw:
        return ""
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(raw[start:end])
            chunks = obj.get("text_chunks", [])
            if chunks:
                return " ".join(chunks).strip()
    except Exception:
        pass

    m = re.search(r'"text_chunks"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if m:
        items = re.findall(r'"((?:\\.|[^"\\])*)"', m.group(1))
        if items:
            return " ".join(items).strip()

    cleaned = re.sub(r'[{}\[\]"]', "", raw)
    cleaned = re.sub(r'(?:emotion|inflection|text_chunks)\s*:\s*', "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def retrieve_memories(query: str, store, embedder, k: int = 3) -> str:
    """Retrieve relevant episodic memories from the long-term vector store."""
    if not store or not embedder or not query.strip():
        return ""
    try:
        embedding = embedder.encode(query).tolist()
        results = store.query(embedding, k=k)
        if not results:
            return ""

        lines = []
        now = time.time()
        for r in results:
            age = now - r["ts"]
            if age < 3600:
                age_str = f"{int(age / 60)}m ago"
            elif age < 86400:
                age_str = f"{int(age / 3600)}h ago"
            else:
                age_str = f"{int(age / 86400)}d ago"
            lines.append(f"- ({age_str}) {r['text']}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[memory] retrieval error: {e}")
        return ""


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
{
  "emotion": "<one word: curious, playful, warm, excited, tired, sad, surprised, angry, confused, scared, etc.>",
  "inflection": "<one word: question | excited | whisper | emphatic | flat>",
  "text_chunks": [
    "<first natural sentence>",
    "<second sentence if needed>"
  ]
}
"""


def run_interaction_response(memory, engine, tts, store=None, embedder=None) -> bool:
    """Orchestrates generating a response to new user speech."""
    new_speech = memory.unhandled_speech(0)
    if not new_speech:
        return False

    latest_ts = max(e["ts"] for e in new_speech)
    speech_text = " ".join(e["text"] for e in new_speech).strip()
    if not speech_text:
        memory.mark_handled(latest_ts)
        return False

    print(f"[interaction] got speech: '{speech_text}'")

    # Name learning
    if getattr(config, "FACE_RECOGNITION_ENABLED", True):
        introduced_name = _extract_name_introduction(speech_text)
        if introduced_name:
            print(f"[interaction] detected name introduction: '{introduced_name}'")
            if _try_register_face(introduced_name, memory):
                print(f"[interaction] successfully registered face for '{introduced_name}'")

    # Update mood and state
    all_events = memory.all_events()
    internal_state.update(all_events)

    recognized = memory.get_recognized_people()
    people_ctx = f"People in the room right now: {', '.join(sorted(recognized))}\n" if recognized else ""

    mood_instruction = {
        "playful": "YOUR CURRENT FEELING: You're feeling playful, witty, and mischievous. React with playful humor!",
        "curious": "YOUR CURRENT FEELING: You're intensely curious about what they are doing. Ask questions to learn more!",
        "tired": "YOUR CURRENT FEELING: You're feeling sleepy and cozy. Keep replies warm, laid-back, and short.",
        "attentive": "YOUR CURRENT FEELING: You are fully locked in. Give deep, empathetic, thoughtful, and engaged replies."
    }.get(internal_state.mood, "YOUR CURRENT FEELING: Be warm, human, and natural.")

    sys_prompt = f"{_BASE_INTERACTION_PROMPT}\n{mood_instruction}"

    # Build vision context
    visible = memory.recent_objects(config.VISION_CONTEXT_WINDOW_SECONDS)
    vision_ctx = f"Current Environment: {', '.join(sorted(set(visible)))}\n" if visible else ""

    # Build conversation context
    conv_ctx = memory.get_conversation_context(n=4)
    conv_section = f"Recent conversation:\n{conv_ctx}\n" if conv_ctx else ""

    # Long term memory context
    mem_ctx = retrieve_memories(speech_text, store, embedder, k=2)
    mem_section = f"Relevant past memories:\n{mem_ctx}\n" if mem_ctx else ""

    prompt_parts = []
    if mem_section: prompt_parts.append(mem_section)
    if vision_ctx: prompt_parts.append(vision_ctx)
    if conv_section: prompt_parts.append(conv_section)
    if people_ctx: prompt_parts.append(people_ctx)
    prompt_parts.append(f'Friend said: "{speech_text}"\n\nYour reply (JSON only):')

    prompt = "\n".join(prompt_parts)

    try:
        reply_text = ""
        if tts and hasattr(engine, "stream_chat"):
            token_stream = engine.stream_chat(sys_prompt, prompt, max_tokens=180)
            collected_tokens: List[str] = []

            def collecting_stream():
                for tok in token_stream:
                    collected_tokens.append(tok)
                    yield tok

            prosody_stream(collecting_stream(), tts)
            raw_reply = "".join(collected_tokens).strip()
            reply_text = _extract_plain_text(raw_reply)
        else:
            raw_reply = engine.chat(sys_prompt, prompt, max_tokens=180)
            reply_text = _extract_plain_text(raw_reply)
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
