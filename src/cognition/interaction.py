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


def _deduplicate_phrase_loops(text: str) -> str:
    """Detects and eliminates runaway token repetition loops."""
    if not text:
        return ""
    # Trims phrases of 1-4 words repeated 3+ times (e.g. 'Ahhh! Ahhh! Ahhh! ...')
    pattern = r'(\b[\w!\?\'\.-]+(?:\s+[\w!\?\'\.-]+){0,3}\b)(?:\s*[,!?]?\s*\1){2,}'
    deduped = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
    # Also clean single words repeated multiple times with punctuation
    deduped = re.sub(r'(\b\w+[\'!]?)(?:\s*[,!?]?\s*\1){2,}', r'\1', deduped, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', deduped).strip()


def _extract_plain_text(raw: str) -> str:
    """Extract plain text string from LLM JSON response for speaking and logging."""
    if not raw:
        return ""

    cleaned_raw = raw.strip()
    # Strip markdown fences if emitted by model
    if "```" in cleaned_raw:
        cleaned_raw = re.sub(r"^```(?:json)?\s*", "", cleaned_raw, flags=re.MULTILINE).strip()
        cleaned_raw = re.sub(r"\s*```$", "", cleaned_raw, flags=re.MULTILINE).strip()

    try:
        start = cleaned_raw.find("{")
        end = cleaned_raw.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(cleaned_raw[start:end])
            for key in ("text_chunks", "response", "reply", "message", "text"):
                val = obj.get(key)
                if val:
                    if isinstance(val, list):
                        text = " ".join(str(c) for c in val).strip()
                    else:
                        text = str(val).strip()
                    if text:
                        return _deduplicate_phrase_loops(text)
    except Exception:
        pass

    m = re.search(r'"(?:text_chunks|response|reply|message)"\s*:\s*(\[.*?\]|"(?:\\.|[^"\\])*")', cleaned_raw, re.DOTALL)
    if m:
        val_str = m.group(1).strip()
        if val_str.startswith("["):
            items = re.findall(r'"((?:\\.|[^"\\])*)"', val_str)
            if items:
                return _deduplicate_phrase_loops(" ".join(items).strip())
        elif val_str.startswith('"') and val_str.endswith('"'):
            return _deduplicate_phrase_loops(val_str[1:-1].strip())

    cleaned = re.sub(r'[{}\[\]"]', "", cleaned_raw)
    cleaned = re.sub(r'(?:emotion|inflection|text_chunks|response|reply|message)\s*:\s*', "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^(?:playful|curious|excited|tired|sad|attentive|warm|surprised|whisper|question|emphatic|flat|neutral)[,\s]+', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return _deduplicate_phrase_loops(cleaned)



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
        config.log_debug(f"[memory] retrieval error: {e}")
        return ""


_BASE_INTERACTION_PROMPT = """You are a warm, genuine, witty human friend having a real spoken conversation with your close friend in your shared space.

HUMAN CONVERSATION STYLE:
- Speak naturally like a real human friend talking out loud.
- Use natural contractions: "I'm", "that's", "it's", "don't", "can't", "you're", "let's", "gonna".
- Express real human emotions: excitement, empathy, humor, curiosity, playfulness, warmth.
- Keep responses short and conversational (1-2 sentences) so the dialogue flows naturally back and forth.
- Never repeat words or sentences in a loop.
- Never sound like an AI assistant, customer service rep, or formal computer program.
- Never mention cameras, sensors, vision models, or AI mechanisms.

OUTPUT FORMAT — CRITICAL:
You MUST respond directly with ONLY a raw JSON object starting with { and ending with }.
Never use markdown code blocks or backticks (no ``` or ```json).
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

    config.log_debug(f"[interaction] got speech: '{speech_text}'")

    # Name learning
    if getattr(config, "FACE_RECOGNITION_ENABLED", True):
        introduced_name = _extract_name_introduction(speech_text)
        if introduced_name:
            config.log_debug(f"[interaction] detected name introduction: '{introduced_name}'")
            if _try_register_face(introduced_name, memory):
                config.log_debug(f"[interaction] successfully registered face for '{introduced_name}'")


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

    # Document RAG context (PDF manuals, reference books, notes)
    doc_ctx = ""
    try:
        from src.memory.rag import retrieve_document_context
        doc_ctx = retrieve_document_context(speech_text, store, embedder, k=2)
    except Exception:
        pass
    doc_section = f"Knowledge from reference documents:\n{doc_ctx}\n" if doc_ctx else ""

    prompt_parts = []
    if doc_section: prompt_parts.append(doc_section)
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

            # Fallback to direct chat if stream produced nothing
            if not reply_text:
                raw_reply = engine.chat(sys_prompt, prompt, max_tokens=180)
                reply_text = _extract_plain_text(raw_reply)
                if tts and reply_text and len(reply_text) > 1:
                    tts.speak(reply_text)
        else:
            raw_reply = engine.chat(sys_prompt, prompt, max_tokens=180)
            reply_text = _extract_plain_text(raw_reply)
            if tts and reply_text and len(reply_text) > 1:
                tts.speak(reply_text)

        if reply_text and len(reply_text) > 1:
            print(f"[reply] ({internal_state.mood}) {reply_text}")
            internal_state.set_karma_speech(reply_text)
            memory.add(kind="reply", text=reply_text, counts_as_activity=True)
            memory.add_conversation(speech_text, reply_text)

    except Exception as e:
        config.log_debug(f"[interaction] generation error: {e}")
    finally:
        memory.mark_handled(latest_ts)



    return True
