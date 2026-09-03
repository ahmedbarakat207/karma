import json
import re
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np

from src import config
from src.speech.prosody import prosody_stream
from src.state import internal_state
from src.memory.face_registry import FACE_REC_LOCK

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
    for pattern in _NAME_PATTERNS:
        m = pattern.search(text)
        if m:
            name = m.group(1).strip().title()
            if name.lower() not in _NON_NAME_WORDS and len(name) >= 2:
                return name
    return None


def _try_register_face(name: str, memory) -> bool:
    try:
        import face_recognition
        from src.memory.face_registry import FaceRegistry

        frame = memory.get_face_frame()
        if frame is None:
            return False

        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with FACE_REC_LOCK:
            encodings = face_recognition.face_encodings(rgb)

        if not encodings:
            return False

        FaceRegistry().register(name, encodings[0])
        memory.add(kind="face_registration", text=f"Learned that the person's name is {name}", counts_as_activity=True)
        return True
    except Exception:
        return False


def _deduplicate_phrase_loops(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'(\b[\w!\?\'\.]+(?:\s+[\w!\?\'\.]+){0,3}\b)(?:\s*[,!?]?\s*\1){2,}', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'(\b\w+[\'!]?)(?:\s*[,!?]?\s*\1){2,}', r'\1', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


def extract_code_blocks(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    if not text or "```" not in text:
        return text, None, None

    m = re.search(r'```(?:(\w+)\s*\n)?(.*?)```', text, flags=re.DOTALL)
    if not m:
        return text, None, None

    lang = m.group(1) or "code"
    code = m.group(2).strip()
    spoken = re.sub(r'\s+', ' ', (text[:m.start()] + " " + text[m.end():])).strip()
    return spoken or "Here is the code on screen.", code, lang


def clean_companion_reply(text: str) -> str:
    if not text:
        return text
    orig = text
    while True:
        prev = text
        text = re.sub(r'^(?:friend said|user said|user|karma|friend)[!:,.\s\"-]*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^(?:hello|hi|hey)(?: there)?[!,.]?\s*how can i (?:assist|help) you(?: today)?\??', "Hey! What's up?", text, flags=re.IGNORECASE).strip()
        text = re.sub(r'how can i (?:assist|help) you(?: today)?\??', "what's up?", text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^as (?:an? )?(?:ai|artificial intelligence|language model|human friend|friend|machine)[^.!?\n]*(?:[.,!?]|\b(?:but|however),?)\s*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^i (?:do not|don\'t) have (?:personal )?(?:preferences|emotions|feelings)[^.!?\n]*(?:[.,!?]|\b(?:but|however),?)\s*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^(?:however|but),?\s*', '', text, flags=re.IGNORECASE).strip()
        if text == prev:
            break
    if text and text != orig:
        text = text[0].upper() + text[1:]
    return text


def _extract_plain_text(raw: str) -> str:
    if not raw:
        return ""

    s = raw.strip()
    if "```" in s and not s.startswith("```json"):
        return clean_companion_reply(s)

    if s.startswith("```json"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE).strip()
        s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE).strip()

    try:
        start = s.find("{")
        end = s.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(s[start:end])
            for key in ("text_chunks", "response", "reply", "message", "text"):
                val = obj.get(key)
                if val:
                    text = " ".join(str(c) for c in val).strip() if isinstance(val, list) else str(val).strip()
                    if text:
                        return clean_companion_reply(_deduplicate_phrase_loops(text))
    except Exception:
        pass

    m = re.search(r'"(?:text_chunks|response|reply|message)"\s*:\s*(\[.*?\]|"(?:\\.|[^"\\])*")', s, re.DOTALL)
    if m:
        val = m.group(1).strip()
        if val.startswith("["):
            items = re.findall(r'"((?:\\.|[^"\\])*)"', val)
            if items:
                return clean_companion_reply(_deduplicate_phrase_loops(" ".join(items)))
        elif val.startswith('"') and val.endswith('"'):
            return clean_companion_reply(_deduplicate_phrase_loops(val[1:-1]))

    if s.startswith("{") and s.endswith("}"):
        cleaned = re.sub(r'[{}\[\]"]', "", s)
        cleaned = re.sub(r'(?:emotion|inflection|text_chunks|response|reply|message)\s*:\s*', "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^(?:playful|curious|excited|tired|sad|attentive|warm|surprised|whisper|question|emphatic|flat|neutral)[,\s]+', '', cleaned, flags=re.IGNORECASE)
        return clean_companion_reply(_deduplicate_phrase_loops(cleaned.strip()))

    return clean_companion_reply(_deduplicate_phrase_loops(s))


def retrieve_memories(query: str, store, embedder, k: int = 3) -> str:
    if not store or not embedder or not query.strip():
        return ""
    try:
        results = store.query(embedder.encode(query).tolist(), k=k)
        if not results:
            return ""
        now = time.time()
        lines = []
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


_BASE_INTERACTION_PROMPT = """You are Karma, a witty, chill friend hanging out in the room.
You speak naturally like a real person with your own opinions and tastes.
You love music, especially jazz, lo-fi, and indie rock.
Never give textbook definitions or lecture like an encyclopedia.
Never sound like a customer service bot or AI assistant.
Keep replies brief (1-2 sentences) like a real casual conversation.
If code is requested, provide a brief conversational note and enclose the code in a ```lang``` block."""


def _check_kiosk_intent(text: str) -> Optional[Tuple[str, Optional[int]]]:
    t = text.lower()

    if any(p in t for p in ["open the map", "show the map", "show map", "open map", "facility map",
                              "where are we", "floor map", "where am i", "show floor", "open floor", "view floor"]):
        if "floor 2" in t or "second floor" in t:
            return ("map", 1)
        elif "floor 1" in t or "first floor" in t or "ground floor" in t:
            return ("map", 0)
        return ("map", None)

    if any(p in t for p in ["show achievements", "open achievements", "view achievements",
                              "show awards", "my achievements", "what are our achievements", "show milestones"]):
        return ("achievements", None)

    if any(p in t for p in ["student apps", "student projects", "show apps",
                              "open apps", "show student apps", "open projects"]):
        return ("apps", None)

    if any(p in t for p in ["open documents", "show documents", "open pdf",
                              "show pdf", "read manual", "open menu", "show menu"]):
        return ("docs", None)

    if any(p in t for p in ["close menu", "close map", "back to face",
                              "show face", "exit kiosk", "hide menu"]):
        return ("face", None)

    return None


def run_interaction_response(memory, engine, tts, store=None, embedder=None) -> bool:
    new_speech = memory.unhandled_speech(0)
    if not new_speech:
        return False

    latest_ts = max(e["ts"] for e in new_speech)
    speech_text = " ".join(e["text"] for e in new_speech).strip()
    if not speech_text:
        memory.mark_handled(latest_ts)
        return False

    config.log_debug(f"[interaction] got speech: '{speech_text}'")

    kiosk_notice = ""
    kiosk_action = _check_kiosk_intent(speech_text)
    if kiosk_action:
        view_name, floor_idx = kiosk_action
        try:
            from src.ui.kiosk import kiosk_manager
            if view_name == "face":
                kiosk_manager.close()
                kiosk_notice = "You just closed the touchscreen kiosk and returned to your digital face."
            else:
                kiosk_manager.open_view(view_name, floor_idx=floor_idx)
                kiosk_notice = f"You tilted your head up to 135 degrees and opened the {view_name.upper()} screen on your 7-inch LCD touchscreen for the user."
            config.log_debug(f"[interaction] voice kiosk: {view_name}")
        except Exception as e:
            config.log_debug(f"[interaction] kiosk error: {e}")

    if getattr(config, "FACE_RECOGNITION_ENABLED", True):
        name = _extract_name_introduction(speech_text)
        if name:
            config.log_debug(f"[interaction] name introduced: '{name}'")
            if _try_register_face(name, memory):
                config.log_debug(f"[interaction] registered face for '{name}'")

    internal_state.update(memory.all_events())

    recognized = memory.get_recognized_people()
    people_ctx = f"People in the room right now: {', '.join(sorted(recognized))}\n" if recognized else ""

    mood_instruction = {
        "playful":   "YOUR CURRENT FEELING: You're feeling playful, witty, and mischievous. React with playful humor!",
        "curious":   "YOUR CURRENT FEELING: You're intensely curious about what they are doing. Ask questions to learn more!",
        "tired":     "YOUR CURRENT FEELING: You're feeling sleepy and cozy. Keep replies warm, laid-back, and short.",
        "attentive": "YOUR CURRENT FEELING: You are fully locked in. Give deep, empathetic, thoughtful, and engaged replies."
    }.get(internal_state.mood, "YOUR CURRENT FEELING: Be warm, human, and natural.")

    sys_prompt = f"{_BASE_INTERACTION_PROMPT}\n{mood_instruction}"

    visible = memory.recent_objects(config.VISION_CONTEXT_WINDOW_SECONDS)
    vision_ctx = f"Current Environment: {', '.join(sorted(set(visible)))}\n" if visible else ""

    history = memory.get_conversation_turns(n=4)

    mem_ctx = retrieve_memories(speech_text, store, embedder, k=2)
    mem_section = f"Relevant past memories:\n{mem_ctx}\n" if mem_ctx else ""

    doc_ctx = ""
    try:
        from src.memory.rag import retrieve_document_context
        doc_ctx = retrieve_document_context(speech_text, store, embedder, k=2)
    except Exception:
        pass
    doc_section = f"Knowledge from reference documents:\n{doc_ctx}\n" if doc_ctx else ""

    parts = []
    if kiosk_notice:  parts.append(f"System Action: {kiosk_notice}")
    if doc_section:   parts.append(doc_section)
    if mem_section:   parts.append(mem_section)
    if vision_ctx:    parts.append(vision_ctx)
    if people_ctx:    parts.append(people_ctx)

    if parts:
        user_prompt = f"{''.join(parts)}\n{speech_text}"
    else:
        user_prompt = speech_text

    try:
        reply_text = ""
        if tts and hasattr(engine, "stream_chat"):
            collected: List[str] = []

            def collecting_stream():
                for tok in engine.stream_chat(sys_prompt, user_prompt, max_tokens=180, history=history):
                    collected.append(tok)
                    yield tok

            prosody_stream(collecting_stream(), tts)
            reply_text = _extract_plain_text("".join(collected).strip())

            if not reply_text:
                raw = engine.chat(sys_prompt, user_prompt, max_tokens=180, history=history)
                reply_text = _extract_plain_text(raw)
                if tts and reply_text and len(reply_text) > 1:
                    spoken, code, lang = extract_code_blocks(reply_text)
                    if code:
                        internal_state.set_active_code(code, lang=lang)
                    if spoken:
                        tts.speak(clean_companion_reply(spoken))
        else:
            raw = engine.chat(sys_prompt, user_prompt, max_tokens=180, history=history)
            reply_text = _extract_plain_text(raw)
            if tts and reply_text and len(reply_text) > 1:
                spoken, code, lang = extract_code_blocks(reply_text)
                if code:
                    internal_state.set_active_code(code, lang=lang)
                if spoken:
                    tts.speak(clean_companion_reply(spoken))

        if reply_text and len(reply_text) > 1:
            spoken, code, lang = extract_code_blocks(reply_text)
            clean_spoken = clean_companion_reply(spoken)
            if code:
                internal_state.set_active_code(code, lang=lang)
            print(f"[reply] ({internal_state.mood}) {clean_spoken}")
            internal_state.set_karma_speech(clean_spoken)
            memory.add(kind="reply", text=clean_spoken, counts_as_activity=True)
            memory.add_conversation(speech_text, clean_spoken)

    except Exception as e:
        config.log_debug(f"[interaction] generation error: {e}")
    finally:
        memory.mark_handled(latest_ts)

    return True
