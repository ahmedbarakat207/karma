import json
import re
import sys
import threading
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np

from src import config
from src.speech.prosody import prosody_stream
from src.speech.arabic_g2p import is_arabic
from src.state import internal_state
from src.memory.face_registry import FACE_REC_LOCK
from src.ui import events as _events

INTERACTION_LOCK = threading.Lock()


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

    code_blocks = []
    langs = []

    def repl(m):
        lang = m.group(1) or "code"
        code = m.group(2).strip()
        langs.append(lang)
        code_blocks.append(code)
        return ""

    spoken = re.sub(r'```(?:(\w+)\s*)?\n?(.*?)```', repl, text, flags=re.DOTALL)
    spoken = re.sub(r'\s+', ' ', spoken).strip()

    if not code_blocks:
        return text, None, None

    combined_code = "\n\n".join(code_blocks)
    primary_lang = langs[0] if langs else "code"
    return spoken or "Here is the code on screen.", combined_code, primary_lang


def clean_companion_reply(text: str, user_input: Optional[str] = None) -> str:
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

    # Strip bilingual / translation leakage
    is_ar_user = any('\u0600' <= ch <= '\u06FF' for ch in user_input) if user_input else False
    if is_ar_user:
        m = re.search(r'\*?بالعامية(?: المصرية)?:\*?\s*(.+)$', text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    else:
        text = re.sub(r'\n+\s*\*?بالعامية(?: المصرية)?:\*?.*$', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\n+\s*(?:In Egyptian Arabic|Egyptian Arabic|Arabic translation):\s*.*$', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

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


def retrieve_memories(query: str, store, embedder, k: int = 3, threshold: float = 1.15) -> str:
    if not store or not embedder or not query.strip():
        return ""
    try:
        results = store.query(embedder.encode(query).tolist(), k=k, kind="memory")
        if not results:
            results = store.query(embedder.encode(query).tolist(), k=k, kind="episodic_summary")
        if not results:
            return ""
        now = time.time()
        lines = []
        for r in results:
            if r.get("distance", 2.0) > threshold:
                continue
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


def active_base_prompt() -> str:
    """System prompt in effect: dashboard override wins, else the default."""
    override = (getattr(config, "PERSONA_OVERRIDE", "") or "").strip()
    return override or _BASE_INTERACTION_PROMPT


def _check_kiosk_intent(text: str) -> Optional[Tuple[str, Optional[int]]]:
    t = text.lower()

    if any(p in t for p in [
        "open the map", "show the map", "show map", "open map", "facility map",
        "where are we", "floor map", "where am i", "show floor", "open floor", "view floor",
        "افتح الخريطة", "افتح خريطة", "وريني الخريطة", "وريني خريطة", "الخريطة", "خريطة", "إحنا فين", "احنا فين", "خريطة المبنى", "فين الدور"
    ]):
        if any(f in t for f in ["floor 2", "second floor", "الدور التاني", "الدور الثاني", "دور 2", "دور تاني"]):
            return ("map", 1)
        elif any(f in t for f in ["floor 1", "first floor", "ground floor", "الدور الاول", "الدور الأول", "دور 1", "الأرضي", "الارضي"]):
            return ("map", 0)
        return ("map", None)

    if any(p in t for p in [
        "show achievements", "open achievements", "view achievements",
        "show awards", "my achievements", "what are our achievements", "show milestones",
        "افتح الإنجازات", "افتح الانجازات", "وريني الانجازات", "الانجازات", "إنجازاتنا", "انجازاتنا", "الشهادات"
    ]):
        return ("achievements", None)

    if any(p in t for p in [
        "student apps", "student projects", "show apps",
        "open apps", "show student apps", "open projects",
        "افتح المشاريع", "وريني المشاريع", "مشاريع الطلاب", "المشاريع", "البرامج", "التطبيقات", "افتح التطبيقات"
    ]):
        return ("apps", None)

    if any(p in t for p in [
        "open documents", "show documents", "open pdf",
        "show pdf", "read manual", "open menu", "show menu",
        "اقرا الملفات", "اقرأ الملفات", "افتح الملفات", "افتح المستندات", "المستندات", "الكتالوج", "المانيوال", "الملفات"
    ]):
        return ("docs", None)

    if any(p in t for p in [
        "close menu", "close map", "back to face",
        "show face", "exit kiosk", "hide menu",
        "اقفل القائمة", "اقفل الخريطة", "ارجع للوش", "ارجع للشاشة", "اقفل", "كفاية", "الوش"
    ]):
        return ("face", None)

    return None


def speak_and_animate(text: str, tts=None) -> None:
    """Speak text through TTS and animate the robot screen's mouth and subtitles."""
    def _worker():
        if not text:
            return
        from src.speech.tts import clean_for_speech
        spoken = clean_for_speech(text)
        if not spoken:
            return

        internal_state.set_karma_speech(spoken)
        internal_state.set_playing_audio(True)
        try:
            from src.ui.server import broadcast_state_threadsafe
            broadcast_state_threadsafe()
        except Exception:
            pass

        engine_tts = tts
        if engine_tts is None:
            try:
                from src.ui.server import _runtime
                engine_tts = _runtime.get("tts")
            except Exception:
                pass
        if engine_tts is None:
            try:
                from src.speech.tts import TTSEngine
                engine_tts = TTSEngine()
                try:
                    from src.ui.server import set_runtime
                    set_runtime(tts=engine_tts)
                except Exception:
                    pass
            except Exception as te:
                print(f"[interaction] fallback TTS init error: {te}", file=sys.stderr)

        audio_played = False
        if engine_tts is not None:
            try:
                audio_played = bool(engine_tts.speak(spoken))
            except Exception as e:
                print(f"[interaction] TTS speak error: {e}", file=sys.stderr)

        if not audio_played:
            # Fallback: keep mouth animating on the screen for the estimated speech duration
            internal_state.set_playing_audio(True)
            try:
                from src.ui.server import broadcast_state_threadsafe
                broadcast_state_threadsafe()
            except Exception:
                pass

            est_duration = max(2.5, min(8.0, len(spoken.split()) * 0.45))
            t0 = time.time()
            while time.time() - t0 < est_duration:
                time.sleep(0.05)

        internal_state.set_playing_audio(False)
        try:
            from src.ui.server import broadcast_state_threadsafe
            broadcast_state_threadsafe()
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True, name="karma_speak_animate").start()



def run_interaction_response(memory, engine, tts=None, store=None, embedder=None, direct_text: Optional[str] = None) -> Optional[str]:

    with INTERACTION_LOCK:
        if direct_text is not None:
            speech_text = str(direct_text).strip()
            if not speech_text:
                return None
            memory.add(kind="speech", text=speech_text, counts_as_activity=True)
            latest_ts = time.time() + 0.05
            memory.mark_handled(latest_ts)
        else:
            new_speech = memory.unhandled_speech(0)
            if not new_speech:
                return None

            latest_ts = max(e["ts"] for e in new_speech)
            speech_text = " ".join(e["text"] for e in new_speech).strip()
            if not speech_text:
                memory.mark_handled(latest_ts)
                return None

        config.log_debug(f"[interaction] got speech: '{speech_text}'")

        if engine is None:
            print("[interaction error] No LLM engine available", file=sys.stderr)
            _events.post("error", "No LLM engine available")
            memory.mark_handled(latest_ts)
            return None

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
                _events.post("kiosk", f"{speech_text} -> {view_name}")
            except Exception as e:
                config.log_debug(f"[interaction] kiosk error: {e}")

        movement_notice = ""
        try:
            from src.cognition.movement_intents import check_move_intent
            move_action = check_move_intent(speech_text)
        except Exception as e:
            config.log_debug(f"[interaction] move intent error: {e}")
            move_action = None
        if move_action:
            action, param = move_action
            try:
                from src.hardware.drive import drive_base
                from src.navigation.explorer import explorer
                if action == "stop":
                    explorer.stop()
                    movement_notice = "You stopped moving and are staying put."
                elif action == "stop_explore":
                    explorer.stop()
                    movement_notice = "You stopped exploring and are hanging out where you are."
                elif action == "unfollow":
                    explorer.stop()
                    movement_notice = "You stopped following and are staying here."
                elif action == "explore":
                    explorer.start_explore()
                    movement_notice = "You started wandering slowly to explore the room."
                elif action == "follow":
                    explorer.start_follow()
                    movement_notice = "You started following the person, rolling slowly toward them."
                elif action == "goto":
                    if explorer.goto(param):
                        movement_notice = f"You are rolling toward the {param}."
                    else:
                        movement_notice = ""
                elif action in ("forward", "backward", "left", "right"):
                    explorer.stop()  # leave auto modes before a manual nudge
                    ok = {
                        "forward": lambda: drive_base.forward(duration=1.0),
                        "backward": lambda: drive_base.backward(duration=1.0),
                        "left": lambda: drive_base.turn_left(duration=0.8),
                        "right": lambda: drive_base.turn_right(duration=0.8),
                    }[action]()
                    if ok:
                        movement_notice = {
                            "forward": "You rolled forward a little.",
                            "backward": "You backed up a little.",
                            "left": "You turned left a little.",
                            "right": "You turned right a little.",
                        }[action]
                        memory.add(kind="movement", text=f"Manual move: {action}",
                                   counts_as_activity=True, salience=0.3)
                    else:
                        movement_notice = "You wanted to move but the drive is stopped."
                config.log_debug(f"[interaction] voice move: {action} {param}")
                if movement_notice:
                    _events.post("movement", f"{speech_text} -> {action} {param}".strip())
            except Exception as e:
                config.log_debug(f"[interaction] movement error: {e}")

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

        sys_prompt = f"{active_base_prompt()}\n{mood_instruction}"

        visible = memory.recent_objects(config.VISION_CONTEXT_WINDOW_SECONDS)
        try:
            # VLM corrections apply at read time: while a correction is active,
            # the prompt sees what was verified ("microwave"), not the raw YOLO
            # label ("tv"). Tracking itself always stays on raw YOLO boxes.
            from src.vision.vlm import verifier as _vlm_verifier
            visible = _vlm_verifier.corrections.apply(list(visible or []))
        except Exception:
            pass
        vision_ctx = f"Current Environment: {', '.join(sorted(set(visible)))}\n" if visible else ""

        vlm_ctx = ""
        try:
            scenes = memory.recent_by_kind("vlm_scene", 120)
            if scenes:
                vlm_ctx = f"Verified sighting: {scenes[-1]}\n"
        except Exception:
            pass

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

        sys_parts = [f"{active_base_prompt()}\n{mood_instruction}"]
        if is_arabic(speech_text):
            sys_parts.append("\nLANGUAGE NOTE: The user is speaking in Arabic. Respond naturally in friendly, witty Egyptian Arabic (عامية مصرية), keeping your answer brief (1-2 sentences).")
        if people_ctx:    sys_parts.append(f"\nPeople present: {people_ctx.strip()}")
        if vision_ctx:    sys_parts.append(f"\nVisual observations: {vision_ctx.strip()}")
        if vlm_ctx:       sys_parts.append(f"\n{vlm_ctx.strip()}")
        try:
            from src.hardware.drive import drive_base as _drive_base
            from src.navigation.explorer import explorer as _explorer
            _loc = memory.consciousness.self_model.get("location", "desk")
            sys_parts.append(f"\nMovement: you are on wheels near the {_loc}. {_explorer.describe()}; base={_drive_base.describe()}. Narrate moves briefly, never lecture.")
        except Exception:
            pass
        if mem_section:   sys_parts.append(f"\nRelevant memories:\n{mem_section.strip()}")
        if doc_section:   sys_parts.append(f"\nReference knowledge:\n{doc_section.strip()}")
        if kiosk_notice:  sys_parts.append(f"\nSystem note: {kiosk_notice}")
        if movement_notice: sys_parts.append(f"\nSystem note: {movement_notice}")

        sys_prompt = "\n".join(sys_parts)
        user_prompt = speech_text

        clean_spoken: Optional[str] = None
        try:
            # Fast generation: 75 max_tokens is optimal for 1-2 punchy conversational sentences
            raw = engine.chat(sys_prompt, user_prompt, max_tokens=75, history=history)
            reply_text = _extract_plain_text(raw)

            if reply_text and len(reply_text) > 1:
                spoken, code, lang = extract_code_blocks(reply_text)
                clean_spoken = clean_companion_reply(spoken)
                if code:
                    internal_state.set_active_code(code, lang=lang)
                print(f"[reply] ({internal_state.mood}) {clean_spoken}")
                internal_state.set_karma_speech(clean_spoken)
                _events.post("reply", clean_spoken, {"mood": internal_state.mood})
                memory.add(kind="reply", text=clean_spoken, counts_as_activity=True)
                memory.add_conversation(speech_text, clean_spoken)

                # Speak audio out loud AND animate mouth on the robot's physical display
                speak_and_animate(clean_spoken, tts)

            if not clean_spoken:
                print("[interaction warning] LLM returned empty response", file=sys.stderr)
                _events.post("error", "LLM returned empty response")


        except Exception as e:
            print(f"[interaction error] {e}", file=sys.stderr)
            _events.post("error", f"LLM error: {e}")
            config.log_debug(f"[interaction] generation error: {e}")
        finally:
            memory.mark_handled(latest_ts)

        return clean_spoken

