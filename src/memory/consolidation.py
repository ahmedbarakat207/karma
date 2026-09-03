
import json
import os
import time
from typing import List, Dict, Any

from src import config


def _format_raw_log(events: List[Dict[str, Any]]) -> str:
    lines = []
    for e in events:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["ts"]))
        lines.append(f"[{t}] ({e['kind']}) {e['text']}")
    return "\n".join(lines)


def _chunk_summary(summary_text: str, max_chars: int = 400) -> List[str]:
    parts = [p.strip("-* \n") for p in summary_text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""

    for p in parts:
        if len(buf) + len(p) > max_chars and buf:
            chunks.append(buf.strip())
            buf = ""
        buf += p + " "

    if buf.strip():
        chunks.append(buf.strip())
    return chunks or [summary_text.strip()]


def consolidate(memory, engine, embedder, store, tts=None, archive_dir: str = config.ARCHIVE_DIR) -> None:
    events = memory.all_events()
    if not events:
        print("[sleep] nothing to consolidate, skipping.")
        return

    print(f"[sleep] consolidating {len(events)} events...")
    if tts:
        try:
            tts.speak("I'm going to rest and think over what I've noticed.")
        except Exception:
            pass

    raw_log = _format_raw_log(events)

    os.makedirs(archive_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(archive_dir, f"raw_{stamp}.jsonl")
    try:
        with open(archive_path, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
    except Exception as e:
        print(f"[sleep] archive write warning: {e}")

    prompt = (
        "Below is a raw log of everything you perceived and thought during a waking period. "
        "Summarize it into a short list of the notable, memorable moments -- skip repetitive or trivial entries. "
        "Write in first person, past tense, as bullet points (one moment per line).\n\n"
        f"{raw_log}"
    )

    try:
        summary = engine.chat(config.PERSONA_SYSTEM_PROMPT, prompt, max_tokens=500, temperature=0.5)
    except Exception as e:
        print(f"[sleep] summarization error: {e}")
        summary = None

    if not summary:
        print("[sleep] summarization produced nothing, aborting consolidation.")
        return

    chunks = _chunk_summary(summary)
    for chunk in chunks:
        try:
            embedding = embedder.encode(chunk).tolist()
            store.add(chunk, embedding, kind="episodic_summary")
        except Exception as e:
            print(f"[sleep] embedding/store error on chunk: {e}")

    print(f"[sleep] stored {len(chunks)} memory chunks. Raw log archived to {archive_path}")

    memory.clear()


def run_idle_watcher(memory, stop_event, on_sleep, idle_minutes: int = config.IDLE_SLEEP_MINUTES,
                      check_every: int = 10) -> None:
    already_fired = False
    while not stop_event.is_set():
        stop_event.wait(check_every)
        if stop_event.is_set():
            break

        idle_for = time.time() - memory.last_activity_ts
        if idle_for >= (idle_minutes * 60):
            if not already_fired:
                print(f"[idle] {idle_minutes} min of silence -- initiating sleep cycle.")
                on_sleep()
                already_fired = True
        else:
            already_fired = False
