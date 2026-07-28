"""
Sleep cycle. Takes the whole raw working-memory log, has the LLM
summarize it into an episodic memory, embeds and stores each chunk in
the long-term vector store, archives the raw log, and clears working
memory for the next waking period.
"""
import json
import os
import time

import config


def _format_raw_log(events):
    lines = []
    for e in events:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["ts"]))
        lines.append(f"[{t}] ({e['kind']}) {e['text']}")
    return "\n".join(lines)


def _chunk_summary(summary_text, max_chars=400):
    """Split the summary into bullet-sized chunks for embedding.
    Each chunk should be independently retrievable/meaningful."""
    parts = [p.strip("-* \n") for p in summary_text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in parts:
        if len(buf) + len(p) > max_chars and buf:
            chunks.append(buf.strip())
            buf = ""
        buf += p + " "
    if buf.strip():
        chunks.append(buf.strip())
    return chunks or [summary_text.strip()]


def consolidate(memory, engine, embedder, store, tts=None, archive_dir=config.ARCHIVE_DIR):
    events = memory.all_events()
    if not events:
        print("[sleep] nothing to consolidate, skipping.")
        return

    print(f"[sleep] consolidating {len(events)} events...")
    if tts:
        tts.speak("I'm going to rest and think over what I've noticed.")

    raw_log = _format_raw_log(events)

    # 1. Archive the raw log first, no matter what happens next.
    os.makedirs(archive_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(archive_dir, f"raw_{stamp}.jsonl")
    with open(archive_path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    # 2. Summarize into an episodic memory.
    prompt = (
        "Below is a raw log of everything you perceived and thought during "
        "a waking period. Summarize it into a short list of the notable, "
        "memorable moments -- skip repetitive or trivial entries. Write in "
        "first person, past tense, as bullet points (one moment per line).\n\n"
        f"{raw_log}"
    )
    summary = engine.chat(config.PERSONA_SYSTEM_PROMPT, prompt, max_tokens=500,
                           temperature=0.5)

    if not summary:
        print("[sleep] summarization produced nothing, aborting consolidation.")
        return

    # 3. Chunk + embed + store.
    chunks = _chunk_summary(summary)
    for chunk in chunks:
        embedding = embedder.encode(chunk).tolist()
        store.add(chunk, embedding, kind="episodic_summary")

    print(f"[sleep] stored {len(chunks)} memory chunks. raw log archived to "
          f"{archive_path}")

    # 4. Clear working memory for the next waking period.
    memory.clear()


def run_idle_watcher(memory, stop_event, on_sleep, idle_minutes=config.IDLE_SLEEP_MINUTES,
                      check_every=10):
    """Fires on_sleep() once after idle_minutes of no new sensor activity,
    then re-arms only after new activity is seen (so it doesn't re-trigger
    every check_every seconds while sitting idle)."""
    already_fired = False
    while not stop_event.is_set():
        stop_event.wait(check_every)
        if stop_event.is_set():
            break

        idle_for = time.time() - memory.last_activity_ts
        if idle_for >= idle_minutes * 60:
            if not already_fired:
                print(f"[idle] {idle_minutes} min of silence -- going to sleep.")
                on_sleep()
                already_fired = True
        else:
            already_fired = False
