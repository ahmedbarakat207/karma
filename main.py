"""
Entry point. Starts perception (vision + audio), cognition (think loop),
and sleep management (idle watcher + manual trigger via stdin), then just
keeps the process alive until you type "quit".

Usage:
    python main.py
    (while running) type "sleep" + Enter to force consolidation now
    (while running) type "quit" + Enter to shut down cleanly
"""
import sys
import threading
import time
import random

import config
from llm_engine import create_engine
from working_memory import WorkingMemory
from memory_store import MemoryStore
from vision import run_vision
from audio import run_audio
from think import think_immediately, think_quietly
from interaction import run_interaction_response
from consolidation import consolidate, run_idle_watcher
from speech import TTSEngine

from sentence_transformers import SentenceTransformer

def consciousness_orchestrator(memory, engine, stop_event, tts, store, embedder, speaking_event):
    print("[main] Consciousness orchestrator started.")
    while not stop_event.is_set():
        state = memory.consciousness
        
        # 1. If salience is high, interrupt and speak
        if state.prediction_error > 0.7:
            think_immediately(memory, engine, tts, store, embedder, urgency="HIGH", speaking_event=speaking_event)
        
        # 2. If user is speaking or there's unhandled speech, switch to Interaction mode
        elif memory.is_user_speaking() or memory.unhandled_speech(0):
            run_interaction_response(memory, engine, tts)
        
        # 3. Background idle thought (the 'hum' of consciousness)
        else:
            time.sleep(2)
            if random.random() < 0.1:  # 10% chance to quietly reflect
                think_quietly(memory, engine, store, embedder)

def main():
    print("[main] loading LLM...")
    engine = create_engine()

    print("[main] loading embedding model...")
    embedder = SentenceTransformer(config.EMBED_MODEL_NAME)

    store = MemoryStore()
    memory = WorkingMemory()
    stop_event = threading.Event()

    # shared flag so the mic thread mutes itself while the agent is talking,
    # instead of transcribing its own voice as something it "heard"
    speaking_event = threading.Event()

    print("[main] loading TTS engine...")
    tts = TTSEngine(speaking_event=speaking_event)

    consolidation_lock = threading.Lock()

    def do_sleep():
        # guard against the idle watcher and a manual "sleep" command
        # firing at the same moment
        if consolidation_lock.acquire(blocking=False):
            try:
                consolidate(memory, engine, embedder, store, tts=tts)
            finally:
                consolidation_lock.release()

    def listen_stdin():
        try:
            for line in sys.stdin:
                cmd = line.strip().lower()
                if cmd in ("sleep", "sleep now"):
                    threading.Thread(target=do_sleep, daemon=True).start()
                elif cmd in ("quit", "exit"):
                    stop_event.set()
                    break
        except Exception:
            pass

    threads = [
        threading.Thread(target=run_audio, args=(memory, stop_event, speaking_event), daemon=True),
        threading.Thread(target=consciousness_orchestrator, args=(memory, engine, stop_event, tts, store, embedder, speaking_event), daemon=True),
        threading.Thread(target=run_idle_watcher, args=(memory, stop_event, do_sleep), daemon=True),
        threading.Thread(target=listen_stdin, daemon=True),
    ]
    for t in threads:
        t.start()

    print("\n[main] running. Type 'sleep' to consolidate now, 'quit' to exit.\n")

    try:
        # Run vision on the main thread so macOS Cocoa OpenCV window works natively!
        run_vision(memory, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        print("[main] shutting down...")
        stop_event.set()


if __name__ == "__main__":
    main()
