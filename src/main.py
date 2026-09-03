
import os
import random
import signal
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.config import SilenceStderrFD

with SilenceStderrFD():
    from sentence_transformers import SentenceTransformer
    from src import config
    from src.audio.pipeline import run_audio
    from src.cognition.engine import create_engine
    from src.cognition.interaction import run_interaction_response
    from src.cognition.think import think_immediately, think_quietly
    from src.memory.consolidation import consolidate, run_idle_watcher
    from src.memory.store import MemoryStore
    from src.memory.working import WorkingMemory
    from src.speech.tts import TTSEngine
    from src.vision.pipeline import run_vision


def cognition_loop(memory, engine, stop_event, tts, store, embedder, speaking_event):
    config.log_debug("[main] Cognition loop started.")
    while not stop_event.is_set():
        try:
            state = memory.consciousness
            if state.prediction_error > 0.7:
                think_immediately(memory, engine, tts, store, embedder, urgency="HIGH", speaking_event=speaking_event)
            elif memory.is_user_speaking() or memory.unhandled_speech(0):
                run_interaction_response(memory, engine, tts, store=store, embedder=embedder)
            else:
                time.sleep(0.5)
                if random.random() < 0.05:
                    think_quietly(memory, engine, store, embedder)
        except Exception as e:
            config.log_debug(f"[main] cognition loop error: {e}")
            time.sleep(0.5)


def main():
    config.apply_cli_args()

    engine_name = f"Groq ({getattr(config, 'GROQ_MODEL', 'gpt-oss-20b')})" if getattr(config, "USE_GROQ", False) else "local llama_cpp"
    config.log_debug(f"[main] loading LLM engine: {engine_name}...")
    engine = create_engine()

    embed_model_path = getattr(config, "EMBED_MODEL_PATH", config.EMBED_MODEL_NAME)
    config.log_debug(f"[main] loading semantic embedding model from {embed_model_path}...")
    embedder = SentenceTransformer(embed_model_path)

    store = MemoryStore()
    memory = WorkingMemory()
    stop_event = threading.Event()
    speaking_event = threading.Event()
    interrupt_event = threading.Event()

    config.log_debug("[main] loading TTS engine...")
    tts = TTSEngine(speaking_event=speaking_event, interrupt_event=interrupt_event)

    consolidation_lock = threading.Lock()

    def do_sleep():
        if consolidation_lock.acquire(blocking=False):
            try:
                consolidate(memory, engine, embedder, store, tts=tts)
            finally:
                consolidation_lock.release()

    def shutdown(signum=None, frame=None):
        if not stop_event.is_set():
            if getattr(config, "DEBUG", False):
                print("\n[main] shutting down Karma cleanly...")
            stop_event.set()
            interrupt_event.set()
            speaking_event.clear()
            time.sleep(0.3)

            try:
                import sounddevice as _sd
                _sd.stop()
            except Exception:
                pass

            try:
                import cv2 as _cv2
                _cv2.destroyAllWindows()
            except Exception:
                pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    def listen_stdin():
        if not sys.stdin or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
            return
        try:
            for line in sys.stdin:
                cmd = line.strip().lower()
                if cmd in ("sleep", "sleep now"):
                    threading.Thread(target=do_sleep, daemon=True, name="manual_sleep").start()
                elif cmd in ("quit", "exit"):
                    shutdown()
                    break
            shutdown()
        except (EOFError, KeyboardInterrupt):
            shutdown()
        except Exception:
            pass

    threads = [
        threading.Thread(target=run_audio, args=(memory, stop_event, speaking_event, interrupt_event),
                         daemon=True, name="audio_streamer"),
        threading.Thread(target=cognition_loop,
                         args=(memory, engine, stop_event, tts, store, embedder, speaking_event),
                         daemon=True, name="cognition"),
        threading.Thread(target=run_idle_watcher, args=(memory, stop_event, do_sleep),
                         daemon=True, name="idle_watcher"),
        threading.Thread(target=listen_stdin, daemon=True, name="stdin_listener"),
    ]

    for t in threads:
        t.start()

    groq_note = f" [Groq: {getattr(config, 'GROQ_MODEL', 'gpt-oss-20b')}]" if getattr(config, "USE_GROQ", False) else ""
    print(f"Karma running{groq_note}. Press Ctrl+D to exit.")


    try:
        run_vision(memory, stop_event, speaking_event)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        shutdown()
        for t in threads:
            t.join(timeout=2.0)


if __name__ == "__main__":
    main()

