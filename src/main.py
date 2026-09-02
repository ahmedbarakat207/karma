"""
Karma Entry Point & Orchestration Supervisor.
Initializes perception subsystems (vision + audio), cognition loops, and sleep management.
Runs vision on the main thread for macOS Cocoa OpenCV UI compatibility.
"""
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

class SilenceStderrFD:
    """Temporarily silences C-level file descriptor 2 (stderr) to suppress Objective-C duplicate symbol warnings."""
    def __enter__(self):
        try:
            sys.stderr.flush()
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.saved_stderr_fd = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.null_fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, "null_fd", None) is not None:
            try:
                sys.stderr.flush()
                os.dup2(self.saved_stderr_fd, 2)
                os.close(self.saved_stderr_fd)
                os.close(self.null_fd)
            except Exception:
                pass

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


def consciousness_orchestrator(memory, engine, stop_event, tts, store, embedder, speaking_event):
    """Coordinates cognition, user interaction turns, and spontaneous idle reflection."""
    config.log_debug("[main] Consciousness orchestrator started.")
    while not stop_event.is_set():
        try:
            state = memory.consciousness

            # 1. Urgent environmental event -> spontaneous high-salience thought
            if state.prediction_error > 0.7:
                think_immediately(memory, engine, tts, store, embedder, urgency="HIGH", speaking_event=speaking_event)

            # 2. Spoken user input -> conversational reply
            elif memory.is_user_speaking() or memory.unhandled_speech(0):
                run_interaction_response(memory, engine, tts, store=store, embedder=embedder)

            # 3. Ambient idle reflection
            else:
                time.sleep(0.5)
                if random.random() < 0.05:
                    think_quietly(memory, engine, store, embedder)
        except Exception as e:
            config.log_debug(f"[main] consciousness loop exception: {e}")
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
        """Clean shutdown handler: notifies threads and frees audio/vision hardware."""
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
            # Trigger clean shutdown when EOF (Ctrl+D) is received
            shutdown()
        except (EOFError, KeyboardInterrupt):
            shutdown()
        except Exception:
            pass

    threads = [
        threading.Thread(target=run_audio, args=(memory, stop_event, speaking_event, interrupt_event),
                         daemon=True, name="audio_streamer"),
        threading.Thread(target=consciousness_orchestrator,
                         args=(memory, engine, stop_event, tts, store, embedder, speaking_event),
                         daemon=True, name="consciousness_orchestrator"),
        threading.Thread(target=run_idle_watcher, args=(memory, stop_event, do_sleep),
                         daemon=True, name="idle_watcher"),
        threading.Thread(target=listen_stdin, daemon=True, name="stdin_listener"),
    ]

    for t in threads:
        t.start()

    groq_note = f" [Groq: {getattr(config, 'GROQ_MODEL', 'gpt-oss-20b')}]" if getattr(config, "USE_GROQ", False) else ""
    if getattr(config, "DEBUG", False):
        print(f"\n[main] Karma is awake and running (DEBUG MODE{groq_note}: Full logs + Camera window enabled). Press Ctrl+D to exit.\n")
    else:
        print(f"\n✨ Karma is awake and running{groq_note}. Fullscreen face active. Press Ctrl+D to exit.\n")


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

