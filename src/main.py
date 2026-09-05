
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
    from src.hardware.drive import drive_base
    from src.memory.consolidation import consolidate, run_idle_watcher
    from src.memory.store import MemoryStore
    from src.memory.working import WorkingMemory
    from src.navigation.explorer import run_explorer
    from src.speech.tts import TTSEngine
    from src.state import internal_state
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

    from src.ui.server import apply_saved_overrides, load_persona_override
    applied = apply_saved_overrides()
    if applied:
        print(f"[main] restored settings: {', '.join(sorted(applied))}")
    if load_persona_override():
        print("[main] custom system prompt loaded from data/persona.json")

    stop_event = threading.Event()
    speaking_event = threading.Event()
    interrupt_event = threading.Event()
    consolidation_lock = threading.Lock()

    electron_proc = None
    if getattr(config, "USE_ELECTRON", True):
        try:
            from src.ui.server import start_ui_server
            start_ui_server(
                host=getattr(config, "UI_WS_HOST", "127.0.0.1"),
                port=getattr(config, "UI_WS_PORT", 8765)
            )
            ui_dir = os.path.join(config.BASE_DIR, "ui")
            if os.path.isdir(ui_dir):
                import shutil
                import subprocess
                electron_bin = shutil.which("electron")
                local_node_bin = os.path.join(ui_dir, "node_modules", ".bin", "electron")
                cmd = None
                if os.path.exists(local_node_bin):
                    cmd = [local_node_bin, ".", "--no-sandbox"]
                elif electron_bin:
                    cmd = [electron_bin, ".", "--no-sandbox"]
                elif shutil.which("chromium-browser"):
                    index_path = os.path.abspath(os.path.join(ui_dir, "index.html"))
                    cmd = ["chromium-browser", "--kiosk", "--noerrdialogs", "--disable-infobars", "--no-first-run", "--no-sandbox", f"file://{index_path}"]
                elif shutil.which("chromium"):
                    index_path = os.path.abspath(os.path.join(ui_dir, "index.html"))
                    cmd = ["chromium", "--kiosk", "--noerrdialogs", "--disable-infobars", "--no-first-run", "--no-sandbox", f"file://{index_path}"]
                elif shutil.which("npx"):
                    cmd = ["npx", "electron", ".", "--no-sandbox"]

                if cmd:
                    electron_proc = subprocess.Popen(
                        cmd,
                        cwd=ui_dir,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
        except Exception as e:
            config.log_debug(f"[main] UI server / electron init note: {e}")

    engine = None
    try:
        engine_name = f"Groq ({getattr(config, 'GROQ_MODEL', 'gpt-oss-20b')})" if getattr(config, "USE_GROQ", False) else "local llama_cpp"
        config.log_debug(f"[main] loading LLM engine: {engine_name}...")
        engine = create_engine()
    except Exception as e:
        config.log_debug(f"[main] engine load note: {e}")

    embedder = None
    try:
        embed_model_path = getattr(config, "EMBED_MODEL_PATH", config.EMBED_MODEL_NAME)
        config.log_debug(f"[main] loading semantic embedding model from {embed_model_path}...")
        embedder = SentenceTransformer(embed_model_path)
    except Exception as e:
        config.log_debug(f"[main] embedder load note: {e}")

    store = MemoryStore()
    memory = WorkingMemory()

    try:
        from src.ui.server import set_runtime
        set_runtime(store, embedder)
    except Exception as e:
        config.log_debug(f"[main] dashboard runtime note: {e}")

    try:
        from src.ui.server import (
            dashboard_password,
            dashboard_password_generated,
            start_dashboard_server,
        )
        from src.ui import telemetry as _telemetry
        dash_host = getattr(config, "UI_DASH_HOST", "0.0.0.0")
        dash_port = int(getattr(config, "UI_DASH_PORT", 8080))
        if start_dashboard_server(host=dash_host, port=dash_port):
            net = _telemetry.net_info(dash_port)
            for ip in net["ips"]:
                print(f"[dash] dashboard: http://{ip}:{dash_port}")
            if not net["ips"]:
                print(f"[dash] dashboard on port {dash_port} (no LAN ip found yet)")
            if dashboard_password_generated():
                print(f"[dash] generated password: {dashboard_password()}")
                print("[dash] set KARMA_UI_PASSWORD to use your own")
            from src.ui import events as _events
            _events.post("system", f"dashboard up on port {dash_port}")
    except Exception as e:
        config.log_debug(f"[main] dashboard init note: {e}")

    tts = None
    try:
        config.log_debug("[main] loading TTS engine...")
        tts = TTSEngine(speaking_event=speaking_event, interrupt_event=interrupt_event)
    except Exception as e:
        config.log_debug(f"[main] TTS load note: {e}")

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
                drive_base.stop()
                drive_base.cleanup()
            except Exception:
                pass

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

            if electron_proc:
                try:
                    electron_proc.terminate()
                except Exception:
                    pass

            try:
                from src.ui.server import stop_ui_server, stop_dashboard_server
                stop_ui_server()
                stop_dashboard_server()
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
                elif cmd in ("stop", "halt"):
                    try:
                        from src.navigation.explorer import explorer
                        explorer.stop()
                    except Exception:
                        pass
                elif cmd in ("explore", "wander"):
                    try:
                        from src.navigation.explorer import explorer
                        explorer.start_explore()
                    except Exception:
                        pass
                elif cmd in ("forward", "back", "left", "right"):
                    try:
                        fn = {"forward": drive_base.forward, "back": drive_base.backward,
                              "left": drive_base.turn_left, "right": drive_base.turn_right}[cmd]
                        threading.Thread(target=fn, kwargs={"duration": 1.0},
                                         daemon=True, name="manual_drive").start()
                    except Exception:
                        pass
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
        threading.Thread(target=run_explorer, args=(memory, stop_event, drive_base, internal_state),
                         daemon=True, name="explorer"),
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
    except Exception as e:
        config.log_debug(f"[main] vision loop caught: {e}")
        while not stop_event.is_set():
            time.sleep(1.0)
    finally:
        shutdown()
        for t in threads:
            t.join(timeout=2.0)


if __name__ == "__main__":
    main()

