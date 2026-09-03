import asyncio
import json
import threading
import time
from typing import Set, Dict, Any, Optional

from aiohttp import web

from src import config
from src.hardware.neck import neck_actuator
from src.state import internal_state
from src.ui.kiosk import kiosk_manager

_clients: Set[web.WebSocketResponse] = set()
_loop: Optional[asyncio.AbstractEventLoop] = None
_server_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_runner: Optional[web.AppRunner] = None


def _get_current_state_payload() -> Dict[str, Any]:
    active_code = internal_state.get_active_code()
    code_text = active_code[0] if active_code else None
    code_lang = active_code[1] if active_code else None

    norm_gaze_x = (internal_state.gaze_x + 1.0) / 2.0
    norm_gaze_y = (internal_state.gaze_y + 1.0) / 2.0

    return {
        "type": "state_update",
        "mood": internal_state.mood,
        "emotion": internal_state.current_emotion or internal_state.mood,
        "energy": round(internal_state.energy, 2),
        "curiosity": round(internal_state.curiosity, 2),
        "speaking": internal_state.is_playing_audio,
        "speech": internal_state.last_karma_speech or "",
        "gaze_x": round(norm_gaze_x, 2),
        "gaze_y": round(norm_gaze_y, 2),
        "code": code_text,
        "code_lang": code_lang,
        "kiosk_view": kiosk_manager.active_view
    }


def _get_kiosk_data_payload() -> Dict[str, Any]:
    return {
        "type": "kiosk_data",
        "student_apps": kiosk_manager.student_apps,
        "achievements": kiosk_manager.achievements,
        "docs": kiosk_manager.indexed_docs
    }


async def _broadcast(payload: Dict[str, Any]) -> None:
    if not _clients:
        return
    msg = json.dumps(payload)
    dead = []
    for ws in list(_clients):
        try:
            if not ws.closed:
                await ws.send_str(msg)
            else:
                dead.append(ws)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


def broadcast_state_threadsafe() -> None:
    if _loop and _loop.is_running():
        payload = _get_current_state_payload()
        asyncio.run_coroutine_threadsafe(_broadcast(payload), _loop)


async def _websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=15.0)
    await ws.prepare(request)
    _clients.add(ws)

    try:
        await ws.send_str(json.dumps(_get_current_state_payload()))
        await ws.send_str(json.dumps(_get_kiosk_data_payload()))

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    action = data.get("action", "")

                    if action == "get_initial_data":
                        await ws.send_str(json.dumps(_get_current_state_payload()))
                        await ws.send_str(json.dumps(_get_kiosk_data_payload()))

                    elif action == "open_view":
                        view = data.get("view", "map")
                        kiosk_manager.open_view(view)
                        neck_actuator.tilt_kiosk_touch()
                        broadcast_state_threadsafe()

                    elif action == "close_kiosk":
                        kiosk_manager.open_view("face")
                        neck_actuator.tilt_face_interaction()
                        broadcast_state_threadsafe()

                    elif action == "switch_view":
                        view = data.get("view", "map")
                        kiosk_manager.open_view(view)

                    elif action == "switch_floor":
                        floor = int(data.get("floor", 0))
                        kiosk_manager.current_floor_idx = floor

                    elif action == "clear_code":
                        internal_state.clear_active_code()
                        broadcast_state_threadsafe()

                    elif action == "tilt_touch":
                        neck_actuator.tilt_kiosk_touch()

                    elif action == "tilt_face":
                        neck_actuator.tilt_face_interaction()

                    elif action == "get_student_apps":
                        kiosk_manager.reload_all_data()
                        await ws.send_str(json.dumps({
                            "type": "kiosk_data",
                            "student_apps": kiosk_manager.student_apps
                        }))

                    elif action == "get_achievements":
                        kiosk_manager.reload_all_data()
                        await ws.send_str(json.dumps({
                            "type": "kiosk_data",
                            "achievements": kiosk_manager.achievements
                        }))

                    elif action == "get_docs":
                        kiosk_manager.reload_documents()
                        await ws.send_str(json.dumps({
                            "type": "kiosk_data",
                            "docs": kiosk_manager.indexed_docs
                        }))

                    elif action == "get_doc_chunks":
                        source = data.get("source", "")
                        if source:
                            kiosk_manager.select_document(source)
                            await ws.send_str(json.dumps({
                                "type": "doc_chunks",
                                "source": source,
                                "chunks": kiosk_manager.doc_chunks
                            }))

                except Exception as e:
                    config.log_debug(f"[ui-server] message handling error: {e}")

    finally:
        _clients.discard(ws)

    return ws


async def _state_broadcaster():
    last_state = ""
    while not _stop_event.is_set():
        try:
            if _clients:
                payload = _get_current_state_payload()
                serialized = json.dumps(payload, sort_keys=True)
                if serialized != last_state:
                    last_state = serialized
                    await _broadcast(payload)
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            break
        except Exception as e:
            config.log_debug(f"[ui-server] broadcast error: {e}")
            await asyncio.sleep(0.5)


def _run_server(host: str, port: int):
    global _loop, _runner
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    app = web.Application()
    app.router.add_get("/", _websocket_handler)
    app.router.add_get("/ws", _websocket_handler)

    _runner = web.AppRunner(app)
    _loop.run_until_complete(_runner.setup())
    site = web.TCPSite(_runner, host, port)
    _loop.run_until_complete(site.start())

    config.log_debug(f"[ui-server] WebSocket server listening on ws://{host}:{port}")

    broadcaster_task = _loop.create_task(_state_broadcaster())

    try:
        _loop.run_forever()
    finally:
        broadcaster_task.cancel()
        _loop.run_until_complete(_runner.cleanup())
        _loop.close()


def start_ui_server(host: str = "127.0.0.1", port: int = 8765) -> bool:
    global _server_thread, _stop_event
    if _server_thread is not None and _server_thread.is_alive():
        return True

    _stop_event.clear()
    _server_thread = threading.Thread(
        target=_run_server,
        args=(host, port),
        daemon=True,
        name="KarmaUIServer"
    )
    _server_thread.start()
    time.sleep(0.2)
    return True


def stop_ui_server() -> None:
    global _server_thread, _loop, _stop_event
    _stop_event.set()
    if _loop and _loop.is_running():
        _loop.call_soon_threadsafe(_loop.stop)
    if _server_thread and _server_thread.is_alive():
        _server_thread.join(timeout=1.5)
    _server_thread = None
