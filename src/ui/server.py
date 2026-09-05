import asyncio
import base64
import functools
import hmac
import json
import os
import re
import threading
import time
from typing import Set, Dict, Any, Optional

from aiohttp import web

from src import config
from src.hardware.neck import neck_actuator
from src.state import internal_state
from src.ui import events
from src.ui import telemetry
from src.ui.kiosk import kiosk_manager

_clients: Set[web.WebSocketResponse] = set()
_loop: Optional[asyncio.AbstractEventLoop] = None
_server_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_runner: Optional[web.AppRunner] = None

_net_cache: Optional[Dict[str, Any]] = None
_net_cache_ts: float = 0.0


def _cached_net() -> Dict[str, Any]:
    global _net_cache, _net_cache_ts
    now = time.time()
    if _net_cache is None or now - _net_cache_ts > 30:
        try:
            _net_cache = telemetry.net_info(int(getattr(config, "UI_DASH_PORT", 8080)))
        except Exception:
            _net_cache = {"hostname": "", "ips": [], "dash_port": 8080, "dash_url": ""}
        _net_cache_ts = now
    return _net_cache


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
        "kiosk_view": kiosk_manager.active_view,
        "telemetry": telemetry.snapshot(),
        "net": _cached_net(),
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
    telemetry.set_ws_clients(len(_clients) + len(_dash_live_clients))


def broadcast_state_threadsafe() -> None:
    if _loop and _loop.is_running():
        payload = _get_current_state_payload()
        asyncio.run_coroutine_threadsafe(_broadcast(payload), _loop)


async def _websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=15.0)
    await ws.prepare(request)
    _clients.add(ws)
    telemetry.set_ws_clients(len(_clients) + len(_dash_live_clients))

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
                        broadcast_state_threadsafe()

                    elif action == "close_kiosk":
                        kiosk_manager.open_view("face")
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
                        neck_actuator.tilt_to_kiosk()

                    elif action == "tilt_face":
                        neck_actuator.tilt_to_face()

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
        telemetry.set_ws_clients(len(_clients) + len(_dash_live_clients))

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


# ---------------------------------------------------------------------------
# LAN dashboard: password-gated HTTP + live socket for phones/laptops.
# Runs on its own port (default 8080, all interfaces). The Electron
# websocket above stays localhost-only and ungated.
# ---------------------------------------------------------------------------

DASH_COOKIE = "karma_dash"
DASH_LONG_COOKIE = "karma_dash_long"
SESSION_HOURS = 12
REMEMBER_DAYS = 30
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
UPLOAD_EXTS = (".pdf", ".md", ".txt", ".markdown")

_dash_loop: Optional[asyncio.AbstractEventLoop] = None
_dash_thread: Optional[threading.Thread] = None
_dash_stop = threading.Event()
_dash_runner: Optional[web.AppRunner] = None
_dash_live_clients: Set[web.WebSocketResponse] = set()

_sessions: Dict[str, float] = {}
_sessions_lock = threading.Lock()
_login_fails: Dict[str, list] = {}

# Long-lived device tokens: only SHA-256 hashes touch disk (0600 file),
# so a stolen SD card still doesn't yield a login. Revoked on logout.
LONG_TOKEN_FILE = os.path.join(config.BASE_DIR, "data", ".dashboard_tokens")
_long_tokens: Dict[str, float] = {}
_long_lock = threading.Lock()


def _hash_token(token: str) -> str:
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()


def _load_long_tokens() -> None:
    try:
        with open(LONG_TOKEN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        now = time.time()
        with _long_lock:
            _long_tokens.clear()
            for digest, exp in data.items():
                if isinstance(exp, (int, float)) and exp > now:
                    _long_tokens[digest] = exp
    except Exception:
        pass


def _save_long_tokens() -> None:
    try:
        os.makedirs(os.path.dirname(LONG_TOKEN_FILE), exist_ok=True)
        with _long_lock:
            data = dict(_long_tokens)
        with open(LONG_TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.chmod(LONG_TOKEN_FILE, 0o600)
    except Exception as e:
        config.log_debug(f"[dash] could not persist device tokens: {e}")


def _mint_long_token() -> str:
    import secrets
    token = secrets.token_urlsafe(32)
    with _long_lock:
        _long_tokens[_hash_token(token)] = time.time() + REMEMBER_DAYS * 86400
    _save_long_tokens()
    return token


def _check_long_token(token: str) -> bool:
    if not token:
        return False
    digest = _hash_token(token)
    with _long_lock:
        exp = _long_tokens.get(digest, 0)
        if exp > time.time():
            return True
        _long_tokens.pop(digest, None)
    return False


def _revoke_long_token(token: str) -> None:
    if not token:
        return
    with _long_lock:
        _long_tokens.pop(_hash_token(token), None)
    _save_long_tokens()


_load_long_tokens()

_dash_password = ""
_dash_password_generated = False

_runtime: Dict[str, Any] = {"store": None, "embedder": None}


def set_runtime(store: Any = None, embedder: Any = None) -> None:
    _runtime["store"] = store
    _runtime["embedder"] = embedder


def _get_store():
    if _runtime.get("store") is not None:
        return _runtime["store"]
    from src.memory.store import MemoryStore
    return MemoryStore()


def _get_embedder():
    if _runtime.get("embedder") is not None:
        return _runtime["embedder"]
    return None


def dashboard_password(generated_ok: bool = True) -> str:
    """Resolve the dashboard password (env wins, then saved file, then generate)."""
    global _dash_password, _dash_password_generated
    if _dash_password:
        return _dash_password
    from src import config as _cfg
    env_pw = (_cfg.KARMA_UI_PASSWORD or "").strip()
    if env_pw:
        _dash_password = env_pw
        return _dash_password
    pass_file = os.path.join(_cfg.BASE_DIR, "data", ".dashboard_pass")
    try:
        with open(pass_file, encoding="utf-8") as f:
            saved = f.read().strip()
        if saved:
            _dash_password = saved
            return _dash_password
    except Exception:
        pass
    if not generated_ok:
        return ""
    import secrets
    _dash_password = secrets.token_urlsafe(12)
    _dash_password_generated = True
    try:
        os.makedirs(os.path.dirname(pass_file), exist_ok=True)
        with open(pass_file, "w", encoding="utf-8") as f:
            f.write(_dash_password)
        os.chmod(pass_file, 0o600)
    except Exception as e:
        config.log_debug(f"[dash] could not persist password: {e}")
    return _dash_password


def dashboard_password_generated() -> bool:
    return _dash_password_generated


def _client_ip(request: web.Request) -> str:
    return request.remote or "unknown"


def _login_blocked(ip: str) -> bool:
    fails = _login_fails.get(ip, [])
    now = time.time()
    fails = [t for t in fails if now - t < 120]
    _login_fails[ip] = fails
    return len(fails) >= 5


def _record_login_fail(ip: str) -> None:
    _login_fails.setdefault(ip, []).append(time.time())


def _check_auth(request: web.Request) -> bool:
    token = request.cookies.get(DASH_COOKIE, "")
    if token:
        with _sessions_lock:
            exp = _sessions.get(token, 0)
            if exp > time.time():
                return True
            _sessions.pop(token, None)
    # fall back to a remembered-device token (survives restarts)
    return _check_long_token(request.cookies.get(DASH_LONG_COOKIE, ""))


def require_auth(handler):
    @functools.wraps(handler)
    async def wrapper(request: web.Request):
        if not _check_auth(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)
    return wrapper


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "upload")
    base = re.sub(r"[^A-Za-z0-9._\- ]", "_", base).strip() or "upload"
    return base[:120]


# -- persona override (data/persona.json) -----------------------------------

PERSONA_FILE = os.path.join(config.BASE_DIR, "data", "persona.json")


def _default_persona() -> str:
    from src.cognition.interaction import _BASE_INTERACTION_PROMPT
    return _BASE_INTERACTION_PROMPT


def load_persona_override() -> str:
    override = ""
    try:
        with open(PERSONA_FILE, encoding="utf-8") as f:
            override = (json.load(f).get("system_prompt") or "").strip()
    except Exception:
        pass
    config.PERSONA_OVERRIDE = override
    return override


def save_persona_override(text: str) -> None:
    text = (text or "").strip()
    if text:
        with open(PERSONA_FILE, "w", encoding="utf-8") as f:
            json.dump({"system_prompt": text}, f, ensure_ascii=False, indent=2)
    else:
        try:
            os.remove(PERSONA_FILE)
        except Exception:
            pass
    config.PERSONA_OVERRIDE = text


# -- editable runtime config (data/config_overrides.json) --------------------

# key -> (type, min, max). Only these can change from the dashboard.
EDITABLE_CONFIG: Dict[str, tuple] = {
    "DEFAULT_TEMPERATURE": (float, 0.0, 1.5),
    "DEFAULT_TOP_P": (float, 0.0, 1.0),
    "DEFAULT_REPEAT_PENALTY": (float, 1.0, 2.0),
    "SPEAK_THOUGHTS": (bool, None, None),
    "THINK_INTERVAL_SECONDS": (int, 2, 120),
    "ENABLE_YOLO": (bool, None, None),
    "VAD_SILENCE_TIMEOUT": (float, 0.1, 2.0),
}

CONFIG_OVERRIDES_FILE = os.path.join(config.BASE_DIR, "data", "config_overrides.json")


def _coerce(key: str, value: Any) -> Any:
    kind, lo, hi = EDITABLE_CONFIG[key]
    if kind is bool:
        if isinstance(value, bool):
            v = value
        elif isinstance(value, str):
            v = value.strip().lower() in ("1", "true", "yes", "on")
        else:
            v = bool(value)
        return v
    v = kind(value)
    if lo is not None and v < lo:
        raise ValueError(f"{key} below minimum {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"{key} above maximum {hi}")
    return v


def apply_saved_overrides() -> Dict[str, Any]:
    applied: Dict[str, Any] = {}
    try:
        with open(CONFIG_OVERRIDES_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return applied
    for key, raw in saved.items():
        if key not in EDITABLE_CONFIG:
            continue
        try:
            setattr(config, key, _coerce(key, raw))
            applied[key] = getattr(config, key)
        except Exception as e:
            config.log_debug(f"[dash] bad saved override {key}: {e}")
    return applied


def persist_overrides(values: Dict[str, Any]) -> None:
    with open(CONFIG_OVERRIDES_FILE, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2)


# -- dashboard route handlers -------------------------------------------------

async def _dash_index(request: web.Request) -> web.Response:
    if not _check_auth(request):
        return web.HTTPFound("/login")
    return web.HTTPFound("/dash/")


async def _dash_login_page(request: web.Request) -> web.Response:
    if _check_auth(request):
        return web.HTTPFound("/dash/")
    html = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KARMA · REMOTE</title>
<style>:root{--bg:#000;--panel:#05090d;--line:#152230;--line-2:#1e3a55;--blue:#2e9bff;--ink:#e8f2ff;--mut:#7d93a8;--danger:#ff5d5d;--font-d:"Space Grotesk",system-ui,sans-serif;--font-m:"IBM Plex Mono",ui-monospace,monospace}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font-family:var(--font-d);margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center}
.login-card{background:#020406;border:1px solid var(--blue);border-top:3px solid var(--blue);padding:22px;width:340px;max-width:92vw}
.brand-block{width:28px;height:28px;background:var(--blue);color:#00101f;font-weight:700;line-height:28px;text-align:center;margin-bottom:14px}
h2{font-size:1.1rem;text-transform:uppercase;letter-spacing:.02em}
input{width:100%;margin:12px 0;padding:10px;background:#000;color:var(--ink);border:1px solid var(--line-2);font-family:var(--font-m);font-size:.72rem}
input:focus{outline:none;border-color:var(--blue)}
button{width:100%;padding:10px;background:transparent;color:#9cceff;border:1px solid var(--blue);font-family:var(--font-m);font-size:.62rem;font-weight:600;letter-spacing:.1em;cursor:pointer}
button:hover{background:var(--blue);color:#00101f}
.err{color:var(--danger);font-family:var(--font-m);font-size:.66rem;min-height:18px}
.rule{height:2px;background:var(--blue);margin:0 0 18px}</style></head><body>
<div style="width:340px;max-width:92vw"><div class="rule"></div><div class="login-card">
<div class="brand-block">K</div><h2>Remote console</h2><div class="err" id="e"></div>
<form id="f"><input type="password" id="p" placeholder="PASSWORD" autocomplete="current-password">
<button>UNLOCK</button></form></div></div>
<script>
document.getElementById('f').onsubmit = async (ev) => {
  ev.preventDefault();
  const r = await fetch('/api/login', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({password: document.getElementById('p').value})});
  if (r.ok) location.href = '/dash/';
  else document.getElementById('e').textContent = 'Wrong password';
};
</script></body></html>"""
    return web.Response(text=html, content_type="text/html")


async def _api_login(request: web.Request) -> web.Response:
    ip = _client_ip(request)
    if _login_blocked(ip):
        return web.json_response({"error": "too many attempts, wait a minute"}, status=429)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "bad request"}, status=400)
    guess = str(body.get("password", ""))
    if not guess or not hmac.compare_digest(guess, dashboard_password()):
        _record_login_fail(ip)
        return web.json_response({"error": "wrong password"}, status=401)
    import secrets
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        _sessions[token] = time.time() + SESSION_HOURS * 3600
    resp = web.json_response({"ok": True})
    resp.set_cookie(DASH_COOKIE, token, max_age=SESSION_HOURS * 3600,
                    httponly=True, samesite="Lax", path="/")
    if body.get("remember") is True:
        long_token = _mint_long_token()
        resp.set_cookie(DASH_LONG_COOKIE, long_token, max_age=REMEMBER_DAYS * 86400,
                        httponly=True, samesite="Lax", path="/")
    events.post("system", f"dashboard login from {ip}")
    return resp


@require_auth
async def _api_logout(request: web.Request) -> web.Response:
    token = request.cookies.get(DASH_COOKIE, "")
    with _sessions_lock:
        _sessions.pop(token, None)
    _revoke_long_token(request.cookies.get(DASH_LONG_COOKIE, ""))
    resp = web.json_response({"ok": True})
    resp.del_cookie(DASH_COOKIE, path="/")
    resp.del_cookie(DASH_LONG_COOKIE, path="/")
    return resp


@require_auth
async def _api_state(request: web.Request) -> web.Response:
    return web.json_response(_get_current_state_payload())


@require_auth
async def _api_telemetry(request: web.Request) -> web.Response:
    return web.json_response({"telemetry": telemetry.snapshot(), "net": _cached_net()})


@require_auth
async def _api_logs(request: web.Request) -> web.Response:
    kind = request.query.get("kind") or None
    try:
        limit = min(500, max(1, int(request.query.get("limit", "150"))))
    except ValueError:
        limit = 150
    if kind and kind not in ("thought", "reply", "heard", "kiosk", "system", "error"):
        return web.json_response({"error": "bad kind"}, status=400)
    return web.json_response({"events": events.recent(kind=kind, limit=limit)})


@require_auth
async def _api_thoughts(request: web.Request) -> web.Response:
    try:
        limit = min(200, max(1, int(request.query.get("limit", "50"))))
    except ValueError:
        limit = 50
    return web.json_response({"thoughts": events.recent(kind="thought", limit=limit)})


@require_auth
async def _api_camera_jpg(request: web.Request) -> web.Response:
    frame = internal_state.get_camera_frame()
    if not frame:
        return web.json_response({"error": "no camera frame yet"}, status=503)
    return web.Response(body=frame, content_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})


@require_auth
async def _api_camera_mjpeg(request: web.Request) -> web.Response:
    boundary = "karmaframe"
    resp = web.StreamResponse(headers={
        "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })
    await resp.prepare(request)
    last = None
    try:
        while True:
            frame = internal_state.get_camera_frame()
            if frame and frame != last:
                last = frame
                await resp.write(
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n".encode() + frame + b"\r\n"
                )
            await asyncio.sleep(0.12)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as e:
        config.log_debug(f"[dash] mjpeg error: {e}")
    return resp


def _rag_store():
    return _get_store()


@require_auth
async def _api_rag_docs(request: web.Request) -> web.Response:
    try:
        docs = _rag_store().list_sources(kind="document")
        return web.json_response({"docs": docs})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@require_auth
async def _api_rag_query(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "bad request"}, status=400)
    q = str(body.get("q", "")).strip()
    if not q:
        return web.json_response({"error": "empty query"}, status=400)
    try:
        k = min(10, max(1, int(body.get("k", 3))))
    except (ValueError, TypeError):
        k = 3
    try:
        from src.memory.rag import DocumentRAG
        rag = DocumentRAG(store=_rag_store(), embedder=_get_embedder())
        loop = asyncio.get_event_loop()
        ctx = await loop.run_in_executor(None, rag.get_rag_context, q, k)
        return web.json_response({"context": ctx})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@require_auth
async def _api_rag_upload(request: web.Request) -> web.Response:
    try:
        form = await request.post()
    except Exception:
        return web.json_response({"error": "bad upload"}, status=400)
    field = form.get("file")
    if field is None or not hasattr(field, "file"):
        return web.json_response({"error": "no file part"}, status=400)
    name = _safe_filename(getattr(field, "filename", "upload"))
    if not name.lower().endswith(UPLOAD_EXTS):
        return web.json_response({"error": f"only {', '.join(UPLOAD_EXTS)} allowed"}, status=400)
    try:
        content = field.file.read(MAX_UPLOAD_BYTES + 1)
    except Exception:
        return web.json_response({"error": "could not read file"}, status=400)
    if len(content) > MAX_UPLOAD_BYTES:
        return web.json_response({"error": "file too large (32 MB max)"}, status=400)
    if not content:
        return web.json_response({"error": "empty file"}, status=400)

    upload_dir = os.path.join(config.BASE_DIR, "data", "documents", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, name)
    base, ext = os.path.splitext(name)
    i = 1
    while os.path.exists(dest):
        dest = os.path.join(upload_dir, f"{base}_{i}{ext}")
        i += 1
    with open(dest, "wb") as f:
        f.write(content)

    def _ingest():
        from src.memory.rag import DocumentRAG
        rag = DocumentRAG(store=_rag_store(), embedder=_get_embedder())
        return rag.ingest_pdf(dest, verbose=False)

    try:
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, _ingest)
    except Exception as e:
        try:
            os.remove(dest)
        except Exception:
            pass
        return web.json_response({"error": f"ingest failed: {e}"}, status=500)
    events.post("system", f"ingested {os.path.basename(dest)} ({chunks} chunks)")
    return web.json_response({"ok": True, "source": os.path.basename(dest), "chunks": chunks})


@require_auth
async def _api_rag_delete(request: web.Request) -> web.Response:
    source = (request.query.get("source") or "").strip()
    if not source or "/" in source or ".." in source:
        return web.json_response({"error": "bad source"}, status=400)
    try:
        removed = _rag_store().delete_by_source(source)
        events.post("system", f"removed {removed} chunks of {source}")
        return web.json_response({"ok": True, "removed": removed})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@require_auth
async def _api_prompt_get(request: web.Request) -> web.Response:
    default = _default_persona()
    override = getattr(config, "PERSONA_OVERRIDE", "") or ""
    return web.json_response({
        "default": default,
        "override": override,
        "active": override or default,
    })


@require_auth
async def _api_prompt_put(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "bad request"}, status=400)
    text = body.get("text")
    if text is not None and not isinstance(text, str):
        return web.json_response({"error": "text must be a string or null"}, status=400)
    if text and len(text) > 4000:
        return web.json_response({"error": "prompt too long (4000 chars max)"}, status=400)
    try:
        save_persona_override(text or "")
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    active = (text or "").strip() or _default_persona()
    events.post("system", "system prompt updated" if text else "system prompt reset to default")
    return web.json_response({"ok": True, "active": active})


@require_auth
async def _api_config_get(request: web.Request) -> web.Response:
    return web.json_response({
        key: getattr(config, key, None) for key in EDITABLE_CONFIG
    })


@require_auth
async def _api_config_put(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "bad request"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "expected an object"}, status=400)
    updated: Dict[str, Any] = {}
    for key, raw in body.items():
        if key not in EDITABLE_CONFIG:
            return web.json_response({"error": f"not editable: {key}"}, status=400)
        try:
            updated[key] = _coerce(key, raw)
        except (ValueError, TypeError) as e:
            return web.json_response({"error": str(e)}, status=400)
    for key, value in updated.items():
        setattr(config, key, value)
    try:
        current: Dict[str, Any] = {}
        try:
            with open(CONFIG_OVERRIDES_FILE, encoding="utf-8") as f:
                current = json.load(f)
        except Exception:
            pass
        current.update(updated)
        persist_overrides(current)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    events.post("system", f"settings updated: {', '.join(sorted(updated))}")
    return web.json_response({"ok": True, "updated": updated})


@require_auth
async def _dash_page(request: web.Request) -> web.Response:
    page = os.path.join(config.BASE_DIR, "ui", "dashboard.html")
    try:
        with open(page, encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")
    except Exception:
        return web.Response(text="dashboard.html missing", status=500)


async def _dash_live(request: web.Request) -> web.WebSocketResponse:
    if not _check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    ws = web.WebSocketResponse(heartbeat=15.0)
    await ws.prepare(request)
    _dash_live_clients.add(ws)
    telemetry.set_ws_clients(len(_clients) + len(_dash_live_clients))
    last_event_ts = time.time()
    try:
        await ws.send_str(json.dumps(_get_current_state_payload()))
        while True:
            await asyncio.sleep(1.0)
            await ws.send_str(json.dumps(_get_current_state_payload()))
            fresh = [e for e in events.recent(limit=50) if e["ts"] > last_event_ts]
            if fresh:
                last_event_ts = max(e["ts"] for e in fresh)
                await ws.send_str(json.dumps({"type": "events", "events": list(reversed(fresh))}))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as e:
        config.log_debug(f"[dash] live socket error: {e}")
    finally:
        _dash_live_clients.discard(ws)
        telemetry.set_ws_clients(len(_clients) + len(_dash_live_clients))
    return ws


# -- interactive shell (dashboard Shell tab) ------------------------------------

_shell_active: Set[int] = set()
_shell_lock = threading.Lock()
_SHELL_INPUT_MAX = 65536


async def _dash_shell(request: web.Request) -> web.WebSocketResponse:
    """PTY shell over websocket. Same auth as the rest of the dashboard.

    Client -> server: {"t":"i","d":"keystrokes"} | {"t":"r","rows":n,"cols":n}
    Server -> client: {"t":"o","d":base64} | {"t":"x","code":n,"reason":"..."}
    """
    if not _check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    if not getattr(config, "SHELL_ENABLED", True):
        return web.json_response({"error": "shell disabled (SHELL_ENABLED=0)"}, status=403)
    with _shell_lock:
        if len(_shell_active) >= max(1, int(getattr(config, "SHELL_MAX_SESSIONS", 3))):
            return web.json_response({"error": "too many shell sessions"}, status=429)

    from src.ui.shell import ShellSession
    ip = _client_ip(request)
    ws = web.WebSocketResponse(heartbeat=15.0)
    await ws.prepare(request)
    loop = asyncio.get_event_loop()
    out_q: asyncio.Queue = asyncio.Queue()
    idle = max(60, int(getattr(config, "SHELL_IDLE_SECONDS", 900)))

    def _on_output(chunk):
        try:
            loop.call_soon_threadsafe(out_q.put_nowait, chunk)
        except Exception:
            pass

    try:
        session = ShellSession(_on_output)
    except Exception as e:
        await ws.send_str(json.dumps({"t": "x", "code": None, "reason": f"spawn failed: {e}"}))
        await ws.close()
        return ws
    with _shell_lock:
        _shell_active.add(id(session))
    events.post("system", f"shell opened from {ip}")

    async def _sender():
        while True:
            try:
                chunk = await asyncio.wait_for(out_q.get(), timeout=idle)
            except asyncio.TimeoutError:
                try:
                    await ws.send_str(json.dumps({"t": "x", "code": None,
                                                  "reason": f"idle timeout ({idle}s)"}))
                except Exception:
                    pass
                break
            if chunk is None:
                try:
                    await ws.send_str(json.dumps({"t": "x", "code": session.exit_code(),
                                                  "reason": "shell exited"}))
                except Exception:
                    pass
                break
            try:
                await ws.send_str(json.dumps(
                    {"t": "o", "d": base64.b64encode(chunk).decode("ascii")}))
            except Exception:
                break

    sender = asyncio.ensure_future(_sender())
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                kind = data.get("t")
                if kind == "i":
                    text = str(data.get("d", ""))[:_SHELL_INPUT_MAX]
                    if text and not session.write(text):
                        break
                elif kind == "r":
                    try:
                        rows = min(100, max(5, int(data.get("rows", 30))))
                        cols = min(300, max(20, int(data.get("cols", 100))))
                    except (TypeError, ValueError):
                        continue
                    session.resize(rows, cols)
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING,
                              web.WSMsgType.ERROR):
                break
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as e:
        config.log_debug(f"[dash] shell socket error: {e}")
    finally:
        sender.cancel()
        session.close()
        with _shell_lock:
            _shell_active.discard(id(session))
        events.post("system", f"shell closed from {ip}")
        try:
            await ws.close()
        except Exception:
            pass
    return ws


def _build_dash_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", _dash_index)
    app.router.add_get("/login", _dash_login_page)
    app.router.add_get("/dash/", _dash_page)
    app.router.add_post("/api/login", _api_login)
    app.router.add_post("/api/logout", _api_logout)
    app.router.add_get("/api/state", _api_state)
    app.router.add_get("/api/telemetry", _api_telemetry)
    app.router.add_get("/api/logs", _api_logs)
    app.router.add_get("/api/thoughts", _api_thoughts)
    app.router.add_get("/api/camera.jpg", _api_camera_jpg)
    app.router.add_get("/api/camera.mjpeg", _api_camera_mjpeg)
    app.router.add_get("/api/rag/docs", _api_rag_docs)
    app.router.add_post("/api/rag/query", _api_rag_query)
    app.router.add_post("/api/rag/upload", _api_rag_upload)
    app.router.add_delete("/api/rag/docs", _api_rag_delete)
    app.router.add_get("/api/prompt", _api_prompt_get)
    app.router.add_put("/api/prompt", _api_prompt_put)
    app.router.add_get("/api/config", _api_config_get)
    app.router.add_put("/api/config", _api_config_put)
    app.router.add_get("/api/live", _dash_live)
    app.router.add_get("/api/shell", _dash_shell)
    return app


def _run_dash(host: str, port: int) -> None:
    global _dash_loop, _dash_runner
    _dash_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_dash_loop)
    app = _build_dash_app()
    _dash_runner = web.AppRunner(app)
    _dash_loop.run_until_complete(_dash_runner.setup())
    site = web.TCPSite(_dash_runner, host, port)
    _dash_loop.run_until_complete(site.start())
    config.log_debug(f"[dash] dashboard on http://{host}:{port}")
    try:
        _dash_loop.run_forever()
    finally:
        _dash_loop.run_until_complete(_dash_runner.cleanup())
        _dash_loop.close()


def start_dashboard_server(host: str = "0.0.0.0", port: int = 8080,
                           password: Optional[str] = None) -> bool:
    """Start the password-gated LAN dashboard. Returns True when serving."""
    global _dash_thread, _dash_password
    if password:
        _dash_password = password
    else:
        dashboard_password()
    if _dash_thread is not None and _dash_thread.is_alive():
        return True
    _dash_stop.clear()
    _dash_thread = threading.Thread(
        target=_run_dash, args=(host, port), daemon=True, name="KarmaDash")
    _dash_thread.start()
    time.sleep(0.3)
    return _dash_thread.is_alive()


def stop_dashboard_server() -> None:
    global _dash_thread, _dash_loop
    _dash_stop.set()
    if _dash_loop and _dash_loop.is_running():
        _dash_loop.call_soon_threadsafe(_dash_loop.stop)
    if _dash_thread and _dash_thread.is_alive():
        _dash_thread.join(timeout=1.5)
    _dash_thread = None
