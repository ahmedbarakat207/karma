import asyncio
import base64
import json
import pytest
import aiohttp

from src import config
from src.state import internal_state
from src.ui import events
from src.ui import server as dash


def make_session():
    return aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True))


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
async def dash_server(tmp_path):
    port = 8897
    started = dash.start_dashboard_server(host="127.0.0.1", port=port, password="test-pass-123")
    assert started is True
    await asyncio.sleep(0.4)
    yield f"http://127.0.0.1:{port}"
    dash.stop_dashboard_server()
    await asyncio.sleep(0.2)


@pytest.fixture
async def authed(dash_server):
    async with make_session() as session:
        async with session.post(dash_server + "/api/login",
                                json={"password": "test-pass-123"}) as r:
            assert r.status == 200
            cookies = session.cookie_jar.filter_cookies(dash_server)
            assert "karma_dash" in cookies
        yield session, dash_server


@pytest.mark.anyio
async def test_auth_gate_and_login(dash_server):
    async with make_session() as session:
        async with session.get(dash_server + "/api/state") as r:
            assert r.status == 401
        async with session.post(dash_server + "/api/login",
                                json={"password": "wrong"}) as r:
            assert r.status == 401
        async with session.post(dash_server + "/api/login",
                                json={"password": "test-pass-123"}) as r:
            assert r.status == 200
        async with session.get(dash_server + "/api/state") as r:
            assert r.status == 200
            body = await r.json()
            assert "mood" in body and "telemetry" in body and "net" in body


@pytest.mark.anyio
async def test_telemetry_shape(authed):
    session, base = authed
    async with session.get(base + "/api/telemetry") as r:
        assert r.status == 200
        body = await r.json()
        assert "telemetry" in body and "net" in body
        t = body["telemetry"]
        for key in ("cpu_temp_c", "mem_used_pct", "disk_used_pct",
                    "uptime_s", "llm", "ws_clients"):
            assert key in t, key
        assert "calls" in t["llm"]


@pytest.mark.anyio
async def test_logs_and_thoughts(authed):
    session, base = authed
    events.post("thought", "test thought here")
    events.post("reply", "test reply here")
    async with session.get(base + "/api/logs?limit=10") as r:
        assert r.status == 200
        kinds = {e["kind"] for e in (await r.json())["events"]}
        assert {"thought", "reply"} <= kinds
    async with session.get(base + "/api/logs?kind=thought&limit=5") as r:
        assert r.status == 200
        got = (await r.json())["events"]
        assert got and all(e["kind"] == "thought" for e in got)
    async with session.get(base + "/api/logs?kind=bogus") as r:
        assert r.status == 400
    async with session.get(base + "/api/thoughts?limit=5") as r:
        assert r.status == 200
        assert any("test thought here" in e["text"] for e in (await r.json())["thoughts"])
    events.clear()


@pytest.mark.anyio
async def test_camera_endpoints(authed):
    session, base = authed
    internal_state.set_camera_frame(None)
    async with session.get(base + "/api/camera.jpg") as r:
        assert r.status == 503
    internal_state.set_camera_frame(b"\xff\xd8fakejpeg\xff\xd9")
    try:
        async with session.get(base + "/api/camera.jpg") as r:
            assert r.status == 200
            assert r.headers.get("Content-Type") == "image/jpeg"
            assert await r.read() == b"\xff\xd8fakejpeg\xff\xd9"
    finally:
        internal_state.set_camera_frame(None)


@pytest.mark.anyio
async def test_prompt_get_set_reset(authed):
    session, base = authed
    async with session.get(base + "/api/prompt") as r:
        assert r.status == 200
        body = await r.json()
        assert body["default"] and body["active"] == (body["override"] or body["default"])
    async with session.put(base + "/api/prompt", json={"text": "x" * 4001}) as r:
        assert r.status == 400
    async with session.put(base + "/api/prompt", json={"text": 123}) as r:
        assert r.status == 400
    async with session.put(base + "/api/prompt", json={"text": "You are TestBot."}) as r:
        assert r.status == 200
        assert (await r.json())["active"] == "You are TestBot."
        assert config.PERSONA_OVERRIDE == "You are TestBot."
    async with session.put(base + "/api/prompt", json={"text": None}) as r:
        assert r.status == 200
    assert (getattr(config, "PERSONA_OVERRIDE", "") or "") == ""
    try:
        import os
        os.remove("data/persona.json")
    except Exception:
        pass


@pytest.mark.anyio
async def test_config_get_set_validation(authed):
    session, base = authed
    old_temp = config.DEFAULT_TEMPERATURE
    try:
        async with session.get(base + "/api/config") as r:
            assert r.status == 200
            assert "DEFAULT_TEMPERATURE" in await r.json()
        async with session.put(base + "/api/config", json={"NOPE": 1}) as r:
            assert r.status == 400
        async with session.put(base + "/api/config", json={"DEFAULT_TEMPERATURE": 9}) as r:
            assert r.status == 400
        async with session.put(base + "/api/config",
                               json={"DEFAULT_TEMPERATURE": 0.5, "SPEAK_THOUGHTS": True}) as r:
            assert r.status == 200
            assert config.DEFAULT_TEMPERATURE == 0.5
            assert config.SPEAK_THOUGHTS is True
    finally:
        config.DEFAULT_TEMPERATURE = old_temp
        config.SPEAK_THOUGHTS = False
        try:
            import os
            os.remove("data/config_overrides.json")
        except Exception:
            pass


@pytest.mark.anyio
async def test_upload_rejects_bad_files(authed):
    session, base = authed
    data = aiohttp.FormData()
    data.add_field("file", b"MZ...", filename="evil.exe", content_type="application/octet-stream")
    async with session.post(base + "/api/rag/upload", data=data) as r:
        assert r.status == 400
    data = aiohttp.FormData()
    data.add_field("file", b"", filename="empty.txt", content_type="text/plain")
    async with session.post(base + "/api/rag/upload", data=data) as r:
        assert r.status == 400


@pytest.mark.anyio
async def test_remember_device_survives_restart_and_logout_revokes(dash_server, tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "LONG_TOKEN_FILE", str(tmp_path / "tokens.json"))
    dash._long_tokens.clear()
    try:
        async with make_session() as session:
            async with session.post(dash_server + "/api/login",
                                    json={"password": "test-pass-123", "remember": True}) as r:
                assert r.status == 200
            jar = session.cookie_jar.filter_cookies(dash_server)
            assert "karma_dash" in jar and "karma_dash_long" in jar
            long_val = jar["karma_dash_long"].value
        # only hashes touch disk, never the raw token
        saved = open(tmp_path / "tokens.json", encoding="utf-8").read()
        assert long_val not in saved

        hdr = {"Cookie": f"karma_dash_long={long_val}"}
        async with make_session() as s2:
            async with s2.get(dash_server + "/api/state", headers=hdr) as r:
                assert r.status == 200
            # simulated restart: in-memory sessions wiped, device token lives on
            dash._sessions.clear()
            async with s2.get(dash_server + "/api/state", headers=hdr) as r:
                assert r.status == 200
            # logout revokes the device token as well
            async with s2.post(dash_server + "/api/logout", headers=hdr) as r:
                assert r.status == 200
            async with s2.get(dash_server + "/api/state", headers=hdr) as r:
                assert r.status == 401
    finally:
        dash._long_tokens.clear()


@pytest.mark.anyio
async def test_plain_login_sets_no_device_cookie(dash_server):
    async with make_session() as session:
        async with session.post(dash_server + "/api/login",
                                json={"password": "test-pass-123"}) as r:
            assert r.status == 200
        jar = session.cookie_jar.filter_cookies(dash_server)
        assert "karma_dash" in jar
        assert "karma_dash_long" not in jar


@pytest.mark.anyio
async def test_live_socket_streams_state(authed):
    session, base = authed
    cookies = session.cookie_jar.filter_cookies(base)
    headers = {"Cookie": f"karma_dash={cookies['karma_dash'].value}"}
    ws_url = base.replace("http://", "ws://") + "/api/live"
    async with session.ws_connect(ws_url, headers=headers) as ws:
        msg = await asyncio.wait_for(ws.receive_json(), timeout=3.0)
        assert msg["type"] == "state_update"
        assert "telemetry" in msg


@pytest.mark.anyio
async def test_upload_txt_ingests(tmp_path, authed, monkeypatch):
    from src.memory.store import MemoryStore
    from sentence_transformers import SentenceTransformer
    session, base = authed
    store = MemoryStore(db_path=str(tmp_path / "t.db"))
    embedder = SentenceTransformer(config.EMBED_MODEL_PATH)
    monkeypatch.setitem(dash._runtime, "store", store)
    monkeypatch.setitem(dash._runtime, "embedder", embedder)
    data = aiohttp.FormData()
    data.add_field("file", b"# Test Doc\n\nKarma test knowledge about wombats.",
                   filename="dash_test_doc.txt", content_type="text/plain")
    async with session.post(base + "/api/rag/upload", data=data) as r:
        assert r.status == 200
        body = await r.json()
        assert body["chunks"] >= 1
    async with session.post(base + "/api/rag/query",
                            json={"q": "wombats", "k": 3}) as r:
        assert r.status == 200
        assert "wombat" in (await r.json())["context"].lower()
    async with session.get(base + "/api/rag/docs") as r:
        assert r.status == 200
        assert any("dash_test_doc" in d["source"] for d in (await r.json())["docs"])
    async with session.delete(base + "/api/rag/docs?source=dash_test_doc.txt") as r:
        assert r.status == 200
    import os
    for f in os.listdir("data/documents/uploads"):
        if f.startswith("dash_test_doc"):
            os.remove(os.path.join("data/documents/uploads", f))


def _shell_headers(base, session):
    cookies = session.cookie_jar.filter_cookies(base)
    return {"Cookie": f"karma_dash={cookies['karma_dash'].value}"}


async def _shell_read_text(ws, timeout=10.0):
    msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
    assert msg.type == aiohttp.WSMsgType.TEXT, msg.type
    return json.loads(msg.data)


@pytest.mark.anyio
async def test_shell_requires_auth(dash_server):
    async with make_session() as session:
        with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
            await session.ws_connect(dash_server + "/api/shell")
        assert exc.value.status == 401


@pytest.mark.anyio
async def test_shell_echo_and_exit(authed):
    session, base = authed
    ws_url = base.replace("http://", "ws://") + "/api/shell"
    async with session.ws_connect(ws_url, headers=_shell_headers(base, session)) as ws:
        await asyncio.sleep(2.0)  # let bash print its first prompt
        await ws.send_str(json.dumps({"t": "i", "d": "echo shell-ok-987\n"}))
        found = False
        for _ in range(100):
            data = await _shell_read_text(ws)
            if data.get("t") == "o":
                if "shell-ok-987" in base64.b64decode(data["d"]).decode("utf-8", "replace"):
                    found = True
                    break
            elif data.get("t") == "x":
                break
        assert found, "shell did not echo command output"
        await ws.send_str(json.dumps({"t": "i", "d": "exit\n"}))
        closed = False
        for _ in range(50):
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
            except asyncio.TimeoutError:
                break
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                closed = True
                break
            try:
                if json.loads(msg.data).get("t") == "x":
                    closed = True
                    break
            except Exception:
                pass
        assert closed, "shell did not report exit"
