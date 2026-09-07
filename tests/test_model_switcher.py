import asyncio
import json
import os
import pytest
import aiohttp
from unittest.mock import MagicMock, patch

from src import config
from src.cognition.engine import (
    SwitchableEngine,
    create_switchable_engine,
    create_engine,
    LocalEngine,
    GroqEngine,
)
from src.ui import server as dash


def make_session():
    return aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True))


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
async def dash_server():
    port = 8899
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
        yield session, dash_server


def test_switchable_engine_init_and_switch():
    engine = create_switchable_engine(use_groq=False)
    assert engine.active_provider == "local"
    assert not engine.is_groq

    # Switch to groq
    engine.switch("groq", groq_model="openai/gpt-oss-20b")
    assert engine.active_provider == "groq"
    assert engine.is_groq
    assert "gpt-oss-20b" in engine.active_model_name

    # Switch to groq with preset shorthand
    engine.switch("groq", groq_model="gpt-oss-120b")
    assert config.GROQ_MODEL == "openai/gpt-oss-120b"

    # Switch back to local
    engine.switch("local")
    assert engine.active_provider == "local"
    assert not engine.is_groq

    # Invalid provider
    with pytest.raises(ValueError):
        engine.switch("unsupported_backend")


def test_switchable_engine_sync_with_config():
    engine = create_switchable_engine(use_groq=False)
    assert engine.active_provider == "local"

    config.USE_GROQ = True
    config.GROQ_MODEL = "llama-3.3-70b-versatile"
    engine.sync_with_config()
    assert engine.active_provider == "groq"
    assert engine.is_groq

    config.USE_GROQ = False
    engine.sync_with_config()
    assert engine.active_provider == "local"


def test_switchable_engine_delegation():
    engine = SwitchableEngine(use_groq=True, groq_model="openai/gpt-oss-20b")

    mock_groq = MagicMock()
    mock_groq.chat.return_value = "hello from mock groq"
    mock_groq.stream_chat.return_value = iter(["hello", " from", " stream"])
    mock_groq.model = "openai/gpt-oss-20b"

    mock_local = MagicMock()
    mock_local.chat.return_value = "hello from mock local"
    mock_local.stream_chat.return_value = iter(["local", " stream"])

    engine._groq_engine = mock_groq
    engine._local_engine = mock_local

    # While in groq mode
    assert engine.chat("sys", "user") == "hello from mock groq"
    assert list(engine.stream_chat("sys", "user")) == ["hello", " from", " stream"]

    # Switch to local mode
    engine.switch("local")
    assert engine.chat("sys", "user") == "hello from mock local"
    assert list(engine.stream_chat("sys", "user")) == ["local", " stream"]


def test_create_engine_backward_compatibility():
    # Calling with switchable=True returns SwitchableEngine
    sw = create_engine(switchable=True)
    assert isinstance(sw, SwitchableEngine)

    # Calling with use_groq=True returns GroqEngine directly
    groq = create_engine(use_groq=True, model_name="gpt-oss-20b")
    assert isinstance(groq, GroqEngine)
    assert groq.model == "openai/gpt-oss-20b"


@pytest.mark.anyio
async def test_api_model_auth_and_get(dash_server, authed):
    # Unauthenticated request
    async with make_session() as session:
        async with session.get(dash_server + "/api/model") as r:
            assert r.status == 401

    session, base = authed
    async with session.get(base + "/api/model") as r:
        assert r.status == 200
        data = await r.json()
        assert "provider" in data
        assert "use_groq" in data
        assert "groq_model" in data
        assert "available_groq_models" in data
        assert "has_groq_key" in data
        assert "openai/gpt-oss-20b" in data["available_groq_models"]


@pytest.mark.anyio
async def test_api_model_post_switching(authed):
    session, base = authed

    # Create and set a mock switchable engine in server runtime
    mock_engine = MagicMock(spec=SwitchableEngine)
    mock_engine.active_provider = "local"
    mock_engine.is_groq = False
    dash.set_runtime(engine=mock_engine)

    # Switch to groq via POST /api/model
    async with session.post(base + "/api/model", json={
        "provider": "groq",
        "groq_model": "llama-3.3-70b-versatile",
        "api_key": "gsk_test_12345"
    }) as r:
        assert r.status == 200
        data = await r.json()
        assert data["ok"] is True
        assert data["provider"] == "groq"
        assert data["use_groq"] is True
        assert data["groq_model"] == "llama-3.3-70b-versatile"
        assert config.USE_GROQ is True
        assert config.GROQ_MODEL == "llama-3.3-70b-versatile"
        assert os.environ.get("GROQ_API_KEY") == "gsk_test_12345"
        mock_engine.switch.assert_called_with("groq", groq_model="llama-3.3-70b-versatile")

    # State update should reflect groq
    async with session.get(base + "/api/state") as r:
        assert r.status == 200
        state = await r.json()
        assert "model" in state
        assert state["model"]["provider"] == "groq"
        assert state["model"]["use_groq"] is True

    # Switch back to local
    async with session.post(base + "/api/model", json={"provider": "local"}) as r:
        assert r.status == 200
        data = await r.json()
        assert data["ok"] is True
        assert data["provider"] == "local"
        assert data["use_groq"] is False
        assert config.USE_GROQ is False
        mock_engine.switch.assert_called_with("local", groq_model=config.GROQ_MODEL)

    # Test bad input
    async with session.post(base + "/api/model", json={"provider": "invalid"}) as r:
        assert r.status == 400


@pytest.mark.anyio
async def test_api_config_with_groq_settings(authed):
    session, base = authed
    async with session.get(base + "/api/config") as r:
        assert r.status == 200
        cfg = await r.json()
        assert "USE_GROQ" in cfg
        assert "GROQ_MODEL" in cfg

    async with session.put(base + "/api/config", json={"USE_GROQ": True, "GROQ_MODEL": "openai/gpt-oss-120b"}) as r:
        assert r.status == 200
        res = await r.json()
        assert res["ok"] is True
        assert res["updated"]["USE_GROQ"] is True
        assert res["updated"]["GROQ_MODEL"] == "openai/gpt-oss-120b"
        assert config.USE_GROQ is True
        assert config.GROQ_MODEL == "openai/gpt-oss-120b"
