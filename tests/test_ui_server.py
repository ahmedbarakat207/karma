import asyncio
import json
import pytest
import aiohttp

from src.state import internal_state
from src.ui.kiosk import kiosk_manager
from src.ui.server import start_ui_server, stop_ui_server, broadcast_state_threadsafe


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
async def test_ui_websocket_lifecycle_and_actions():
    port = 8799
    started = start_ui_server(host="127.0.0.1", port=port)
    assert started is True

    await asyncio.sleep(0.3)

    internal_state.mood = "warm"
    internal_state.energy = 0.90
    internal_state.curiosity = 0.80
    internal_state.set_active_code("int x = 42;", lang="c")

    ws_url = f"http://127.0.0.1:{port}/ws"

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            msg1 = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
            assert msg1["type"] == "state_update"
            assert msg1["mood"] == "warm"
            assert msg1["energy"] == 0.90
            assert msg1["code"] == "int x = 42;"

            msg2 = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
            assert msg2["type"] == "kiosk_data"
            assert isinstance(msg2["student_apps"], list)
            assert isinstance(msg2["achievements"], list)

            await ws.send_json({"action": "open_view", "view": "map"})
            await asyncio.sleep(0.1)
            assert kiosk_manager.active_view == "map"

            await ws.send_json({"action": "switch_floor", "floor": 1})
            await asyncio.sleep(0.1)
            assert kiosk_manager.current_floor_idx == 1

            await ws.send_json({"action": "clear_code"})
            await asyncio.sleep(0.1)
            assert internal_state.get_active_code() is None

            await ws.send_json({"action": "close_kiosk"})
            await asyncio.sleep(0.1)
            assert kiosk_manager.active_view == "face"

    stop_ui_server()
