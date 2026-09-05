"""In-memory event bus for the dashboard and logs.

Producers (think loop, interaction, audio, vision) post small dicts.
Consumers (dashboard HTTP/WS) read recent ones. Thread-safe, bounded,
stdlib only so anything can import it without cycles.
"""

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

_MAX_EVENTS = 500

_lock = threading.Lock()
_events: Deque[Dict[str, Any]] = deque(maxlen=_MAX_EVENTS)


def post(kind: str, text: str = "", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Append an event. kind is one of: thought, reply, heard, kiosk, system, error."""
    ev = {"ts": time.time(), "kind": kind, "text": text or "", "data": data or {}}
    with _lock:
        _events.append(ev)
    return ev


def recent(kind: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Newest-first slice, optionally filtered by kind."""
    with _lock:
        items = list(_events)
    if kind:
        items = [e for e in items if e["kind"] == kind]
    items.sort(key=lambda e: e["ts"], reverse=True)
    return items[: max(0, limit)]


def count() -> int:
    with _lock:
        return len(_events)


def clear() -> None:
    with _lock:
        _events.clear()
