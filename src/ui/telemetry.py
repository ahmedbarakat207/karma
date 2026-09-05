"""Real system telemetry for the dashboard and the on-device screen.

Everything here is stdlib-only (no psutil dependency) so it works on the
Pi service image. Snapshots are cached for 2 seconds because the websocket
broadcaster asks often.
"""

import os
import shutil
import socket
import time
from typing import Any, Dict, List, Optional

_CACHE_SECONDS = 2.0

_snapshot_cache: Optional[Dict[str, Any]] = None
_snapshot_ts: float = 0.0

_boot_time = time.time()

# cpu percent via /proc/stat deltas
_last_cpu: Optional[tuple] = None
_last_cpu_ts: float = 0.0

# rolling llm stats, fed by the engines
_llm_lock_ts = 0.0
_llm = {"calls": 0, "last_tps": 0.0, "last_ttft_ms": 0.0, "avg_tps": 0.0, "last_tokens": 0}


def record_llm(ttft_s: float, duration_s: float, tokens: int) -> None:
    tps = (tokens / duration_s) if duration_s > 0 and tokens > 0 else 0.0
    n = _llm["calls"]
    _llm["calls"] = n + 1
    _llm["last_tps"] = round(tps, 1)
    _llm["last_ttft_ms"] = round(ttft_s * 1000.0, 0)
    _llm["last_tokens"] = tokens
    _llm["avg_tps"] = round(((_llm["avg_tps"] * n) + tps) / (n + 1), 1)


_ws_clients = 0


def set_ws_clients(n: int) -> None:
    global _ws_clients
    _ws_clients = n


def cpu_temp_c() -> Optional[float]:
    for zone in ("thermal_zone0", "thermal_zone1"):
        try:
            with open(f"/sys/class/thermal/{zone}/temp") as f:
                return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            continue
    return None


def throttled_state() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=2).strip()
        code = out.split("=")[1] if "=" in out else ""
        return "clean" if code in ("throttled=0x0", "0x0") else code or "unknown"
    except Exception:
        return "unknown"


def cpu_percent() -> Optional[float]:
    global _last_cpu, _last_cpu_ts
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()[1:8]
        vals = [int(x) for x in parts]
        idle = vals[3] + vals[4]
        total = sum(vals)
        now = time.time()
        if _last_cpu is None:
            _last_cpu, _last_cpu_ts = (idle, total), now
            return None
        (idle0, total0), t0 = _last_cpu, _last_cpu_ts
        _last_cpu, _last_cpu_ts = (idle, total), now
        d_total, d_idle = total - total0, idle - idle0
        if d_total <= 0:
            return None
        return round(100.0 * (1.0 - d_idle / d_total), 1)
    except Exception:
        return None


def mem_info() -> Dict[str, Any]:
    try:
        info: Dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.strip().split()[0])
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        used_pct = round(100.0 * (1.0 - avail / total), 1) if total else None
        return {"total_kb": total, "avail_kb": avail, "used_pct": used_pct}
    except Exception:
        return {"total_kb": 0, "avail_kb": 0, "used_pct": None}


def disk_info() -> Dict[str, Any]:
    try:
        du = shutil.disk_usage("/")
        return {
            "total_gb": round(du.total / 1e9, 1),
            "free_gb": round(du.free / 1e9, 1),
            "used_pct": round(100.0 * du.used / du.total, 1),
        }
    except Exception:
        return {"total_gb": 0.0, "free_gb": 0.0, "used_pct": None}


def uptime_s() -> float:
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return time.time() - _boot_time


def lan_ips() -> List[str]:
    ips: List[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
        finally:
            s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


def snapshot() -> Dict[str, Any]:
    """Cached full telemetry snapshot (safe to call at 20 Hz)."""
    global _snapshot_cache, _snapshot_ts
    now = time.time()
    if _snapshot_cache is not None and now - _snapshot_ts < _CACHE_SECONDS:
        return _snapshot_cache
    mem = mem_info()
    snap = {
        "cpu_temp_c": cpu_temp_c(),
        "cpu_pct": cpu_percent(),
        "throttled": throttled_state(),
        "mem_used_pct": mem["used_pct"],
        "mem_avail_mb": round(mem["avail_kb"] / 1024.0, 0) if mem["avail_kb"] else None,
        "disk_used_pct": disk_info()["used_pct"],
        "disk_free_gb": disk_info()["free_gb"],
        "load_1": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
        "uptime_s": int(uptime_s()),
        "llm": dict(_llm),
        "ws_clients": _ws_clients,
    }
    _snapshot_cache = snap
    _snapshot_ts = now
    return snap


def net_info(dash_port: int) -> Dict[str, Any]:
    ips = lan_ips()
    return {
        "hostname": socket.gethostname(),
        "ips": ips,
        "dash_port": dash_port,
        "dash_url": f"http://{ips[0]}:{dash_port}" if ips else "",
    }
