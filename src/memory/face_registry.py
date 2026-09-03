
import json
import os
import threading
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np

from src import config

# Global thread lock to protect dlib/face_recognition C++ bindings across threads
FACE_REC_LOCK = threading.Lock()


class FaceRegistry:

    def __init__(self, path: Optional[str] = None):
        self._path = path or getattr(config, "FACE_REGISTRY_PATH",
                                     os.path.join(config.BASE_DIR, "faces.json"))
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                print(f"[face_registry] loaded {len(self._entries)} known face(s) from {self._path}")
            except Exception as e:
                print(f"[face_registry] failed to load {self._path}: {e}")
                self._entries = []
        else:
            self._entries = []

    def _save_atomic(self) -> None:
        tmp_path = f"{self._path}.tmp"
        try:
            parent = os.path.dirname(self._path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2)
            os.replace(tmp_path, self._path)
        except Exception as e:
            print(f"[face_registry] atomic save error: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def register(self, name: str, face_encoding: Union[np.ndarray, List[float]]) -> None:
        name = name.strip().title()
        enc_list = face_encoding.tolist() if isinstance(face_encoding, np.ndarray) else list(face_encoding)

        with self._lock:
            for entry in self._entries:
                if entry["name"].lower() == name.lower():
                    existing = np.array(entry["encoding"], dtype=np.float64)
                    new_avg = ((existing + np.array(enc_list, dtype=np.float64)) / 2.0).tolist()
                    entry["encoding"] = new_avg
                    entry["registered_ts"] = time.time()
                    self._save_atomic()
                    print(f"[face_registry] updated face encoding for '{name}'")
                    return

            self._entries.append({
                "name": name,
                "encoding": enc_list,
                "registered_ts": time.time(),
            })
            self._save_atomic()
            print(f"[face_registry] registered new face: '{name}'")

    def recognize(self, face_encoding: Union[np.ndarray, List[float]],
                  tolerance: Optional[float] = None) -> Optional[str]:
        if tolerance is None:
            tolerance = getattr(config, "FACE_RECOGNITION_TOLERANCE", 0.55)

        enc = np.array(face_encoding, dtype=np.float64)

        with self._lock:
            if not self._entries:
                return None

            best_name = None
            best_dist = float("inf")

            for entry in self._entries:
                known_enc = np.array(entry["encoding"], dtype=np.float64)
                dist = float(np.linalg.norm(enc - known_enc))
                if dist < best_dist:
                    best_dist = dist
                    best_name = entry["name"]

            if best_dist <= tolerance:
                return best_name
        return None

    def known_names(self) -> List[str]:
        with self._lock:
            return [e["name"] for e in self._entries]

    def count(self) -> int:
        with self._lock:
            return len(self._entries)
