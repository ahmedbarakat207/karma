import time
from typing import List, Tuple, Set, Optional
import cv2
import numpy as np

from src import config

_face_recognition = None
try:
    if getattr(config, "FACE_RECOGNITION_ENABLED", True):
        import face_recognition as _face_recognition
except ImportError:
    pass


class FaceAndGazeTracker:
    def __init__(self):
        self.face_cascade = None
        self.profile_cascade = None
        self.smile_cascade = None

        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
                self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
            except Exception as e:
                config.log_debug(f"[vision] CascadeClassifier load note: {e}")

        self.face_registry = None
        if _face_recognition and getattr(config, "FACE_RECOGNITION_ENABLED", True):
            from src.memory.face_registry import FaceRegistry
            self.face_registry = FaceRegistry()
            config.log_debug(f"[vision] Face recognition enabled ({self.face_registry.count()} known face(s))")

        self.last_face_rec_time = 0.0
        self.face_rec_interval = getattr(config, "FACE_RECOGNITION_INTERVAL", 0.5)
        self._cached_names: Set[str] = set()
        self._max_faces = 4  # Pi perf cap: ID largest N faces per recognition tick

    def _detect_smile(self, gray, x: int, y: int, w: int, h: int) -> str:
        if self.smile_cascade is None:
            return "neutral"
        try:
            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size == 0:
                return "neutral"
            smiles = self.smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20)
            if len(smiles) > 0:
                return "smiling"
        except Exception:
            pass
        return "neutral"

    def process(self, frame: np.ndarray, memory) -> Tuple[List[str], Set[str], Optional[Tuple[int, int, int, int, str, str]], List[Tuple[int, int, int, int, str, str]]]:
        """Detect + identify all visible faces.

        Returns (labels, recognized_names, primary_face, all_faces).
        all_faces is needed to fuse YOLO person boxes -> names.
        recognized_names is throttled (0.5s) and cached so the HUD and
        prompt context don't flicker on skipped frames.
        """
        labels: List[str] = []
        recognized_names: Set[str] = set()
        primary_face_info = None
        all_faces: List[Tuple[int, int, int, int, str, str]] = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

        faces = []
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
            except Exception:
                faces = []

        if len(faces) == 0 and _face_recognition is not None:
            try:
                from src.memory.face_registry import FACE_REC_LOCK
                rgb_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
                with FACE_REC_LOCK:
                    locs = _face_recognition.face_locations(rgb_small, model="hog")
                if locs:
                    faces = [(left, top, right - left, bottom - top)
                             for top, right, bottom, left in locs]
            except Exception:
                faces = []

        now = time.time()

        if len(faces) > 0:
            # Largest first, cap count for Pi CPU (dlib encodings are expensive).
            scaled = [(sx * 2, sy * 2, sw * 2, sh * 2) for sx, sy, sw, sh in faces]
            scaled.sort(key=lambda f: f[2] * f[3], reverse=True)
            scaled = scaled[:self._max_faces]

            memory.set_face_frame(frame.copy())

            should_recognize = (
                self.face_registry is not None
                and _face_recognition is not None
                and (now - self.last_face_rec_time) >= self.face_rec_interval
            )
            fresh_names: Set[str] = set()
            if should_recognize:
                self.last_face_rec_time = now
                try:
                    from src.memory.face_registry import FACE_REC_LOCK
                    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    for (x, y, w, h) in scaled:
                        try:
                            with FACE_REC_LOCK:
                                encodings = _face_recognition.face_encodings(
                                    rgb_frame, known_face_locations=[(y, x + w, y + h, x)]
                                )
                            if encodings:
                                name = self.face_registry.recognize(encodings[0])
                                if name:
                                    fresh_names.add(name)
                        except Exception:
                            continue
                except Exception:
                    pass
                self._cached_names = set(fresh_names)
                recognized_names = set(fresh_names)
            else:
                # Throttled frame: reuse last IDs so presence doesn't flicker.
                recognized_names = set(self._cached_names)

            # Assign cached names to boxes by size order (largest face = most
            # likely the previously identified person in the 1-person case).
            ordered_names = sorted(recognized_names)
            for i, (x, y, w, h) in enumerate(scaled):
                emotion = self._detect_smile(gray, x, y, w, h)
                # Single-face case: attribute the cached name to it.
                if len(scaled) == 1 and len(ordered_names) == 1:
                    display_name = ordered_names[0]
                elif i < len(ordered_names) and len(scaled) == len(ordered_names):
                    display_name = ordered_names[i]
                else:
                    display_name = ordered_names[0] if len(ordered_names) == 1 and i == 0 else "Face"
                    # Multi-face with ambiguous cache: only first box keeps the
                    # name unless a fresh recognition just ran with equal counts.
                    if should_recognize and i < len(sorted(fresh_names)):
                        display_name = sorted(fresh_names)[i]
                labels.append(f"{display_name} looking at you ({emotion})")
                all_faces.append((x, y, w, h, display_name, emotion))

            if all_faces:
                primary_face_info = max(all_faces, key=lambda f: f[2] * f[3])

        else:
            profiles = []
            if self.profile_cascade is not None:
                try:
                    profiles = self.profile_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
                except Exception:
                    profiles = []

            if len(profiles) > 0:
                sx, sy, sw, sh = max(profiles, key=lambda f: f[2] * f[3])
                x, y, w, h = sx * 2, sy * 2, sw * 2, sh * 2
                labels.append("Face (looking away)")
                primary_face_info = (x, y, w, h, "Face", "looking away")
                all_faces.append((x, y, w, h, "Face", "looking away"))

        return labels, recognized_names, primary_face_info, all_faces
