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

    def process(self, frame: np.ndarray, memory) -> Tuple[List[str], Set[str], Optional[Tuple[int, int, int, int, str, str]]]:
        labels: List[str] = []
        recognized_names: Set[str] = set()
        primary_face_info = None

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
                    top, right, bottom, left = locs[0]
                    faces = [(left, top, right - left, bottom - top)]
            except Exception:
                faces = []

        now = time.time()

        if len(faces) > 0:
            sx, sy, sw, sh = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = sx * 2, sy * 2, sw * 2, sh * 2

            memory.set_face_frame(frame.copy())

            recognized_name = None
            if self.face_registry and _face_recognition and (now - self.last_face_rec_time) >= self.face_rec_interval:
                self.last_face_rec_time = now
                try:
                    from src.memory.face_registry import FACE_REC_LOCK
                    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    face_locations = [(y, x + w, y + h, x)]
                    with FACE_REC_LOCK:
                        encodings = _face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
                    if encodings:
                        recognized_name = self.face_registry.recognize(encodings[0])
                except Exception:
                    pass

            if recognized_name:
                recognized_names.add(recognized_name)

            emotion = "neutral"
            if self.smile_cascade is not None:
                try:
                    face_roi = gray[y:y+h, x:x+w]
                    smiles = self.smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20)
                    if len(smiles) > 0:
                        emotion = "smiling"
                except Exception:
                    pass

            display_name = recognized_name or "Face"
            labels.append(f"{display_name} looking at you ({emotion})")
            primary_face_info = (x, y, w, h, display_name, emotion)

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

        return labels, recognized_names, primary_face_info
