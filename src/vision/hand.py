"""
MediaPipe 3D Hand and Finger Landmark Tracking.
"""
import atexit
import os
import sys
from typing import List, Tuple
import cv2
import numpy as np

from src import config


class SilenceStderrFD:
    """Temporarily silences C-level file descriptor 2 (stderr) to suppress C++ framework log output."""
    def __enter__(self):
        try:
            sys.stderr.flush()
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.saved_stderr_fd = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.null_fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, "null_fd", None) is not None:
            try:
                sys.stderr.flush()
                os.dup2(self.saved_stderr_fd, 2)
                os.close(self.saved_stderr_fd)
                os.close(self.null_fd)
            except Exception:
                pass


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]


class HandTracker:
    """Tracks 3D hand and finger landmarks via MediaPipe."""

    def __init__(self):
        self.detector = None
        task_path = os.path.join(config.BASE_DIR, "models", "hand_landmarker.task")
        if os.path.exists(task_path):
            try:
                with SilenceStderrFD():
                    from mediapipe.tasks import python as mp_python
                    from mediapipe.tasks.python import vision as mp_vision
                    base_opts = mp_python.BaseOptions(model_asset_path=task_path)
                    opts = mp_vision.HandLandmarkerOptions(
                        base_options=base_opts,
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.5,
                    )
                    self.detector = mp_vision.HandLandmarker.create_from_options(opts)
                print("[vision] MediaPipe 3D Hand & Finger Tracking initialized!")
                atexit.register(self.close)
            except Exception as e:
                print(f"[vision] HandLandmarker init note: {e}")

    def process(self, frame: np.ndarray) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
        labels: List[str] = []
        hand_landmarks_pixels: List[List[Tuple[int, int]]] = []

        if not self.detector:
            return labels, hand_landmarks_pixels

        try:
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with SilenceStderrFD():
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detector.detect(mp_img)

            if result.hand_landmarks:
                h, w = frame.shape[:2]
                for hand in result.hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                    hand_landmarks_pixels.append(pts)

                    # Finger tip coordinates
                    thumb_tip = hand[4]
                    index_tip = hand[8]
                    middle_tip = hand[12]
                    ring_tip = hand[16]
                    pinky_tip = hand[20]

                    # Fingers extended checks
                    index_up = index_tip.y < hand[6].y
                    middle_up = middle_tip.y < hand[10].y
                    ring_up = ring_tip.y < hand[14].y
                    pinky_up = pinky_tip.y < hand[18].y
                    thumb_up = thumb_tip.y < hand[2].y

                    if index_up and middle_up and ring_up and pinky_up:
                        labels.append("person waving hand")
                    elif thumb_up and not index_up and not middle_up:
                        labels.append("person giving thumbs up")
                    elif index_up and not middle_up and not ring_up:
                        labels.append("person pointing finger")
                    else:
                        labels.append("hand gesture")
        except Exception:
            pass

        return labels, hand_landmarks_pixels

    def close(self) -> None:
        """Cleanly close MediaPipe detector without throwing during python interpreter shutdown."""
        if self.detector:
            try:
                detector = self.detector
                self.detector = None
                # Neutralize __del__ on instance to avoid late futures submission during interpreter teardown
                type(detector).__del__ = lambda self: None
                with SilenceStderrFD():
                    detector.close()
            except Exception:
                pass
