"""
Vision HUD, ASCII emotions, and bounding box drawing.
"""
from typing import List, Tuple
import cv2
import numpy as np

from src.state import internal_state
from src.vision.hand import HAND_CONNECTIONS


class VisionRenderer:
    """Renders real-time HUD, ASCII emotions, bounding boxes, and hand landmarks."""

    @staticmethod
    def draw_hud(frame: np.ndarray, fps: float, display_fps: float, is_talking: bool = False) -> np.ndarray:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)

        expression = internal_state.get_expression(is_talking=is_talking)
        mood = (internal_state.current_emotion or internal_state.mood).upper()
        cv2.putText(frame, f"Karma: {expression} [{mood}]", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        fps_text = f"{display_fps:.1f} FPS"
        cv2.putText(frame, fps_text, (w - 110, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)

        return frame

    @staticmethod
    def draw_hands(frame: np.ndarray, hand_landmarks: List[List[Tuple[int, int]]]) -> None:
        for pts in hand_landmarks:
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(pts) and p2 < len(pts):
                    cv2.line(frame, pts[p1], pts[p2], (0, 255, 255), 2)
            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 150, 255), -1)

    @staticmethod
    def draw_face(frame: np.ndarray, face_info: Tuple[int, int, int, int, str, str]) -> None:
        x, y, w, h, name, emotion = face_info
        color = (255, 165, 0) if name != "Face" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} ({emotion})", (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def draw_objects(frame: np.ndarray, bboxes: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> None:
        for name, conf, (x1, y1, x2, y2) in bboxes:
            if name == "person":
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 100), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
