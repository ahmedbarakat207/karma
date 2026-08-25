"""
YOLOv8 Object Detection and Spatial Tracking.
"""
import time
from collections import deque
from typing import List, Tuple, Dict
import numpy as np

from src import config


class ObjectDetector:
    """YOLOv8 object detector with spatial movement tracking."""

    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(config.YOLO_MODEL)
        self.device = config.YOLO_DEVICE
        self.previous_positions: Dict[str, Tuple[float, float]] = {}
        self.user_x_history = deque(maxlen=10)
        self._startle_cooldowns: Dict[str, float] = {}

    def process(self, frame: np.ndarray, memory) -> Tuple[List[str], Dict[str, Tuple[float, float]], List[Tuple[str, float, Tuple[int, int, int, int]]]]:
        """
        Runs YOLO detection, calculates movements, and triggers startle events.
        Returns (labels, positions_dict, bboxes_for_drawing).
        """
        labels: List[str] = []
        positions: Dict[str, Tuple[float, float]] = {}
        bboxes: List[Tuple[str, float, Tuple[int, int, int, int]]] = []

        now = time.time()
        imgsz = getattr(config, "YOLO_IMGSZ", 320)
        results = self.model(frame, device=self.device, verbose=False, conf=config.YOLO_CONFIDENCE, imgsz=imgsz)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                labels.append(cls_name)
                positions[cls_name] = (cx, cy)
                bboxes.append((cls_name, conf, (x1, y1, x2, y2)))

                if cls_name == "person":
                    self.user_x_history.append(cx)

                # Sudden motion / startle trigger
                if cls_name in self.previous_positions:
                    prev_cx, prev_cy = self.previous_positions[cls_name]
                    dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                    if dist > 80:
                        last_startle = self._startle_cooldowns.get(cls_name, 0.0)
                        if (now - last_startle) > 10:
                            self._startle_cooldowns[cls_name] = now
                            memory.add(kind="conscious_trigger", text=f"The {cls_name} moved suddenly!", salience=0.85)

        self.previous_positions = positions
        return labels, positions, bboxes
