"""
Main Vision Pipeline Loop.
"""
import time
from typing import Set
import cv2

from src import config
from src.vision.detector import ObjectDetector
from src.vision.face import FaceAndGazeTracker
from src.vision.hand import HandTracker
from src.vision.render import VisionRenderer


def run_vision(memory, stop_event, speaking_event=None) -> None:
    """Main camera capture and computer vision loop."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[vision] could not open webcam -- check macOS camera permissions.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_tracker = FaceAndGazeTracker()
    hand_tracker = HandTracker()
    object_detector = ObjectDetector()

    show_window = getattr(config, "SHOW_VISION_WINDOW", True)
    log_console = getattr(config, "LOG_VISION_TO_CONSOLE", True)

    fps_time = time.time()
    frame_count = 0
    display_fps = 0.0
    last_seen_labels: Set[str] = set()

    print(f"[vision] running high-FPS vision pipeline on {config.YOLO_DEVICE}")

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                display_fps = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now

            # 1. Object detection
            obj_labels, positions, bboxes = object_detector.process(frame, memory)

            # 2. Face, gaze, and identity tracking
            face_labels, recognized_people, primary_face = face_tracker.process(frame, memory)
            if recognized_people:
                memory.set_recognized_people(recognized_people)

            # 3. 3D Hand tracking
            hand_labels, hand_pts = hand_tracker.process(frame)

            # Aggregate observations
            current_labels = set(obj_labels + face_labels + hand_labels)
            for p in recognized_people:
                current_labels.discard("person")
                current_labels.add(p)

            # Update working memory consciousness spatial map
            spatial_objects = [(lbl, positions.get(lbl, (320.0, 240.0))) for lbl in current_labels]
            memory.consciousness.update(spatial_objects)

            # Log newly appeared objects
            new_labels = current_labels - last_seen_labels
            for label in new_labels:
                if log_console:
                    print(f"[vision] {label}")
                memory.add(kind="object", text=label, dedup_seconds=config.OBJECT_DEDUP_SECONDS)

            last_seen_labels = current_labels

            # UI Rendering
            if show_window:
                annotated = frame.copy()
                VisionRenderer.draw_objects(annotated, bboxes)
                VisionRenderer.draw_hands(annotated, hand_pts)
                if primary_face:
                    VisionRenderer.draw_face(annotated, primary_face)

                is_talking = bool(speaking_event and speaking_event.is_set())
                annotated = VisionRenderer.draw_hud(annotated, display_fps, display_fps, is_talking=is_talking)

                cv2.imshow("Karma Vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

    except Exception as e:
        print(f"[vision] pipeline exception: {e}")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
