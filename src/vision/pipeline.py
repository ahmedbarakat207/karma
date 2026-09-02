"""
Main Vision Pipeline Loop.
"""
import time
from typing import Set
import cv2

from src import config
from src.state import internal_state
from src.vision.detector import ObjectDetector
from src.vision.face import FaceAndGazeTracker
from src.vision.hand import HandTracker
from src.vision.render import VisionRenderer, FaceRenderer


def get_display_resolution() -> Tuple[int, int]:
    """Auto-detects native display resolution across macOS, Linux, and Windows."""
    try:
        from AppKit import NSScreen
        f = NSScreen.mainScreen().frame()
        return int(f.size.width), int(f.size.height)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(["xdotool", "getdisplaygeometry"], text=True).strip()
        parts = out.split()
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        pass
    # Default to the 7" 800x480 LCD Screen specified in PARTS.MD
    return getattr(config, "DISPLAY_WIDTH", 800), getattr(config, "DISPLAY_HEIGHT", 480)


def run_vision(memory, stop_event, speaking_event=None) -> None:
    """Main camera capture, face rendering, and computer vision loop."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        config.log_debug("[vision] could not open webcam -- check camera permissions.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_tracker = FaceAndGazeTracker()
    hand_tracker = HandTracker()
    object_detector = ObjectDetector()

    screen_w, screen_h = get_display_resolution()
    face_renderer = FaceRenderer(width=screen_w, height=screen_h)

    face_window_name = "Karma"
    camera_window_name = "Karma Vision [Debug]"

    # Setup Fullscreen Companion Face Window
    cv2.namedWindow(face_window_name, cv2.WINDOW_NORMAL)
    if getattr(config, "FULLSCREEN_FACE", True):
        try:
            cv2.setWindowProperty(face_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass

    fps_time = time.time()
    frame_count = 0
    display_fps = 0.0
    last_seen_labels: Set[str] = set()


    config.log_debug(f"[vision] running high-FPS vision pipeline on {config.YOLO_DEVICE}")

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

            # Update gaze tracking coordinates for Face UI
            if primary_face:
                fx, fy, fw, fh, fname, femotion = primary_face
                # Mirrored horizontal gaze tracking
                norm_x = -((fx + fw / 2.0) - 320.0) / 320.0
                norm_y = ((fy + fh / 2.0) - 240.0) / 240.0
                internal_state.set_gaze(norm_x, norm_y, is_present=True)
            else:
                internal_state.set_gaze(0.0, 0.0, is_present=False)

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

            # Log newly appeared objects (when enabled)
            new_labels = current_labels - last_seen_labels
            for label in new_labels:
                if getattr(config, "LOG_VISION_TO_CONSOLE", False):
                    print(f"[vision] {label}")
                memory.add(kind="object", text=label, dedup_seconds=config.OBJECT_DEDUP_SECONDS)

            last_seen_labels = current_labels

            # UI State
            is_talking = bool(
                (speaking_event and speaking_event.is_set())
                or getattr(internal_state, "is_playing_audio", False)
            )
            is_user_speaking = bool(memory.is_user_speaking())

            # 4. Render Primary Full-Screen Face at exact display/window resolution
            target_w, target_h = screen_w, screen_h
            try:
                rect = cv2.getWindowImageRect(face_window_name)
                if rect is not None and len(rect) == 4 and rect[2] > 100 and rect[3] > 100:
                    target_w, target_h = int(rect[2]), int(rect[3])
            except Exception:
                pass

            face_frame = face_renderer.render(
                is_talking=is_talking,
                is_user_speaking=is_user_speaking,
                fps=display_fps,
                target_shape=(target_h, target_w)
            )
            cv2.imshow(face_window_name, face_frame)


            # 5. Render Debug Camera Window (only when SHOW_VISION_WINDOW / --debug is enabled)
            if getattr(config, "SHOW_VISION_WINDOW", False):
                annotated = frame.copy()
                VisionRenderer.draw_objects(annotated, bboxes)
                VisionRenderer.draw_hands(annotated, hand_pts)
                if primary_face:
                    VisionRenderer.draw_face(annotated, primary_face)
                annotated = VisionRenderer.draw_hud(annotated, display_fps, display_fps, is_talking=is_talking)
                cv2.imshow(camera_window_name, annotated)

            # Handle Keyboard Events & Clean Exit
            key = cv2.waitKey(1) & 0xFF
            if key in (4, 27, ord('q'), ord('Q')):  # 4 = Ctrl+D (EOT), 27 = Esc, 'q' = Quit
                stop_event.set()
                break
            elif key in (ord('f'), ord('F')):  # 'f' = Toggle Fullscreen
                config.FULLSCREEN_FACE = not getattr(config, "FULLSCREEN_FACE", True)
                prop = cv2.WINDOW_FULLSCREEN if config.FULLSCREEN_FACE else cv2.WINDOW_NORMAL
                try:
                    cv2.setWindowProperty(face_window_name, cv2.WND_PROP_FULLSCREEN, prop)
                except Exception:
                    pass
            elif key in (ord('d'), ord('D')):  # 'd' = Toggle Debug Camera Window
                config.SHOW_VISION_WINDOW = not getattr(config, "SHOW_VISION_WINDOW", False)
                if not config.SHOW_VISION_WINDOW:
                    try:
                        cv2.destroyWindow(camera_window_name)
                    except Exception:
                        pass

            # Check if user closed the window
            try:
                if cv2.getWindowProperty(face_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    stop_event.set()
                    break
            except Exception:
                pass

    except Exception as e:
        config.log_debug(f"[vision] pipeline exception: {e}")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

