import math
import time
from typing import Set, Tuple, List, Dict, Optional
import cv2

from src import config
from src.hardware.neck import neck_actuator
from src.state import internal_state
from src.ui.kiosk import kiosk_manager
from src.vision.detector import ObjectDetector
from src.vision.face import FaceAndGazeTracker
from src.vision.hand import HandTracker
from src.vision.render import VisionRenderer, FaceRenderer


def get_display_resolution() -> Tuple[int, int]:
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
    return getattr(config, "DISPLAY_WIDTH", 800), getattr(config, "DISPLAY_HEIGHT", 480)


def run_vision(memory, stop_event, speaking_event=None) -> None:
    cam_idx = getattr(config, "CAMERA_INDEX", 0)
    cap = None
    has_camera = False

    try:
        test_cap = cv2.VideoCapture(cam_idx)
        if test_cap is not None and test_cap.isOpened():
            ok, test_frame = test_cap.read()
            if ok and test_frame is not None and test_frame.size > 0:
                has_camera = True
                cap = test_cap
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                test_cap.release()
    except Exception as e:
        config.log_debug(f"[vision] camera probe note: {e}")

    if not has_camera:
        print(f"\n[vision] ⚠️ No camera hardware found at index {cam_idx}.")
        print("[vision] 💤 YOLO object detection & spatial trackers are DISABLED to save CPU and RAM.")
        print("[vision] 📺 Fullscreen companion face and voice cognition loop remain active.\n")
        object_detector = None
        face_tracker = None
        hand_tracker = None
    else:
        config.log_debug(f"[vision] camera verified at index {cam_idx}. Initializing detectors...")
        face_tracker = FaceAndGazeTracker()
        hand_tracker = HandTracker()
        if getattr(config, "ENABLE_YOLO", True):
            try:
                object_detector = ObjectDetector()
                config.log_debug(f"[vision] YOLOv8 enabled on {config.YOLO_DEVICE}")
            except Exception as e:
                print(f"[vision] ⚠️ Could not load YOLO model: {e}. Disabling YOLO.")
                object_detector = None
        else:
            object_detector = None
            config.log_debug("[vision] YOLOv8 disabled via ENABLE_YOLO=0")

    screen_w, screen_h = get_display_resolution()
    face_renderer = FaceRenderer(width=screen_w, height=screen_h)

    face_window_name = "Karma"
    cv2.namedWindow(face_window_name, cv2.WINDOW_NORMAL)
    if getattr(config, "FULLSCREEN_FACE", True):
        try:
            cv2.setWindowProperty(face_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass

    curr_dims = [screen_w, screen_h]

    def on_touch(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if internal_state.get_active_code() is not None and x >= int(curr_dims[0] * 0.30):
                internal_state.clear_active_code()
                return
            kiosk_manager.handle_touch(x, y, curr_dims[0], curr_dims[1])

    try:
        cv2.setMouseCallback(face_window_name, on_touch)
    except Exception as e:
        config.log_debug(f"[vision] mouse callback warning: {e}")

    fps_time = time.time()
    frame_count = 0
    display_fps = 30.0
    last_seen_labels: Set[str] = set()

    try:
        while not stop_event.is_set():
            frame = None
            bboxes = []
            hand_pts = None
            primary_face = None

            if has_camera and cap is not None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                now = time.time()
                if now - fps_time >= 1.0:
                    display_fps = frame_count / (now - fps_time)
                    frame_count = 0
                    fps_time = now

                obj_labels, positions, bboxes = ([], {}, [])
                if object_detector is not None:
                    obj_labels, positions, bboxes = object_detector.process(frame, memory)

                face_labels, recognized_people, primary_face = ([], [], None)
                if face_tracker is not None:
                    face_labels, recognized_people, primary_face = face_tracker.process(frame, memory)
                    if recognized_people:
                        memory.set_recognized_people(recognized_people)

                if primary_face:
                    fx, fy, fw, fh, fname, femotion = primary_face
                    norm_x = -((fx + fw / 2.0) - 320.0) / 320.0
                    norm_y = ((fy + fh / 2.0) - 240.0) / 240.0
                    internal_state.set_gaze(norm_x, norm_y, is_present=True)
                else:
                    internal_state.set_gaze(0.0, 0.0, is_present=False)

                hand_labels = []
                if hand_tracker is not None:
                    hand_labels, hand_pts = hand_tracker.process(frame)

                current_labels = set(obj_labels + face_labels + hand_labels)
                for p in recognized_people:
                    current_labels.discard("person")
                    current_labels.add(p)

                spatial_objects = [(lbl, positions.get(lbl, (320.0, 240.0))) for lbl in current_labels]
                memory.consciousness.update(spatial_objects)

                new_labels = current_labels - last_seen_labels
                for label in new_labels:
                    if getattr(config, "LOG_VISION_TO_CONSOLE", False):
                        print(f"[vision] {label}")
                    memory.add(kind="object", text=label, dedup_seconds=config.OBJECT_DEDUP_SECONDS)
                last_seen_labels = current_labels

            else:
                time.sleep(getattr(config, "VISION_POLL_SECONDS", 0.033))
                t = time.time()
                idle_x = 0.12 * math.sin(t * 0.5)
                idle_y = 0.06 * math.cos(t * 0.35)
                internal_state.set_gaze(idle_x, idle_y, is_present=False)
                display_fps = 30.0

            is_talking = bool(
                (speaking_event and speaking_event.is_set())
                or getattr(internal_state, "is_playing_audio", False)
            )
            is_user_speaking = bool(memory.is_user_speaking())

            target_w, target_h = screen_w, screen_h
            try:
                rect = cv2.getWindowImageRect(face_window_name)
                if rect is not None and len(rect) == 4 and rect[2] > 100 and rect[3] > 100:
                    target_w, target_h = int(rect[2]), int(rect[3])
            except Exception:
                pass
            curr_dims[0], curr_dims[1] = target_w, target_h

            if kiosk_manager.is_active():
                display_frame = kiosk_manager.render_kiosk(width=target_w, height=target_h)
            else:
                display_frame = face_renderer.render(
                    is_talking=is_talking,
                    is_user_speaking=is_user_speaking,
                    fps=display_fps,
                    target_shape=(target_h, target_w)
                )
                kiosk_manager.render_overlay_button(display_frame)

            cv2.imshow(face_window_name, display_frame)

            if getattr(config, "SHOW_VISION_WINDOW", False) and frame is not None:
                annotated = frame.copy()
                VisionRenderer.draw_objects(annotated, bboxes)
                VisionRenderer.draw_hands(annotated, hand_pts)
                if primary_face:
                    VisionRenderer.draw_face(annotated, primary_face)
                annotated = VisionRenderer.draw_hud(annotated, display_fps, display_fps, is_talking=is_talking)
                cv2.imshow(camera_window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (4, 27, ord('q'), ord('Q')):
                stop_event.set()
                break
            elif key in (ord('m'), ord('M')):
                if kiosk_manager.is_active():
                    kiosk_manager.close()
                else:
                    kiosk_manager.open_view("map")
            elif key in (ord('f'), ord('F')):
                config.FULLSCREEN_FACE = not getattr(config, "FULLSCREEN_FACE", True)
                prop = cv2.WINDOW_FULLSCREEN if config.FULLSCREEN_FACE else cv2.WINDOW_NORMAL
                try:
                    cv2.setWindowProperty(face_window_name, cv2.WND_PROP_FULLSCREEN, prop)
                except Exception:
                    pass
            elif key in (ord('d'), ord('D')) and has_camera:
                config.SHOW_VISION_WINDOW = not getattr(config, "SHOW_VISION_WINDOW", False)
                if not config.SHOW_VISION_WINDOW:
                    try:
                        cv2.destroyWindow(camera_window_name)
                    except Exception:
                        pass

            try:
                if cv2.getWindowProperty(face_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    stop_event.set()
                    break
            except Exception:
                pass

    except Exception as e:
        config.log_debug(f"[vision] pipeline exception: {e}")
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        try:
            neck_actuator.cleanup()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

