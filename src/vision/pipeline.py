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
from src.vision.identity import fuse_person_identities, positions_from_bboxes
from src.vision.render import VisionRenderer, FaceRenderer
from src.vision.vlm import verifier as _vlm_verifier


CAMERA_WINDOW_NAME = "Karma Vision"
_last_published = 0.0
# Presence hysteresis: don't clear a recognized name on a single missed frame.
_last_presence_names: Set[str] = set()
_last_presence_time: float = 0.0
PRESENCE_CLEAR_SECONDS = 2.0


def _store_vlm_result(memory, result, job) -> None:
    """Persist one VLM snapshot: scene note for memory/prompts.

    Corrections are already in the verifier's cache (applied to prompt
    context + HUD labels); here we store the human-readable scene so the
    think loop and replies can talk about what was actually seen.
    """
    try:
        if not result or not (result.get("scene") or result.get("objects")
                              or result.get("people") or result.get("corrections")):
            config.log_debug("[vlm] empty result, ignoring")
            return
        parts = []
        if result.get("scene"):
            parts.append(result["scene"].strip())
        for p in result.get("people", []):
            desc = (p.get("appearance") or "").strip()
            name = p.get("name")
            if desc:
                parts.append(f"Person ({name}): {desc}" if name else f"Person: {desc}")
        objs = result.get("objects", []) or []
        if objs:
            parts.append("Things here: " + ", ".join(o for o in objs[:8] if o))
        corr = result.get("corrections", {}) or {}
        if corr:
            parts.append("Corrections: " + ", ".join(
                f"{k}->{v}" for k, v in list(corr.items())[:6]))
        text = " ".join(p for p in parts if p).strip()[:600]
        if not text:
            return
        memory.add(kind="vlm_scene", text=text, counts_as_activity=True, salience=0.35)
        config.log_debug(f"[vlm] scene: {text}")
        try:
            from src.ui import events as _events
            _events.post("vlm", text)
        except Exception:
            pass
    except Exception as e:
        config.log_debug(f"[vlm] store note: {e}")


def _publish_frame(frame) -> None:
    """Downscaled JPEG for the dashboard MJPEG stream (throttled)."""
    global _last_published
    now = time.time()
    if now - _last_published < 0.2:
        return
    _last_published = now
    try:
        small = cv2.resize(frame, (480, 360))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            internal_state.set_camera_frame(bytes(buf))
    except Exception:
        pass


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
    global _last_presence_names, _last_presence_time
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

    use_electron = getattr(config, "USE_ELECTRON", True)
    face_window_name = "Karma"
    curr_dims = [screen_w, screen_h]

    if not use_electron:
        cv2.namedWindow(face_window_name, cv2.WINDOW_NORMAL)
        if getattr(config, "FULLSCREEN_FACE", True):
            try:
                cv2.setWindowProperty(face_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception:
                pass

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

                face_labels, recognized_people, primary_face, all_faces = ([], set(), None, [])
                if face_tracker is not None:
                    face_result = face_tracker.process(frame, memory)
                    # Backward compat with 3-tuple callers/tests.
                    if len(face_result) == 4:
                        face_labels, recognized_people, primary_face, all_faces = face_result
                    else:
                        face_labels, recognized_people, primary_face = face_result
                        all_faces = [primary_face] if primary_face else []
                    recognized_people = set(recognized_people or [])
                    if recognized_people:
                        _last_presence_names = set(recognized_people)
                        _last_presence_time = now
                        memory.set_recognized_people(recognized_people)
                    elif primary_face is None and all_faces == []:
                        # No face at all: clear after grace period, else keep last.
                        if now - _last_presence_time >= PRESENCE_CLEAR_SECONDS:
                            if _last_presence_names:
                                _last_presence_names = set()
                            memory.set_recognized_people(set())
                    # else: face visible but throttled frame -> keep previous presence

                # YOLO `person` + face names -> `Sara` boxes. The HUD, memory,
                # and prompt context then see names instead of `person`.
                try:
                    bboxes = fuse_person_identities(bboxes, all_faces)
                except Exception:
                    pass
                fused_labels = [lbl for lbl, _conf, _box in bboxes]

                # Vision e-stop: worm base has no bump sensors, so a large
                # centered obstacle box halts the drive (throttled).
                try:
                    from src.navigation.explorer import explorer as _explorer
                    from src.navigation.explorer import is_blocked as _is_blocked
                    blocker = _is_blocked(bboxes, frame_w=frame.shape[1],
                                          frame_h=frame.shape[0])
                    if blocker:
                        from src.hardware.drive import drive_base as _drive
                        if _drive.is_moving:
                            _drive.stop()
                            _explorer.note_blocked(blocker)
                            memory.add(kind="obstacle", text=f"Blocked by {blocker}, stopped",
                                       counts_as_activity=True, salience=0.85,
                                       dedup_seconds=getattr(config, "OBSTACLE_COOLDOWN_SECONDS", 3.0))
                except Exception as e:
                    config.log_debug(f"[vision] estop check note: {e}")

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

                current_labels = set(fused_labels + face_labels + hand_labels)
                # Safety net: if fusion missed (no person box) still surface names.
                for p in recognized_people:
                    current_labels.discard("person")
                    current_labels.add(p)

                try:
                    fused_positions = positions_from_bboxes(bboxes)
                except Exception:
                    fused_positions = {}
                spatial_objects = []
                for lbl, _conf, (x1, y1, x2, y2) in bboxes:
                    spatial_objects.append((lbl, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
                for lbl in set(face_labels + hand_labels):
                    if lbl not in fused_positions:
                        # positions may be {label: [(cx,cy)]} (new) or {label: (cx,cy)} (old/mock)
                        pos = positions.get(lbl) if isinstance(positions, dict) else None
                        if isinstance(pos, list) and pos:
                            spatial_objects.append((lbl, pos[0]))
                        elif isinstance(pos, tuple):
                            spatial_objects.append((lbl, pos))
                        else:
                            spatial_objects.append((lbl, (320.0, 240.0)))
                memory.consciousness.update(spatial_objects)

                new_labels = current_labels - last_seen_labels
                for label in new_labels:
                    if getattr(config, "LOG_VISION_TO_CONSOLE", False):
                        print(f"[vision] {label}")
                    memory.add(kind="object", text=label, dedup_seconds=config.OBJECT_DEDUP_SECONDS)
                # One-shot VLM check on genuinely novel YOLO sightings.
                # YOLO keeps tracking every frame; the VLM snapshot runs
                # async and only corrects labels / describes the scene.
                try:
                    _vlm_verifier.maybe_verify(
                        frame, new_labels, recognized_people, now,
                        on_result=lambda res, jb: _store_vlm_result(memory, res, jb),
                    )
                except Exception as e:
                    config.log_debug(f"[vision] vlm submit note: {e}")
                last_seen_labels = current_labels
                _publish_frame(frame)

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

            if not use_electron:
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
                try:
                    shown_bboxes = [
                        (_vlm_verifier.corrections.lookup(lbl), conf, box)
                        for lbl, conf, box in bboxes
                    ]
                except Exception:
                    shown_bboxes = bboxes
                VisionRenderer.draw_objects(annotated, shown_bboxes)
                VisionRenderer.draw_hands(annotated, hand_pts)
                if primary_face:
                    VisionRenderer.draw_face(annotated, primary_face)
                annotated = VisionRenderer.draw_hud(annotated, display_fps, display_fps, is_talking=is_talking)
                cv2.imshow(CAMERA_WINDOW_NAME, annotated)

            if not use_electron or (getattr(config, "SHOW_VISION_WINDOW", False) and has_camera):
                key = cv2.waitKey(1) & 0xFF
                if key in (4, 27, ord('q'), ord('Q')):
                    stop_event.set()
                    break
                elif key in (ord('m'), ord('M')) and not use_electron:
                    if kiosk_manager.is_active():
                        kiosk_manager.close()
                    else:
                        kiosk_manager.open_view("map")
                elif key in (ord('f'), ord('F')) and not use_electron:
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
                            cv2.destroyWindow(CAMERA_WINDOW_NAME)
                        except Exception:
                            pass

            if not use_electron:
                try:
                    if cv2.getWindowProperty(face_window_name, cv2.WND_PROP_VISIBLE) < 1:
                        stop_event.set()
                        break
                except Exception:
                    pass
            else:
                time.sleep(0.01)

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

