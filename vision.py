"""
Webcam -> real-time high-FPS object detection + face tracking + 3D hand/finger tracking + VLM Scene Understanding -> working memory.
Optimized for 60-80+ FPS performance on Apple Silicon MPS GPU.
"""
import os
import threading
import time
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

import config
from vlm_analyzer import analyze_scene_vlm

# 21 Hand joint skeletal connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]


def run_vision(memory, stop_event, speaking_event=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[vision] could not open webcam -- check macOS camera permissions "
              "for your terminal app in System Settings > Privacy & Security.")
        return

    # Set camera resolution for high FPS performance (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    model = YOLO(config.YOLO_MODEL)
    device = config.YOLO_DEVICE

    # Load face and smile cascades for gaze & emotion tracking
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    # Initialize MediaPipe 3D Hand & Finger Landmarker
    hand_detector = None
    task_path = os.path.join(config.BASE_DIR, "models", "hand_landmarker.task")
    if os.path.exists(task_path):
        try:
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
            hand_detector = mp_vision.HandLandmarker.create_from_options(opts)
            print("[vision] MediaPipe 3D Hand & Finger Tracking initialized!")
        except Exception as e:
            print(f"[vision] HandLandmarker init warning: {e}")

    print(f"[vision] running high-FPS pipeline on device={device}")
    show_window = getattr(config, "SHOW_VISION_WINDOW", True)
    log_console = getattr(config, "LOG_VISION_TO_CONSOLE", True)
    enable_vlm = getattr(config, "ENABLE_VLM_VISION", True)

    # Shared variable for VLM worker thread
    latest_raw_frame = None
    frame_lock = threading.Lock()

    def vlm_worker():
        last_vlm_time = 0
        vlm_interval = getattr(config, "VLM_POLL_SECONDS", 4)
        while not stop_event.is_set():
            time.sleep(1)
            now = time.time()
            if now - last_vlm_time < vlm_interval:
                continue

            frame_snapshot = None
            with frame_lock:
                if latest_raw_frame is not None:
                    frame_snapshot = latest_raw_frame.copy()

            if frame_snapshot is not None:
                last_vlm_time = now
                desc = analyze_scene_vlm(frame_snapshot)
                if desc:
                    if log_console:
                        print(f"[vision VLM] {desc}")
                    memory.add(
                        kind="object",
                        text=f"scene analysis: {desc}",
                        dedup_seconds=10,
                    )

    if enable_vlm:
        vlm_thread = threading.Thread(target=vlm_worker, daemon=True)
        vlm_thread.start()

    # FPS performance counter variables
    fps_time = time.time()
    frame_count = 0
    display_fps = 0.0
    last_seen_labels = set()
    previous_objects = set()
    previous_positions = {}
    user_x_history = deque(maxlen=10)
    # Per-label startle cooldown: label → last startle timestamp
    # Prevents the same object from triggering startle dozens of times per second
    _startle_cooldowns = {}
    STARTLE_COOLDOWN_SECONDS = 10  # min seconds between startles for the same label

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Calculate FPS
            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                display_fps = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now

            with frame_lock:
                latest_raw_frame = frame.copy()

            # High-FPS YOLO object detection
            results = model.predict(
                frame, device=device, conf=getattr(config, "YOLO_CONFIDENCE", 0.50), verbose=False
            )

            # Get annotated frame with YOLO bounding boxes and labels
            annotated_frame = results[0].plot() if show_window else frame

            labels = set()
            current_positions = {}
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    labels.add(label)
                    
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    current_positions[label] = (float(cx), float(cy))

                    if label == "person":
                        user_x_history.append(float(cx))

            # Update Global Workspace Consciousness
            memory.consciousness.update(current_positions.items(), None, time.time())

            # Object movement / novelty consciousness
            for label, pos in current_positions.items():
                if label not in previous_objects:
                    memory.add(kind="conscious_trigger", text=f"FOCUS: New {label} appeared at {pos}", salience=1.0)
                elif label in previous_positions:
                    dist = np.linalg.norm(np.array(pos) - np.array(previous_positions[label]))
                    if dist > 30:
                        memory.add(kind="conscious_trigger", text=f"FOCUS: {label} is moving toward me", salience=0.8)
            
            previous_objects = set(current_positions.keys())
            previous_positions = current_positions

            # Prediction engine on user position
            if len(user_x_history) > 5:
                predicted_x = np.polyval(np.polyfit(range(5), list(user_x_history)[-5:], 1), 5)
                actual_x = user_x_history[-1]
                if abs(actual_x - predicted_x) > 50:
                    memory.add(kind="conscious_trigger", text="The user suddenly shifted position unexpectedly!", salience=1.0)
                    user_x_history.clear() # Reset to avoid spamming

            # Optimized Face detection (2x downsampled grayscale for 4x speedup)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))

            if len(faces) > 0:
                # Scale face coords back to original size
                sx, sy, sw, sh = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = sx * 2, sy * 2, sw * 2, sh * 2
                
                # Check for smile / mood emotion
                face_roi = gray[y:y+h, x:x+w]
                smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20)
                emotion = "smiling" if len(smiles) > 0 else "neutral"

                gaze_label = f"looking at you ({emotion})"
                labels.add(gaze_label)

                if show_window:
                    color = (0, 255, 0)  # Bright Green for direct gaze
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        annotated_frame,
                        f"Face ({emotion}, looking at you)",
                        (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
            else:
                # Check for profile face
                profiles = profile_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
                if len(profiles) > 0:
                    sx, sy, sw, sh = max(profiles, key=lambda f: f[2] * f[3])
                    x, y, w, h = sx * 2, sy * 2, sw * 2, sh * 2
                    gaze_label = "looking away"
                    labels.add(gaze_label)

                    if show_window:
                        color = (0, 255, 255)  # Yellow for turned head
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(
                            annotated_frame,
                            "Face (looking away)",
                            (x, max(25, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )

            # MediaPipe 3D Hand & Finger Tracking (runs on alternating frames to reduce thermal load)
            if hand_detector and (frame_count % 2 == 0):
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    hand_results = hand_detector.detect(mp_image)

                    if hand_results and hand_results.hand_landmarks:
                        h_h, h_w = frame.shape[:2]
                        for idx, hand_landmarks in enumerate(hand_results.hand_landmarks):
                            pts = [(int(lm.x * h_w), int(lm.y * h_h)) for lm in hand_landmarks]

                            if show_window:
                                # Draw 20 skeletal connections in bright cyan
                                for p1_idx, p2_idx in HAND_CONNECTIONS:
                                    cv2.line(annotated_frame, pts[p1_idx], pts[p2_idx], (255, 255, 0), 2)
                                # Draw 21 joint landmark keypoints in red
                                for pt in pts:
                                    cv2.circle(annotated_frame, pt, 4, (0, 0, 255), -1)

                            # Gesture & finger orientation detection
                            index_tip, index_pip = pts[8], pts[6]
                            middle_tip, middle_pip = pts[12], pts[10]
                            ring_tip, ring_pip = pts[16], pts[14]
                            pinky_tip, pinky_pip = pts[20], pts[18]

                            index_up = index_tip[1] < index_pip[1]
                            middle_up = middle_tip[1] < middle_pip[1]
                            ring_up = ring_tip[1] < ring_pip[1]
                            pinky_up = pinky_tip[1] < pinky_pip[1]

                            if index_up and not middle_up and not ring_up and not pinky_up:
                                gesture = "pointing finger"
                            elif index_up and middle_up and not ring_up and not pinky_up:
                                gesture = "peace sign"
                            elif index_up and middle_up and ring_up and pinky_up:
                                gesture = "open hand"
                            elif not index_up and not middle_up and not ring_up and not pinky_up:
                                gesture = "holding object / fist"
                            else:
                                gesture = "hand gesture"

                            labels.add(f"hand ({gesture})")

                            if show_window:
                                wrist_pt = pts[0]
                                cv2.putText(
                                    annotated_frame,
                                    f"Hand: {gesture}",
                                    (wrist_pt[0], max(35, wrist_pt[1] + 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 0),
                                    2,
                                )
                except Exception:
                    pass

            # Environmental Startle Response: detect sudden appearance of new objects/gestures
            new_labels = labels - last_seen_labels
            startle_items = [l for l in new_labels if "looking" not in l and "face" not in l]
            if startle_items and len(last_seen_labels) > 0:
                # Apply per-label cooldown so the same object doesn't startle every frame
                now_t = time.time()
                cooled = [
                    l for l in startle_items
                    if now_t - _startle_cooldowns.get(l, 0) >= STARTLE_COOLDOWN_SECONDS
                ]
                if cooled:
                    for l in cooled:
                        _startle_cooldowns[l] = now_t
                    item_str = ", ".join(sorted(cooled))
                    if log_console:
                        print(f"[vision startle] suddenly noticed: {item_str}")
                    memory.add(
                        kind="urgent_observation",
                        text=f"suddenly noticed {item_str}",
                        counts_as_activity=True,
                        salience=1.0,
                    )
            last_seen_labels = labels

            # Add detections to memory
            for label in sorted(labels):
                memory.add(
                    kind="object",
                    text=f"saw a {label}",
                    dedup_seconds=config.OBJECT_DEDUP_SECONDS,
                )

            # Display live camera window with real-time FPS overlay
            if show_window:
                try:
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {display_fps:.1f}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("Ambient Agent Vision (YOLO + Face Tracking)", annotated_frame)
                except Exception as e:
                    pass

            # --- AI Face window ---
            try:
                from state import internal_state
                face_img = np.zeros((300, 400, 3), dtype=np.uint8)
                is_talking = internal_state.is_playing_audio
                face_str = internal_state.get_expression(is_talking)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                font_thickness = 4
                text_size = cv2.getTextSize(face_str, font, font_scale, font_thickness)[0]
                text_x = (400 - text_size[0]) // 2
                text_y = (300 + text_size[1]) // 2
                cv2.putText(face_img, face_str, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                cv2.imshow("Karma Face", face_img)
            except Exception as e:
                pass

            # waitKey handles all active OpenCV windows
            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[vision] window closed by user")
                    show_window = False
                    cv2.destroyAllWindows()
            except cv2.error:
                print("[vision] Note: macOS Cocoa GUI requires main thread for window display. Vision continues headlessly with console logging.")
                show_window = False
            except Exception as e:
                print(f"[vision] window display error: {e}")
                show_window = False

            if getattr(config, "VISION_POLL_SECONDS", 0.001) > 0:
                stop_event.wait(config.VISION_POLL_SECONDS)
    finally:
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        cap.release()
