"""
Karma Face Renderer & Vision HUD Drawing.
Renders high-FPS, expressive procedural digital companion face and debug vision overlays.
"""
import math
import random
import time
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np

from src.state import internal_state
from src.vision.hand import HAND_CONNECTIONS


class FaceRenderer:
    """
    Renders Karma's full-screen animated digital companion face.
    Features expressive procedural eyes, smooth blinking, gaze tracking,
    audio-reactive mouth lip-sync, mood glow, and dialogue subtitles.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self._last_blink = time.time()
        self._blink_interval = 3.5
        self._blink_duration = 0.16
        self._curr_gaze_x = 0.0
        self._curr_gaze_y = 0.0
        self._idle_gaze_time = time.time()
        self._idle_target_x = 0.0
        self._idle_target_y = 0.0
        self._mouth_phase = 0.0

        # Theme color palettes (BGR format)
        self.palettes: Dict[str, Dict[str, Tuple[int, int, int]]] = {
            "playful": {
                "primary": (240, 220, 0),      # Vivid Cyan
                "glow": (140, 120, 0),
                "accent": (255, 180, 40),
            },
            "curious": {
                "primary": (255, 180, 40),     # Electric Aqua Blue
                "glow": (150, 90, 20),
                "accent": (240, 220, 0),
            },
            "excited": {
                "primary": (50, 210, 255),     # Radiant Amber Gold
                "glow": (20, 110, 140),
                "accent": (100, 240, 255),
            },
            "attentive": {
                "primary": (255, 230, 0),      # Neon Cyan
                "glow": (140, 130, 0),
                "accent": (200, 255, 100),
            },
            "tired": {
                "primary": (220, 150, 180),    # Soft Lavender Violet
                "glow": (120, 70, 90),
                "accent": (180, 120, 150),
            },
            "sad": {
                "primary": (230, 140, 70),     # Muted Slate Blue
                "glow": (120, 60, 30),
                "accent": (200, 110, 50),
            },
            "love": {
                "primary": (180, 105, 255),    # Soft Rose Pink
                "glow": (90, 50, 140),
                "accent": (210, 140, 255),
            },
            "angry": {
                "primary": (60, 70, 255),      # Coral Red
                "glow": (30, 35, 130),
                "accent": (80, 100, 255),
            },
        }

    def _get_theme(self, mood: str) -> Dict[str, Tuple[int, int, int]]:
        mood_key = mood.lower().strip()
        for k in self.palettes:
            if k in mood_key:
                return self.palettes[k]
        return self.palettes["playful"]

    def _update_blinking(self, now: float) -> float:
        """Returns eye height multiplier (0.05 when closed, 1.0 when open)."""
        dt = now - self._last_blink
        if dt > self._blink_interval:
            self._last_blink = now
            self._blink_interval = random.uniform(3.0, 5.5)
            dt = 0.0

        if dt < self._blink_duration:
            prog = dt / self._blink_duration
            # Sine blink curve (0 -> 1 -> 0)
            return max(0.06, 1.0 - math.sin(prog * math.pi))
        return 1.0

    def _update_gaze(self, now: float) -> Tuple[float, float]:
        """Calculates smoothed gaze coordinates with natural saccades."""
        target_x = internal_state.gaze_x
        target_y = internal_state.gaze_y

        # If user is not directly tracked, perform natural idle eye drift
        if not internal_state.is_user_present:
            if now - self._idle_gaze_time > random.uniform(2.0, 4.5):
                self._idle_gaze_time = now
                if random.random() < 0.6:
                    self._idle_target_x = random.uniform(-0.35, 0.35)
                    self._idle_target_y = random.uniform(-0.25, 0.25)
                else:
                    self._idle_target_x = 0.0
                    self._idle_target_y = 0.0
            target_x = self._idle_target_x
            target_y = self._idle_target_y

        # Smooth lerp
        self._curr_gaze_x += (target_x - self._curr_gaze_x) * 0.15
        self._curr_gaze_y += (target_y - self._curr_gaze_y) * 0.15
        return self._curr_gaze_x, self._curr_gaze_y

    def render(self, is_talking: bool = False, is_user_speaking: bool = False,
               fps: float = 0.0, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Renders full-screen face frame."""
        now = time.time()
        w = target_shape[1] if target_shape else self.width
        h = target_shape[0] if target_shape else self.height

        # Create dark obsidian background with subtle radial vignette
        canvas = np.full((h, w, 3), (15, 12, 10), dtype=np.uint8)

        mood = (internal_state.current_emotion or internal_state.mood or "playful").lower()
        theme = self._get_theme(mood)
        primary_color = theme["primary"]
        glow_color = theme["glow"]

        # Gaze and blink state
        blink_mult = self._update_blinking(now)
        gaze_x, gaze_y = self._update_gaze(now)
        gaze_off_x = int(gaze_x * (w * 0.033))
        gaze_off_y = int(gaze_y * (h * 0.05))

        # Check if code snippet display is active
        code_data = internal_state.get_active_code()
        is_coding_mode = (code_data is not None)

        if is_coding_mode:
            code_text, code_lang = code_data
            # Karma's face is sized smaller and pushed to the left 32%
            left_w = int(w * 0.32)
            cx = left_w // 2
            cy = int(h * 0.46)
            eye_w = int(left_w * 0.20)
            eye_h = int(h * 0.18)
            eye_gap = int(left_w * 0.28)
            mouth_w = int(left_w * 0.36)
            mouth_h = int(h * 0.05)

            # Eyes look toward the code snippet on the right
            gaze_off_x += int(eye_w * 0.35)
        else:
            left_w = 0
            cx, cy = w // 2, int(h * 0.44)
            eye_w = int(w * 0.11)
            eye_h = int(h * 0.24)
            eye_gap = int(w * 0.17)
            mouth_w = int(w * 0.18)
            mouth_h = int(h * 0.07)

        # Render Left & Right Eyes
        self._draw_eye(canvas, cx - eye_gap, cy, eye_w, eye_h, blink_mult,
                       gaze_off_x, gaze_off_y, mood, primary_color, glow_color, is_left=True)
        self._draw_eye(canvas, cx + eye_gap, cy, eye_w, eye_h, blink_mult,
                       gaze_off_x, gaze_off_y, mood, primary_color, glow_color, is_left=False)

        # Render Reactive Mouth
        mouth_y = int(cy + eye_h * 0.82)
        self._draw_mouth(canvas, cx, mouth_y, mouth_w, mouth_h,
                         is_talking, is_user_speaking, mood, primary_color, glow_color)

        if is_coding_mode:
            self._draw_code_panel(canvas, left_w, w, h, code_text, code_lang, primary_color)
        else:
            # Render Top HUD & Mood Badges
            self._draw_top_bar(canvas, w, h, mood, is_talking, is_user_speaking, fps, primary_color)

            # Render Active Dialogue Subtitles (bottom overlay)
            self._draw_subtitles(canvas, w, h, primary_color)

            # Render subtle exit hint in bottom right
            cv2.putText(canvas, "Ctrl+D to exit  |  'f' fullscreen  |  'm' kiosk menu",
                        (max(10, w - 440), max(20, h - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 80, 75), 1, cv2.LINE_AA)

        return canvas

    def _draw_code_panel(self, canvas: np.ndarray, left_w: int, width: int, height: int,
                         code_text: str, code_lang: str, primary_color: Tuple[int, int, int]) -> None:
        """Renders center-right IDE code editor card on screen when coding requests are active."""
        x1 = left_w + 14
        x2 = width - 16
        y1 = 18
        y2 = height - 18

        # Divider line between face on left and code in center (soft slate blue)
        cv2.line(canvas, (left_w + 2, 20), (left_w + 2, height - 20), (90, 60, 40), 1, cv2.LINE_AA)

        # Card shadow / background (deep midnight navy blue)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 20, 14), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 160, 20), 2)  # Glowing electric blue border

        # Header bar (rich dark navy slate)
        header_h = 34
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + header_h), (46, 32, 22), -1)
        cv2.line(canvas, (x1, y1 + header_h), (x2, y1 + header_h), (110, 75, 50), 1)

        # 3 Terminal dots (macOS style)
        cv2.circle(canvas, (x1 + 18, y1 + 17), 5, (70, 70, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1 + 34, y1 + 17), 5, (255, 200, 50), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1 + 50, y1 + 17), 5, (90, 220, 100), -1, cv2.LINE_AA)

        # Language title badge (electric cyber blue)
        lang_title = f"{code_lang.upper()} SNIPPET"
        cv2.putText(canvas, lang_title, (x1 + 72, y1 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 185, 40), 1, cv2.LINE_AA)

        # Touch dismiss hint (soft muted ice blue)
        cv2.putText(canvas, "[TAP TO CLOSE]", (x2 - 120, y1 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 160, 120), 1, cv2.LINE_AA)

        # Split and draw code lines with line numbers
        lines = code_text.splitlines()
        line_y = y1 + 56
        max_lines = max(5, (y2 - y1 - 48) // 22)

        for i, line in enumerate(lines[:max_lines]):
            # Line number (steel blue)
            cv2.putText(canvas, f"{i+1:02d}", (x1 + 16, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 120, 80), 1, cv2.LINE_AA)

            # Code text
            line_str = line[:60]
            is_kw = any(line_str.strip().startswith(kw) for kw in ["def ", "class ", "import ", "from ", "return ", "if ", "for ", "while ", "const ", "let ", "function "])
            text_color = (255, 180, 50) if is_kw else (245, 240, 230)  # Electric ice blue keywords

            cv2.putText(canvas, line_str, (x1 + 46, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)
            line_y += 22

        if len(lines) > max_lines:
            cv2.putText(canvas, f"... [{len(lines) - max_lines} more lines]", (x1 + 46, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 185, 40), 1, cv2.LINE_AA)

    def _draw_eye(self, canvas: np.ndarray, cx: int, cy: int, w: int, h: int,
                  blink: float, gx: int, gy: int, mood: str,
                  color: Tuple[int, int, int], glow: Tuple[int, int, int], is_left: bool) -> None:
        """Draws an expressive, anti-aliased glowing eye with mood styling."""
        cur_h = max(4, int(h * blink))

        # Mood-specific eye expressions
        if "happy" in mood or "playful" in mood:
            # Curved happy eyes (^ ^)
            thickness = max(4, int(w * 0.14))
            axes = (w // 2, max(6, cur_h // 2))
            cv2.ellipse(canvas, (cx + gx, cy + gy), axes, 0, 190, 350, glow, thickness + 4, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx + gx, cy + gy), axes, 0, 190, 350, color, thickness, cv2.LINE_AA)
            return

        if "tired" in mood or "sleepy" in mood:
            # Sleepy relaxed horizontal slits (- -)
            thickness = max(4, int(w * 0.12))
            pt1 = (cx - w // 2 + gx, cy + gy)
            pt2 = (cx + w // 2 + gx, cy + gy + (4 if is_left else -4))
            cv2.line(canvas, pt1, pt2, glow, thickness + 4, cv2.LINE_AA)
            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
            return

        if "excited" in mood or "surprised" in mood:
            # Wide open round eyes (O O)
            r = int(min(w, cur_h) * 0.55)
            cv2.circle(canvas, (cx + gx, cy + gy), r + 4, glow, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx + gx, cy + gy), r, color, -1, cv2.LINE_AA)
            # Center pupil sparkle
            cv2.circle(canvas, (cx + gx + r // 3, cy + gy - r // 3), max(3, r // 3), (255, 255, 255), -1, cv2.LINE_AA)
            return

        # Default / Attentive / Curious Stadium Capsule Eyes
        rx = w // 2
        ry = cur_h // 2

        # Draw soft outer glow
        if cur_h > 12:
            cv2.ellipse(canvas, (cx + gx, cy + gy), (rx + 4, ry + 4), 0, 0, 360, glow, -1, cv2.LINE_AA)

        # Draw main eye body
        cv2.ellipse(canvas, (cx + gx, cy + gy), (rx, ry), 0, 0, 360, color, -1, cv2.LINE_AA)

        # Specular gloss highlight (white shine reflection)
        if cur_h > 20:
            spec_x = cx + gx + int(rx * 0.35)
            spec_y = cy + gy - int(ry * 0.35)
            spec_r = max(3, int(min(rx, ry) * 0.28))
            cv2.circle(canvas, (spec_x, spec_y), spec_r, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_mouth(self, canvas: np.ndarray, cx: int, cy: int, w: int, h: int,
                    is_talking: bool, is_user_speaking: bool, mood: str,
                    color: Tuple[int, int, int], glow: Tuple[int, int, int]) -> None:
        """Draws dynamic lip-sync mouth or resting pleasant smile."""
        if is_talking:
            # Dynamic talking mouth waveform
            self._mouth_phase += 0.35
            pts = []
            steps = 18
            for i in range(steps):
                t = i / (steps - 1)
                x = int(cx - w // 2 + t * w)
                amp = math.sin(t * math.pi) * (h * 0.85) * (0.6 + 0.4 * math.sin(self._mouth_phase + t * 4.0))
                y = int(cy + amp)
                pts.append((x, y))

            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [pts_np], False, glow, 7, cv2.LINE_AA)
            cv2.polylines(canvas, [pts_np], False, color, 4, cv2.LINE_AA)
            cv2.polylines(canvas, [pts_np], False, (255, 255, 255), 2, cv2.LINE_AA)

        elif is_user_speaking:
            # Listening subtle wave
            pts = []
            steps = 14
            for i in range(steps):
                t = i / (steps - 1)
                x = int(cx - w // 2 + t * w)
                y = int(cy + math.sin(time.time() * 6.0 + t * math.pi * 2) * (h * 0.35))
                pts.append((x, y))
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [pts_np], False, (0, 255, 180), 3, cv2.LINE_AA)

        else:
            # Resting subtle curved smile
            axes = (w // 2, max(6, h // 2))
            cv2.ellipse(canvas, (cx, cy - h // 3), axes, 0, 20, 160, glow, 6, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx, cy - h // 3), axes, 0, 20, 160, color, 3, cv2.LINE_AA)

    def _draw_top_bar(self, canvas: np.ndarray, w: int, h: int, mood: str,
                      is_talking: bool, is_user_speaking: bool, fps: float,
                      color: Tuple[int, int, int]) -> None:
        """Renders top header with companion identity, pulse indicator, mood tag, and energy meters."""
        bar_h = max(55, int(h * 0.072))
        mid_y = bar_h // 2

        # Top background panel flush with the top edge
        cv2.rectangle(canvas, (0, 0), (w, bar_h), (10, 8, 7), -1)
        cv2.line(canvas, (0, bar_h), (w, bar_h), (35, 30, 28), 1)

        # Companion Title & Pulse Dot
        title_scale = max(0.75, min(1.1, w / 1600.0 * 0.90))
        cv2.putText(canvas, "KARMA", (28, mid_y + 8), cv2.FONT_HERSHEY_DUPLEX, title_scale, (255, 255, 255), 2, cv2.LINE_AA)

        (tw, _), _ = cv2.getTextSize("KARMA", cv2.FONT_HERSHEY_DUPLEX, title_scale, 2)
        dot_x = 28 + tw + 20
        pulse_color = (50, 220, 255) if is_talking else ((0, 255, 150) if is_user_speaking else (0, 220, 100))
        pulse_r = max(5, int(6 + 2 * math.sin(time.time() * 5.0)))
        cv2.circle(canvas, (dot_x, mid_y), pulse_r, pulse_color, -1, cv2.LINE_AA)

        # State / Mood Tag Pill
        tag_text = "SPEAKING" if is_talking else ("LISTENING" if is_user_speaking else mood.upper())
        pill_scale = max(0.48, min(0.65, title_scale * 0.65))
        (tag_w, tag_th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, pill_scale, 1)
        pill_w = tag_w + 24
        pill_h = tag_th + 16
        pill_x = dot_x + 24
        pill_y = mid_y - pill_h // 2
        cv2.rectangle(canvas, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (30, 25, 22), -1)
        cv2.rectangle(canvas, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, tag_text, (pill_x + 12, pill_y + tag_th + 6), cv2.FONT_HERSHEY_SIMPLEX, pill_scale, color, 1, cv2.LINE_AA)

        # Energy & Curiosity Meters (Top Right)
        meter_w = max(80, int(w * 0.065))
        meter_total_w = meter_w * 2 + 150
        meter_x = max(pill_x + pill_w + 30, w - meter_total_w - 30)
        energy_pct = internal_state.energy
        curiosity_pct = internal_state.curiosity

        bar_thick = max(8, int(bar_h * 0.16))
        bar_y1 = mid_y - bar_thick // 2
        bar_y2 = bar_y1 + bar_thick

        # Energy bar
        cv2.putText(canvas, "NRG", (meter_x, mid_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 150, 140), 1, cv2.LINE_AA)
        bx1 = meter_x + 40
        cv2.rectangle(canvas, (bx1, bar_y1), (bx1 + meter_w, bar_y2), (40, 35, 30), -1)
        cv2.rectangle(canvas, (bx1, bar_y1), (bx1 + int(meter_w * energy_pct), bar_y2), (0, 220, 255), -1)

        # Curiosity bar
        cx_lbl = bx1 + meter_w + 24
        cv2.putText(canvas, "CUR", (cx_lbl, mid_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 150, 140), 1, cv2.LINE_AA)
        bx2 = cx_lbl + 40
        cv2.rectangle(canvas, (bx2, bar_y1), (bx2 + meter_w, bar_y2), (40, 35, 30), -1)
        cv2.rectangle(canvas, (bx2, bar_y1), (bx2 + int(meter_w * curiosity_pct), bar_y2), (255, 180, 0), -1)


    def _draw_subtitles(self, canvas: np.ndarray, w: int, h: int, color: Tuple[int, int, int]) -> None:
        """Renders dialogue subtitles overlay with smooth word wrapping."""
        sub = internal_state.get_active_subtitle(max_age=7.0)
        if not sub:
            return

        speaker, text = sub
        if not text:
            return

        # Clean text
        display_text = f"{speaker}: \"{text}\""
        if len(display_text) > 110:
            display_text = display_text[:107] + "..."

        # Calculate bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)

        pill_w = min(w - 60, text_w + 36)
        pill_h = text_h + 24
        pill_x = (w - pill_w) // 2
        pill_y = h - 75

        # Rounded translucent subtitle pill
        overlay = canvas.copy()
        cv2.rectangle(overlay, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (18, 14, 12), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (55, 48, 42), 1, cv2.LINE_AA)

        text_x = pill_x + 18
        text_y = pill_y + text_h + 8
        text_color = color if speaker == "Karma" else (220, 255, 220)
        cv2.putText(canvas, display_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


class VisionRenderer:
    """Renders debug camera HUD, bounding boxes, face tracking, and hand landmarks."""

    @staticmethod
    def draw_hud(frame: np.ndarray, fps: float, display_fps: float, is_talking: bool = False) -> np.ndarray:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)

        expression = internal_state.get_expression(is_talking=is_talking)
        mood = (internal_state.current_emotion or internal_state.mood).upper()
        cv2.putText(frame, f"Karma Vision [Debug]: {expression} [{mood}]", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

        fps_text = f"{display_fps:.1f} FPS"
        cv2.putText(frame, fps_text, (w - 110, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2, cv2.LINE_AA)

        return frame

    @staticmethod
    def draw_hands(frame: np.ndarray, hand_landmarks: List[List[Tuple[int, int]]]) -> None:
        for pts in hand_landmarks:
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(pts) and p2 < len(pts):
                    cv2.line(frame, pts[p1], pts[p2], (0, 255, 255), 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 150, 255), -1, cv2.LINE_AA)

    @staticmethod
    def draw_face(frame: np.ndarray, face_info: Tuple[int, int, int, int, str, str]) -> None:
        x, y, w, h, name, emotion = face_info
        color = (255, 165, 0) if name != "Face" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{name} ({emotion})", (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    @staticmethod
    def draw_objects(frame: np.ndarray, bboxes: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> None:
        for name, conf, (x1, y1, x2, y2) in bboxes:
            if name == "person":
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 100), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1, cv2.LINE_AA)
