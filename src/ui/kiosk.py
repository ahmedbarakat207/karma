import glob
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src import config
from src.hardware.neck import neck_actuator
from src.memory.store import MemoryStore

DATA_DIR = os.path.abspath("data")
MAPS_DIR = os.path.join(DATA_DIR, "maps")
STUDENT_APPS_FILE = os.path.join(DATA_DIR, "student_apps.json")
ACHIEVEMENTS_FILE = os.path.join(DATA_DIR, "achievements.json")


def _load_json(path: str, fallback=None):
    if not os.path.exists(path):
        return fallback or []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        config.log_debug(f"[kiosk] error reading {path}: {e}")
        return fallback or []


class KioskManager:
    def __init__(self, store: Optional[MemoryStore] = None):
        self.active_view: str = "face"
        self.store = store
        self.current_floor_idx: int = 0
        self.selected_doc: Optional[str] = None
        self.doc_page_idx: int = 0
        self.apps_scroll_idx: int = 0
        self.achievements_scroll_idx: int = 0

        self.student_apps: List[Dict[str, Any]] = []
        self.achievements: List[Dict[str, Any]] = []
        self.floor_maps: List[Tuple[str, str]] = []
        self.indexed_docs: List[Dict[str, Any]] = []
        self.doc_chunks: List[str] = []

        self.reload_all_data()

    def reload_all_data(self) -> None:
        self.student_apps = _load_json(STUDENT_APPS_FILE)
        self.achievements = _load_json(ACHIEVEMENTS_FILE)

        self.floor_maps = []
        if os.path.isdir(MAPS_DIR):
            files = sorted(glob.glob(os.path.join(MAPS_DIR, "*.jpg")) + glob.glob(os.path.join(MAPS_DIR, "*.png")))
            for p in files:
                label = os.path.splitext(os.path.basename(p))[0].replace("_", " ").title()
                self.floor_maps.append((label, p))

        self.reload_documents()

    def reload_documents(self) -> None:
        if self.store is None:
            try:
                self.store = MemoryStore()
            except Exception:
                return

        try:
            self.indexed_docs = self.store.list_sources(kind="document")
            if self.indexed_docs and self.selected_doc is None:
                self.select_document(self.indexed_docs[0]["source"])
        except Exception as e:
            config.log_debug(f"[kiosk] error loading docs: {e}")
            self.indexed_docs = []

    def select_document(self, doc_name: str) -> None:
        self.selected_doc = doc_name
        self.doc_page_idx = 0
        self.doc_chunks = []
        if not self.store:
            return
        try:
            rows = self.store.db.execute(
                "SELECT text FROM memories WHERE kind = 'document' AND source = ? ORDER BY id ASC",
                (doc_name,)
            ).fetchall()
            self.doc_chunks = [r[0] for r in rows]
        except Exception as e:
            config.log_debug(f"[kiosk] error loading chunks for {doc_name}: {e}")

    def open_view(self, view_name: str, floor_idx: Optional[int] = None) -> None:
        if view_name not in ("docs", "map", "apps", "achievements", "face"):
            view_name = "map"

        self.active_view = view_name
        if floor_idx is not None and 0 <= floor_idx < len(self.floor_maps):
            self.current_floor_idx = floor_idx

        if view_name != "face":
            self.reload_all_data()
            neck_actuator.tilt_to_kiosk()
        else:
            neck_actuator.tilt_to_face()

    def close(self) -> None:
        self.open_view("face")

    def is_active(self) -> bool:
        return self.active_view != "face"

    def handle_touch(self, x: int, y: int, screen_w: int, screen_h: int) -> bool:
        if self.active_view == "face":
            if x >= (screen_w - 140) and y <= 55:
                self.open_view("map")
                return True
            return False

        # top nav bar
        if y <= 50:
            if x >= (screen_w - 95):
                self.close()
            elif (screen_w - 240) <= x < (screen_w - 100):
                self.active_view = "achievements"
            elif (screen_w - 385) <= x < (screen_w - 245):
                self.active_view = "apps"
            elif (screen_w - 515) <= x < (screen_w - 390):
                self.active_view = "map"
            elif (screen_w - 640) <= x < (screen_w - 520):
                self.active_view = "docs"
            return True

        if self.active_view == "map":
            if y >= (screen_h - 60):
                tab_w = 120
                for idx in range(len(self.floor_maps)):
                    bx1 = 30 + idx * (tab_w + 12)
                    if bx1 <= x <= bx1 + tab_w:
                        self.current_floor_idx = idx
                        return True

        elif self.active_view == "docs":
            if x <= 240 and y >= 70:
                idx = (y - 70) // 44
                if 0 <= idx < len(self.indexed_docs):
                    self.select_document(self.indexed_docs[idx]["source"])
                    return True
            elif y >= (screen_h - 55):
                if 420 <= x <= 530:
                    self.doc_page_idx = max(0, self.doc_page_idx - 1)
                    return True
                elif 550 <= x <= 660:
                    self.doc_page_idx = min(len(self.doc_chunks) - 1, self.doc_page_idx + 1)
                    return True

        elif self.active_view == "apps":
            if y >= (screen_h - 55):
                if (screen_w - 260) <= x <= (screen_w - 150):
                    self.apps_scroll_idx = max(0, self.apps_scroll_idx - 1)
                    return True
                elif (screen_w - 140) <= x <= (screen_w - 30):
                    self.apps_scroll_idx = min(max(0, len(self.student_apps) - 3), self.apps_scroll_idx + 1)
                    return True

        elif self.active_view == "achievements":
            if y >= (screen_h - 55):
                if (screen_w - 260) <= x <= (screen_w - 150):
                    self.achievements_scroll_idx = max(0, self.achievements_scroll_idx - 1)
                    return True
                elif (screen_w - 140) <= x <= (screen_w - 30):
                    self.achievements_scroll_idx = min(max(0, len(self.achievements) - 3), self.achievements_scroll_idx + 1)
                    return True

        return True

    def render_overlay_button(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        bx1, by1 = w - 135, 14
        bx2, by2 = w - 15, 48
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (28, 34, 42), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 220, 255), 2)
        cv2.putText(frame, ":: MENU", (bx1 + 18, by1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

    def render_kiosk(self, width: int = 800, height: int = 480) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (18, 22, 28)

        self._render_top_navbar(canvas, width)

        if self.active_view == "map":
            self._render_map_view(canvas, width, height)
        elif self.active_view == "docs":
            self._render_docs_view(canvas, width, height)
        elif self.active_view == "apps":
            self._render_apps_view(canvas, width, height)
        elif self.active_view == "achievements":
            self._render_achievements_view(canvas, width, height)

        return canvas

    def _render_top_navbar(self, canvas: np.ndarray, width: int) -> None:
        cv2.rectangle(canvas, (0, 0), (width, 52), (28, 34, 44), -1)
        cv2.line(canvas, (0, 52), (width, 52), (50, 65, 85), 1)
        cv2.putText(canvas, "KARMA // KIOSK", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

        tabs = [
            ("DOCS",   "docs",         width - 640, width - 525),
            ("MAP",    "map",          width - 515, width - 395),
            ("APPS",   "apps",         width - 385, width - 250),
            ("AWARDS", "achievements", width - 240, width - 105),
        ]

        for label, key, x1, x2 in tabs:
            sel = (self.active_view == key)
            cv2.rectangle(canvas, (x1, 10), (x2, 44), (0, 180, 230) if sel else (38, 46, 58), -1)
            cv2.rectangle(canvas, (x1, 10), (x2, 44), (0, 220, 255) if sel else (60, 75, 95), 1)
            tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)[0]
            tx = x1 + (x2 - x1 - tsize[0]) // 2
            color = (15, 20, 25) if sel else (200, 215, 230)
            cv2.putText(canvas, label, (tx, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

        bx1 = width - 95
        cv2.rectangle(canvas, (bx1, 10), (width - 15, 44), (45, 30, 35), -1)
        cv2.rectangle(canvas, (bx1, 10), (width - 15, 44), (80, 100, 255), 1)
        cv2.putText(canvas, "EXIT", (bx1 + 22, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 150, 255), 2, cv2.LINE_AA)

    def _render_map_view(self, canvas: np.ndarray, width: int, height: int) -> None:
        if not self.floor_maps:
            cv2.putText(canvas, "No maps found in data/maps/", (60, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 160, 180), 2, cv2.LINE_AA)
            return

        label, path = self.floor_maps[self.current_floor_idx]
        img = cv2.imread(path)

        if img is not None:
            avail_w = width - 40
            avail_h = height - 125
            mh, mw = img.shape[:2]
            scale = min(avail_w / mw, avail_h / mh)
            nw, nh = int(mw * scale), int(mh * scale)
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            ox = (width - nw) // 2
            oy = 60 + (avail_h - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized
            cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + nw + 1, oy + nh + 1), (0, 200, 255), 1)

        tab_w = 125
        for idx, (flabel, _) in enumerate(self.floor_maps):
            bx1 = 30 + idx * (tab_w + 12)
            bx2 = bx1 + tab_w
            by1 = height - 52
            by2 = height - 14
            sel = (idx == self.current_floor_idx)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 180, 230) if sel else (32, 40, 52), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 220, 255) if sel else (60, 75, 95), 1)
            tsize = cv2.getTextSize(flabel[:14], cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            tx = bx1 + (tab_w - tsize[0]) // 2
            cv2.putText(canvas, flabel[:14], (tx, by1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 20, 25) if sel else (210, 225, 240), 1, cv2.LINE_AA)

    def _render_docs_view(self, canvas: np.ndarray, width: int, height: int) -> None:
        panel_w = 230
        cv2.rectangle(canvas, (15, 62), (panel_w, height - 15), (24, 30, 38), -1)
        cv2.rectangle(canvas, (15, 62), (panel_w, height - 15), (45, 55, 70), 1)
        cv2.putText(canvas, "INDEXED DOCUMENTS", (26, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        if not self.indexed_docs:
            cv2.putText(canvas, "No PDFs in RAG yet.", (26, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 140, 160), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Run: chat.py --pdf", (26, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 110, 130), 1, cv2.LINE_AA)
        else:
            for i, doc in enumerate(self.indexed_docs[:7]):
                dy = 102 + i * 42
                sel = (doc["source"] == self.selected_doc)
                cv2.rectangle(canvas, (20, dy), (panel_w - 6, dy + 36), (40, 52, 68) if sel else (28, 34, 44), -1)
                cv2.rectangle(canvas, (20, dy), (panel_w - 6, dy + 36), (0, 220, 255) if sel else (45, 55, 70), 1)
                label = doc["source"]
                if len(label) > 18:
                    label = label[:16] + ".."
                cv2.putText(canvas, label, (28, dy + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 240, 255), 1, cv2.LINE_AA)

        rx1, rx2 = panel_w + 15, width - 15
        ry1, ry2 = 62, height - 15
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (24, 30, 38), -1)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (45, 55, 70), 1)

        title = f"EXCERPT VIEWER // {(self.selected_doc or 'Document Reader').upper()}"
        cv2.putText(canvas, title, (rx1 + 15, ry1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        if not self.doc_chunks:
            cv2.putText(canvas, "Select a document to read its passages.", (rx1 + 20, ry1 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 180), 1, cv2.LINE_AA)
        else:
            text = self.doc_chunks[self.doc_page_idx]
            for j, line in enumerate(self._wrap_text(text, 60)[:13]):
                cv2.putText(canvas, line, (rx1 + 16, ry1 + 60 + j * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 225, 240), 1, cv2.LINE_AA)

            cv2.putText(canvas, f"Chunk {self.doc_page_idx + 1} of {len(self.doc_chunks)}",
                        (rx1 + 20, ry2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 145, 170), 1, cv2.LINE_AA)

            cv2.rectangle(canvas, (rx2 - 220, ry2 - 40), (rx2 - 120, ry2 - 10), (38, 48, 62), -1)
            cv2.rectangle(canvas, (rx2 - 220, ry2 - 40), (rx2 - 120, ry2 - 10), (0, 200, 255), 1)
            cv2.putText(canvas, "PREV", (rx2 - 195, ry2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (rx2 - 105, ry2 - 40), (rx2 - 10, ry2 - 10), (38, 48, 62), -1)
            cv2.rectangle(canvas, (rx2 - 105, ry2 - 40), (rx2 - 10, ry2 - 10), (0, 200, 255), 1)
            cv2.putText(canvas, "NEXT", (rx2 - 80, ry2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    def _render_apps_view(self, canvas: np.ndarray, width: int, height: int) -> None:
        cv2.putText(canvas, "STUDENT INNOVATION APPS & PROJECTS", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        if not self.student_apps:
            cv2.putText(canvas, "No apps loaded. Add entries to data/student_apps.json", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 150, 170), 1, cv2.LINE_AA)
            return

        card_h, card_w = 100, width - 60
        for i, app in enumerate(self.student_apps[self.apps_scroll_idx:self.apps_scroll_idx + 3]):
            cy1 = 105 + i * (card_h + 14)
            cy2 = cy1 + card_h
            cv2.rectangle(canvas, (30, cy1), (30 + card_w, cy2), (26, 32, 42), -1)
            cv2.rectangle(canvas, (30, cy1), (30 + card_w, cy2), (50, 65, 85), 1)
            cv2.rectangle(canvas, (30, cy1), (36, cy2), (0, 220, 255), -1)

            name = app.get("name", "Unnamed Project")
            cv2.putText(canvas, name, (50, cy1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 250, 255), 2, cv2.LINE_AA)
            badge = f"[{app.get('category', 'Project').upper()}]"
            cv2.putText(canvas, badge, (50 + len(name) * 12 + 10, cy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Creator: {app.get('author', 'Anonymous')}", (50, cy1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 170, 190), 1, cv2.LINE_AA)
            desc = app.get("description", "")
            if len(desc) > 85:
                desc = desc[:82] + "..."
            cv2.putText(canvas, desc, (50, cy1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 215, 230), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"• {app.get('status', 'Active')}", (30 + card_w - 95, cy1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 120), 1, cv2.LINE_AA)

        if len(self.student_apps) > 3:
            n = len(self.student_apps)
            s = self.apps_scroll_idx
            cv2.putText(canvas, f"Showing {s + 1}-{min(n, s + 3)} of {n}", (35, height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 140, 160), 1, cv2.LINE_AA)
            self._scroll_buttons(canvas, width, height)

    def _render_achievements_view(self, canvas: np.ndarray, width: int, height: int) -> None:
        cv2.putText(canvas, "ROBOT & LAB MILESTONES // ACHIEVEMENTS", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        if not self.achievements:
            cv2.putText(canvas, "No achievements in data/achievements.json", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 150, 170), 1, cv2.LINE_AA)
            return

        card_h, card_w = 100, width - 60
        for i, ach in enumerate(self.achievements[self.achievements_scroll_idx:self.achievements_scroll_idx + 3]):
            cy1 = 105 + i * (card_h + 14)
            cy2 = cy1 + card_h
            cv2.rectangle(canvas, (30, cy1), (30 + card_w, cy2), (26, 32, 42), -1)
            cv2.rectangle(canvas, (30, cy1), (30 + card_w, cy2), (60, 75, 95), 1)
            cv2.rectangle(canvas, (30, cy1), (36, cy2), (50, 210, 255), -1)

            cv2.putText(canvas, ach.get("badge", "MILESTONE"), (50, cy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 210, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, ach.get("title", "Achievement"), (50, cy1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 250, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, ach.get("description", ""), (50, cy1 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"✓ {ach.get('date', 'Unlocked')}", (30 + card_w - 120, cy1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 120), 1, cv2.LINE_AA)

        if len(self.achievements) > 3:
            n = len(self.achievements)
            s = self.achievements_scroll_idx
            cv2.putText(canvas, f"Showing {s + 1}-{min(n, s + 3)} of {n}", (35, height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 140, 160), 1, cv2.LINE_AA)
            self._scroll_buttons(canvas, width, height)

    def _scroll_buttons(self, canvas: np.ndarray, width: int, height: int) -> None:
        cv2.rectangle(canvas, (width - 230, height - 44), (width - 130, height - 12), (35, 45, 58), -1)
        cv2.putText(canvas, "UP", (width - 195, height - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (width - 115, height - 44), (width - 15, height - 12), (35, 45, 58), -1)
        cv2.putText(canvas, "DOWN", (width - 85, height - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    def _wrap_text(self, text: str, max_chars: int = 60) -> List[str]:
        words = text.replace("\r", "").split()
        lines = []
        cur: List[str] = []
        cur_len = 0
        for w in words:
            if "\n" in w:
                parts = w.split("\n")
                for p in parts[:-1]:
                    cur.append(p)
                    lines.append(" ".join(cur))
                    cur = []
                    cur_len = 0
                w = parts[-1]

            if cur_len + len(w) + 1 > max_chars:
                lines.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += len(w) + 1

        if cur:
            lines.append(" ".join(cur))
        return lines


kiosk_manager = KioskManager()
