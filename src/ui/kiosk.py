"""
Karma Touchscreen Kiosk Overlay & Interactive Facility Menu.
Provides a 7" LCD touch-friendly interface (800x480 native resolution)
with PDF document reader, multi-floor facility maps, student apps showcase,
and achievements display. Supports both direct capacitive touch and voice navigation.
"""
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


class KioskManager:
    """
    State machine and renderer for the 7" LCD Touchscreen Kiosk System.
    """

    def __init__(self, store: Optional[MemoryStore] = None):
        self.active_view: str = "face"  # "face", "docs", "map", "apps", "achievements"
        self.store = store
        self.current_floor_idx: int = 0
        self.selected_doc: Optional[str] = None
        self.doc_page_idx: int = 0
        self.apps_scroll_idx: int = 0
        self.achievements_scroll_idx: int = 0

        # Cached data
        self.student_apps: List[Dict[str, Any]] = []
        self.achievements: List[Dict[str, Any]] = []
        self.floor_maps: List[Tuple[str, str]] = []  # (floor_label, file_path)
        self.indexed_docs: List[Dict[str, Any]] = []
        self.doc_chunks: List[str] = []

        self.reload_all_data()

    def reload_all_data(self) -> None:
        """Reloads JSON and image assets from data directory."""
        # 1. Student Apps
        if os.path.exists(STUDENT_APPS_FILE):
            try:
                with open(STUDENT_APPS_FILE, "r", encoding="utf-8") as f:
                    self.student_apps = json.load(f)
            except Exception as e:
                config.log_debug(f"[kiosk] error reading {STUDENT_APPS_FILE}: {e}")
                self.student_apps = []

        # 2. Achievements
        if os.path.exists(ACHIEVEMENTS_FILE):
            try:
                with open(ACHIEVEMENTS_FILE, "r", encoding="utf-8") as f:
                    self.achievements = json.load(f)
            except Exception as e:
                config.log_debug(f"[kiosk] error reading {ACHIEVEMENTS_FILE}: {e}")
                self.achievements = []

        # 3. Floor Maps
        self.floor_maps = []
        if os.path.isdir(MAPS_DIR):
            map_files = sorted(glob.glob(os.path.join(MAPS_DIR, "*.jpg")) + glob.glob(os.path.join(MAPS_DIR, "*.png")))
            for p in map_files:
                basename = os.path.splitext(os.path.basename(p))[0]
                label = basename.replace("_", " ").title()
                self.floor_maps.append((label, p))

        # 4. Indexed Documents from RAG / MemoryStore
        self.reload_documents()

    def reload_documents(self) -> None:
        """Queries the vector store for all indexed document sources."""
        if self.store is None:
            try:
                self.store = MemoryStore()
            except Exception:
                self.store = None

        if self.store:
            try:
                self.indexed_docs = self.store.list_sources(kind="document")
                if self.indexed_docs and self.selected_doc is None:
                    self.select_document(self.indexed_docs[0]["source"])
            except Exception as e:
                config.log_debug(f"[kiosk] error loading document sources: {e}")
                self.indexed_docs = []

    def select_document(self, doc_name: str) -> None:
        """Loads chunks for a specific document to display in the reader."""
        self.selected_doc = doc_name
        self.doc_page_idx = 0
        self.doc_chunks = []
        if self.store:
            try:
                rows = self.store.db.execute(
                    "SELECT text FROM memories WHERE kind = 'document' AND source = ? ORDER BY id ASC",
                    (doc_name,)
                ).fetchall()
                self.doc_chunks = [r[0] for r in rows]
            except Exception as e:
                config.log_debug(f"[kiosk] error loading chunks for {doc_name}: {e}")

    # --------------------------------------------------------------------------
    # Navigation Actions & Servo Integration
    # --------------------------------------------------------------------------
    def open_view(self, view_name: str, floor_idx: Optional[int] = None) -> None:
        """Switches active view and tilts head to 135° for touch interaction."""
        valid_views = ("docs", "map", "apps", "achievements", "face")
        if view_name not in valid_views:
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
        """Closes the kiosk menu, returns to animated face, and pitches head forward."""
        self.open_view("face")

    def is_active(self) -> bool:
        """Returns True if user is currently interacting with the kiosk menu."""
        return self.active_view != "face"

    # --------------------------------------------------------------------------
    # Touch Event Handler (Mouse Click)
    # --------------------------------------------------------------------------
    def handle_touch(self, x: int, y: int, screen_w: int, screen_h: int) -> bool:
        """
        Processes touch coordinates (x, y). Returns True if touch was handled.
        """
        # 1. In Face mode: Check top-right [ ☰ MENU ] trigger button
        if self.active_view == "face":
            # Button area: top right corner (x: w - 130 .. w - 15, y: 12 .. 48)
            if x >= (screen_w - 140) and y <= 55:
                self.open_view("map")
                return True
            return False

        # 2. In Kiosk Mode: Check Top Navigation Bar
        # Nav buttons:
        # [ ✕ Back ]    : screen_w - 90 .. screen_w - 15
        # [ 🏆 Achiev ] : screen_w - 240 .. screen_w - 105
        # [ 🚀 Apps ]   : screen_w - 385 .. screen_w - 250
        # [ 🗺️ Maps ]   : screen_w - 515 .. screen_w - 395
        # [ 📄 Docs ]   : screen_w - 635 .. screen_w - 525
        if y <= 50:
            if x >= (screen_w - 95):
                self.close()
                return True
            elif (screen_w - 240) <= x < (screen_w - 100):
                self.active_view = "achievements"
                return True
            elif (screen_w - 385) <= x < (screen_w - 245):
                self.active_view = "apps"
                return True
            elif (screen_w - 515) <= x < (screen_w - 390):
                self.active_view = "map"
                return True
            elif (screen_w - 640) <= x < (screen_w - 520):
                self.active_view = "docs"
                return True
            return True

        # 3. View-Specific Touch Interaction
        if self.active_view == "map":
            # Check floor selector tabs (bottom left bar: y: screen_h - 55 .. screen_h - 15)
            if y >= (screen_h - 60):
                tab_w = 120
                for idx in range(len(self.floor_maps)):
                    bx1 = 30 + idx * (tab_w + 12)
                    bx2 = bx1 + tab_w
                    if bx1 <= x <= bx2:
                        self.current_floor_idx = idx
                        return True

        elif self.active_view == "docs":
            # Left panel document selection (x: 20 .. 240, y: 70 .. screen_h - 20)
            if x <= 240 and y >= 70:
                item_h = 44
                idx = (y - 70) // item_h
                if 0 <= idx < len(self.indexed_docs):
                    self.select_document(self.indexed_docs[idx]["source"])
                    return True
            # Right panel scroll buttons (bottom right: y: screen_h - 50 .. screen_h - 10)
            elif y >= (screen_h - 55):
                if 420 <= x <= 530:  # [▲ Prev]
                    if self.doc_page_idx > 0:
                        self.doc_page_idx -= 1
                    return True
                elif 550 <= x <= 660:  # [▼ Next]
                    if self.doc_page_idx < len(self.doc_chunks) - 1:
                        self.doc_page_idx += 1
                    return True

        elif self.active_view == "apps":
            # Scroll buttons on right side
            if y >= (screen_h - 55):
                if (screen_w - 260) <= x <= (screen_w - 150):
                    if self.apps_scroll_idx > 0:
                        self.apps_scroll_idx -= 1
                    return True
                elif (screen_w - 140) <= x <= (screen_w - 30):
                    if self.apps_scroll_idx < max(0, len(self.student_apps) - 3):
                        self.apps_scroll_idx += 1
                    return True

        elif self.active_view == "achievements":
            # Scroll buttons on right side
            if y >= (screen_h - 55):
                if (screen_w - 260) <= x <= (screen_w - 150):
                    if self.achievements_scroll_idx > 0:
                        self.achievements_scroll_idx -= 1
                    return True
                elif (screen_w - 140) <= x <= (screen_w - 30):
                    if self.achievements_scroll_idx < max(0, len(self.achievements) - 3):
                        self.achievements_scroll_idx += 1
                    return True

        return True

    # --------------------------------------------------------------------------
    # Rendering Subsystems
    # --------------------------------------------------------------------------
    def render_overlay_button(self, frame: np.ndarray) -> None:
        """Renders the top-right [ ☰ MENU ] pill button on top of the animated face."""
        h, w = frame.shape[:2]
        bx1, by1 = w - 135, 14
        bx2, by2 = w - 15, 48

        # Glow / Pill background
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (28, 34, 42), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 220, 255), 2)

        # Icon and label
        cv2.putText(frame, ":: MENU", (bx1 + 18, by1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

    def render_kiosk(self, width: int = 800, height: int = 480) -> np.ndarray:
        """Renders the active full-screen kiosk menu frame."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (18, 22, 28)  # Deep cyber dark slate

        # Render Top Navigation Header Bar
        self._render_top_navbar(canvas, width)

        # Render Sub-View Content
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
        """Draws the top interactive navigation bar."""
        # Top banner background
        cv2.rectangle(canvas, (0, 0), (width, 52), (28, 34, 44), -1)
        cv2.line(canvas, (0, 52), (width, 52), (50, 65, 85), 1)

        # Robot Brand Logo
        cv2.putText(canvas, "KARMA // KIOSK", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

        # Tabs definitions: (label, view_key, x1, x2)
        tabs = [
            ("DOCS",   "docs",         width - 640, width - 525),
            ("MAP",    "map",          width - 515, width - 395),
            ("APPS",   "apps",         width - 385, width - 250),
            ("AWARDS", "achievements", width - 240, width - 105),
        ]

        for label, view_key, x1, x2 in tabs:
            is_selected = (self.active_view == view_key)
            bg_color = (0, 180, 230) if is_selected else (38, 46, 58)
            text_color = (15, 20, 25) if is_selected else (200, 215, 230)
            border_color = (0, 220, 255) if is_selected else (60, 75, 95)

            cv2.rectangle(canvas, (x1, 10), (x2, 44), bg_color, -1)
            cv2.rectangle(canvas, (x1, 10), (x2, 44), border_color, 1)

            # Center text in tab
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)[0]
            tx = x1 + (x2 - x1 - text_size[0]) // 2
            cv2.putText(canvas, label, (tx, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 2, cv2.LINE_AA)

        # [ ✕ Back / Exit ] button
        bx1, bx2 = width - 95, width - 15
        cv2.rectangle(canvas, (bx1, 10), (bx2, 44), (45, 30, 35), -1)
        cv2.rectangle(canvas, (bx1, 10), (bx2, 44), (80, 100, 255), 1)
        cv2.putText(canvas, "EXIT", (bx1 + 22, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 150, 255), 2, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # Sub-View: Multi-Floor Facility Map
    # --------------------------------------------------------------------------
    def _render_map_view(self, canvas: np.ndarray, width: int, height: int) -> None:
        if not self.floor_maps:
            cv2.putText(canvas, "No facility map found in data/maps/", (60, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 160, 180), 2, cv2.LINE_AA)
            return

        label, file_path = self.floor_maps[self.current_floor_idx]
        map_img = cv2.imread(file_path)

        if map_img is not None:
            # Scale map image to fit canvas area (leaving room for top nav and bottom floor tabs)
            avail_w = width - 40
            avail_h = height - 125
            mh, mw = map_img.shape[:2]
            scale = min(avail_w / mw, avail_h / mh)
            new_w, new_h = int(mw * scale), int(mh * scale)

            resized = cv2.resize(map_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ox = (width - new_w) // 2
            oy = 60 + (avail_h - new_h) // 2

            canvas[oy:oy + new_h, ox:ox + new_w] = resized
            cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + new_w + 1, oy + new_h + 1), (0, 200, 255), 1)

        # Render Floor Selector Tabs at bottom
        tab_w = 125
        for idx, (flabel, _) in enumerate(self.floor_maps):
            bx1 = 30 + idx * (tab_w + 12)
            bx2 = bx1 + tab_w
            by1 = height - 52
            by2 = height - 14

            is_sel = (idx == self.current_floor_idx)
            bg = (0, 180, 230) if is_sel else (32, 40, 52)
            fg = (15, 20, 25) if is_sel else (210, 225, 240)
            bc = (0, 220, 255) if is_sel else (60, 75, 95)

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bc, 1)

            tsize = cv2.getTextSize(flabel[:14], cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            tx = bx1 + (tab_w - tsize[0]) // 2
            cv2.putText(canvas, flabel[:14], (tx, by1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg, 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # Sub-View: PDF & RAG Knowledge Reader
    # --------------------------------------------------------------------------
    def _render_docs_view(self, canvas: np.ndarray, width: int, height: int) -> None:
        # Left Panel: Document List
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
            item_h = 42
            for i, doc in enumerate(self.indexed_docs[:7]):
                dy = 102 + i * item_h
                is_sel = (doc["source"] == self.selected_doc)
                bg = (40, 52, 68) if is_sel else (28, 34, 44)
                border = (0, 220, 255) if is_sel else (45, 55, 70)

                cv2.rectangle(canvas, (20, dy), (panel_w - 6, dy + 36), bg, -1)
                cv2.rectangle(canvas, (20, dy), (panel_w - 6, dy + 36), border, 1)

                doc_label = doc["source"]
                if len(doc_label) > 18:
                    doc_label = doc_label[:16] + ".."
                cv2.putText(canvas, doc_label, (28, dy + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 240, 255), 1, cv2.LINE_AA)

        # Right Panel: Document Text Viewer
        rx1, rx2 = panel_w + 15, width - 15
        ry1, ry2 = 62, height - 15
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (24, 30, 38), -1)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (45, 55, 70), 1)

        header_title = self.selected_doc or "Document Reader"
        cv2.putText(canvas, f"EXCERPT VIEWER // {header_title.upper()}", (rx1 + 15, ry1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        if not self.doc_chunks:
            cv2.putText(canvas, "Select a document to read its semantic passages.", (rx1 + 20, ry1 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 180), 1, cv2.LINE_AA)
        else:
            total_pages = len(self.doc_chunks)
            curr_text = self.doc_chunks[self.doc_page_idx]

            # Wrap and render text lines
            lines = self._wrap_text(curr_text, max_chars=60)
            for j, line in enumerate(lines[:13]):
                cv2.putText(canvas, line, (rx1 + 16, ry1 + 60 + j * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 225, 240), 1, cv2.LINE_AA)

            # Page navigation indicator and buttons
            page_str = f"Chunk {self.doc_page_idx + 1} of {total_pages}"
            cv2.putText(canvas, page_str, (rx1 + 20, ry2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 145, 170), 1, cv2.LINE_AA)

            # [▲ Prev] and [▼ Next] touch buttons
            cv2.rectangle(canvas, (rx2 - 220, ry2 - 40), (rx2 - 120, ry2 - 10), (38, 48, 62), -1)
            cv2.rectangle(canvas, (rx2 - 220, ry2 - 40), (rx2 - 120, ry2 - 10), (0, 200, 255), 1)
            cv2.putText(canvas, "PREV", (rx2 - 195, ry2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

            cv2.rectangle(canvas, (rx2 - 105, ry2 - 40), (rx2 - 10, ry2 - 10), (38, 48, 62), -1)
            cv2.rectangle(canvas, (rx2 - 105, ry2 - 40), (rx2 - 10, ry2 - 10), (0, 200, 255), 1)
            cv2.putText(canvas, "NEXT", (rx2 - 80, ry2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # Sub-View: Student Apps Showcase
    # --------------------------------------------------------------------------
    def _render_apps_view(self, canvas: np.ndarray, width: int, height: int) -> None:
        cv2.putText(canvas, "STUDENT INNOVATION APPS & PROJECTS", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        if not self.student_apps:
            cv2.putText(canvas, "No apps loaded. Add entries to data/student_apps.json", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 150, 170), 1, cv2.LINE_AA)
            return

        visible_apps = self.student_apps[self.apps_scroll_idx:self.apps_scroll_idx + 3]
        card_h = 100
        card_w = width - 60

        for i, app in enumerate(visible_apps):
            cy1 = 105 + i * (card_h + 14)
            cy2 = cy1 + card_h

            cv2.rectangle(canvas, (30, cy1), (30 + card_w, cy2), (26, 32, 42), -1)
            cv2.rectangle(canvas, (30, cy1), (30 + card_w, cy2), (50, 65, 85), 1)

            # Left accent bar
            cv2.rectangle(canvas, (30, cy1), (36, cy2), (0, 220, 255), -1)

            # App Title & Category Badge
            name = app.get("name", "Unnamed Project")
            cv2.putText(canvas, name, (50, cy1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 250, 255), 2, cv2.LINE_AA)

            badge = app.get("category", "Project").upper()
            cv2.putText(canvas, f"[{badge}]", (50 + len(name) * 12 + 10, cy1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

            # Author / Team
            author = f"Creator: {app.get('author', 'Anonymous')}"
            cv2.putText(canvas, author, (50, cy1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 170, 190), 1, cv2.LINE_AA)

            # Description
            desc = app.get("description", "")
            if len(desc) > 85:
                desc = desc[:82] + "..."
            cv2.putText(canvas, desc, (50, cy1 + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 215, 230), 1, cv2.LINE_AA)

            # Status pill
            status = app.get("status", "Active")
            cv2.putText(canvas, f"• {status}", (30 + card_w - 95, cy1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 120), 1, cv2.LINE_AA)

        # Scroll controls at bottom right
        if len(self.student_apps) > 3:
            cv2.putText(canvas, f"Showing {self.apps_scroll_idx + 1}-{min(len(self.student_apps), self.apps_scroll_idx + 3)} of {len(self.student_apps)}",
                        (35, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 140, 160), 1, cv2.LINE_AA)
            # Buttons
            cv2.rectangle(canvas, (width - 230, height - 44), (width - 130, height - 12), (35, 45, 58), -1)
            cv2.putText(canvas, "UP", (width - 195, height - 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (width - 115, height - 44), (width - 15, height - 12), (35, 45, 58), -1)
            cv2.putText(canvas, "DOWN", (width - 85, height - 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # Sub-View: Achievements Showcase
    # --------------------------------------------------------------------------
    def _render_achievements_view(self, canvas: np.ndarray, width: int, height: int) -> None:
        cv2.putText(canvas, "ROBOT & LAB MILESTONES // ACHIEVEMENTS", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        if not self.achievements:
            cv2.putText(canvas, "No achievements in data/achievements.json", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 150, 170), 1, cv2.LINE_AA)
            return

        visible_achs = self.achievements[self.achievements_scroll_idx:self.achievements_scroll_idx + 3]
        card_h = 100
        card_w = width - 60

        for i, ach in enumerate(visible_achs):
            cy1 = 105 + i * (card_h + 14)
            cy2 = cy1 + card_h

            cv2.rectangle(canvas, (30, cy1), (30 + card_w, cy2), (26, 32, 42), -1)
            cv2.rectangle(canvas, (30, cy1), (30 + card_w, cy2), (60, 75, 95), 1)

            # Left Gold Accent
            cv2.rectangle(canvas, (30, cy1), (36, cy2), (50, 210, 255), -1)

            # Badge pill
            badge = ach.get("badge", "MILESTONE")
            cv2.putText(canvas, badge, (50, cy1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 210, 255), 1, cv2.LINE_AA)

            # Title
            title = ach.get("title", "Achievement")
            cv2.putText(canvas, title, (50, cy1 + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 250, 255), 2, cv2.LINE_AA)

            # Description
            desc = ach.get("description", "")
            cv2.putText(canvas, desc, (50, cy1 + 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)

            # Unlocked indicator
            date_str = ach.get("date", "Unlocked")
            cv2.putText(canvas, f"✓ {date_str}", (30 + card_w - 120, cy1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 120), 1, cv2.LINE_AA)

        # Scroll controls at bottom right
        if len(self.achievements) > 3:
            cv2.putText(canvas, f"Showing {self.achievements_scroll_idx + 1}-{min(len(self.achievements), self.achievements_scroll_idx + 3)} of {len(self.achievements)}",
                        (35, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 140, 160), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (width - 230, height - 44), (width - 130, height - 12), (35, 45, 58), -1)
            cv2.putText(canvas, "UP", (width - 195, height - 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (width - 115, height - 44), (width - 15, height - 12), (35, 45, 58), -1)
            cv2.putText(canvas, "DOWN", (width - 85, height - 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    def _wrap_text(self, text: str, max_chars: int = 60) -> List[str]:
        """Utility to word-wrap text into lines."""
        words = text.replace("\r", "").split()
        lines = []
        cur_line = []
        cur_len = 0
        for w in words:
            if "\n" in w:
                parts = w.split("\n")
                for p in parts[:-1]:
                    cur_line.append(p)
                    lines.append(" ".join(cur_line))
                    cur_line = []
                    cur_len = 0
                w = parts[-1]

            if cur_len + len(w) + 1 > max_chars:
                lines.append(" ".join(cur_line))
                cur_line = [w]
                cur_len = len(w)
            else:
                cur_line.append(w)
                cur_len += len(w) + 1

        if cur_line:
            lines.append(" ".join(cur_line))
        return lines


# Global singleton instance
kiosk_manager = KioskManager()
