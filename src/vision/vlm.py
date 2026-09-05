"""On-demand VLM verifier for the vision loop.

Division of labor (by design):
- YOLO runs every frame and owns position/movement tracking (bboxes are
  never touched by this module).
- The VLM fires at most once per novel YOLO sighting: it takes ONE still
  snapshot and returns (a) what the scene looks like, (b) what visible
  people look like (using stored names only, never inventing new ones),
  (c) what stuff is there, and (d) corrections where YOLO mislabeled.
- Corrections flow into memory labels / prompt context / HUD text only.
  Tracking keeps using raw YOLO boxes.

The verifier runs in a single daemon worker thread with a 1-slot queue:
if a verification is already running, new triggers are dropped so the
vision loop never stalls behind VLM inference (SmolVLM2-256M on Pi CPU
takes tens of seconds per snapshot).
"""

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src import config


def _is_face_tracker_label(label: str) -> bool:
    return "looking at you" in label or "looking away" in label


def _is_hand_tracker_label(label: str) -> bool:
    low = label.lower()
    return "hand" in low or "thumbs up" in low or "pointing finger" in low


def yolo_origin_labels(labels: Set[str]) -> List[str]:
    """Keep only labels that could have come from YOLO (not face/hand trackers)."""
    return sorted(
        lbl for lbl in labels
        if lbl and not _is_face_tracker_label(lbl) and not _is_hand_tracker_label(lbl)
    )


def should_verify(novel_yolo_labels: List[str], known_people: Set[str],
                  busy: bool, now: float, last_verify_time: float,
                  cooldown: Optional[float] = None) -> List[str]:
    """Decide whether a VLM snapshot is warranted. Pure function.

    - Drops everything while a verification is already running (YOLO keeps
      tracking; nothing is lost, the next novel sighting re-triggers).
    - Respects cooldown so a busy room doesn't queue endless snapshots.
    - A novel `person` with already-recognized faces still verifies: the
      VLM describes appearance/clothing, which face embeddings can't.
    """
    if busy:
        return []
    if not novel_yolo_labels:
        return []
    cd = cooldown if cooldown is not None else getattr(config, "VLM_COOLDOWN_SECONDS", 20.0)
    if now - last_verify_time < cd:
        return []
    return list(novel_yolo_labels)


def build_prompt(yolo_labels: List[str], known_names: List[str]) -> str:
    """Prompt the VLM for scene + people + objects + YOLO corrections."""
    names = ", ".join(known_names) if known_names else "none"
    labels = ", ".join(yolo_labels) if yolo_labels else "none"
    return (
        "You are the eyes of a small home robot. Look at this single photo.\n"
        f"An object detector claims these things are visible: {labels}.\n"
        f"People the robot already knows by face: {names}.\n"
        "Reply with JSON only, exactly this shape:\n"
        '{"scene": "one sentence describing the scene", '
        '"people": [{"appearance": "what this person looks like and wears", "name": "known name or null"}], '
        '"objects": ["short list of notable things actually visible"], '
        '"corrections": {"detector label": "what it really is"}}.\n'
        "Rules: use a known name ONLY if the person plausibly matches someone "
        "the robot knows; otherwise null. Only list a correction when the "
        "detector label is clearly wrong; omit correct labels."
    )


def parse_result(text: str) -> Dict[str, Any]:
    """Extract the VLM's JSON payload. Never raises; returns {} on garbage."""
    if not text:
        return {}
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE).strip()
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE).strip()
    start, end = s.find("{"), s.rfind("}") + 1
    if start == -1 or end <= start:
        return {}
    try:
        obj = json.loads(s[start:end])
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    scene = obj.get("scene")
    people = obj.get("people")
    objects = obj.get("objects")
    corrections = obj.get("corrections")
    return {
        "scene": scene if isinstance(scene, str) else "",
        "people": [
            {"appearance": str(p.get("appearance", "")),
             "name": p.get("name") if isinstance(p.get("name"), str) else None}
            for p in people if isinstance(p, dict)
        ] if isinstance(people, list) else [],
        "objects": [str(o) for o in objects] if isinstance(objects, list) else [],
        "corrections": {
            str(k): str(v) for k, v in corrections.items()
            if isinstance(v, str) and str(v).strip()
            and str(k).strip().lower() != str(v).strip().lower()
        } if isinstance(corrections, dict) else {},
    }


def apply_corrections(labels: List[str], corrections: Dict[str, str]) -> List[str]:
    """Remap YOLO labels through verified corrections (order-preserving)."""
    if not corrections:
        return list(labels)
    return [corrections.get(lbl, lbl) for lbl in labels]


class CorrectionCache:
    """YOLO label -> VLM correction with TTL. Thread-safe."""

    def __init__(self, ttl: Optional[float] = None):
        self._ttl = ttl if ttl is not None else getattr(
            config, "VLM_CORRECTION_TTL_SECONDS", 60.0)
        self._lock = threading.Lock()
        self._entries: Dict[str, Tuple[str, float]] = {}

    def update(self, corrections: Dict[str, str]) -> None:
        now = time.time()
        with self._lock:
            for k, v in corrections.items():
                self._entries[k] = (v, now)

    def lookup(self, label: str) -> str:
        with self._lock:
            hit = self._entries.get(label)
            if not hit:
                return label
            corrected, ts = hit
            if time.time() - ts > self._ttl:
                del self._entries[label]
                return label
            return corrected

    def apply(self, labels: List[str]) -> List[str]:
        return [self.lookup(lbl) for lbl in labels]

    def active(self) -> Dict[str, str]:
        now = time.time()
        with self._lock:
            expired = [k for k, (_, ts) in self._entries.items() if now - ts > self._ttl]
            for k in expired:
                del self._entries[k]
            return {k: v for k, (v, _) in self._entries.items()}


@dataclass
class VLMJob:
    jpeg_bytes: bytes
    yolo_labels: List[str]
    known_names: List[str]
    enqueued_at: float = field(default_factory=time.time)


class VLMVerifier:
    """Single-slot async VLM worker. The model loads lazily on first job."""

    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False
        self._last_verify_time = 0.0
        self._model = None
        self._processor = None
        self._load_error: Optional[str] = None
        self.corrections = CorrectionCache()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def last_verify_time(self) -> float:
        with self._lock:
            return self._last_verify_time

    @property
    def load_error(self) -> Optional[str]:
        with self._lock:
            return self._load_error

    def maybe_verify(self, frame_bgr, novel_labels: Set[str], known_people: Set[str],
                     now: Optional[float] = None,
                     on_result: Optional[Callable[[Dict[str, Any], VLMJob], None]] = None,
                     ) -> bool:
        """Snapshot + submit one verification. Returns True if submitted.

        Pure trigger logic + JPEG encode happen here (fast, on the vision
        thread); inference happens on the worker. Never blocks the loop.
        """
        if not getattr(config, "VLM_ENABLED", True):
            return False
        now = now if now is not None else time.time()
        candidates = yolo_origin_labels(set(novel_labels or []))
        with self._lock:
            busy = self._busy
            last = self._last_verify_time
        wanted = should_verify(candidates, set(known_people or []), busy, now, last)
        if not wanted:
            return False
        if frame_bgr is None:
            return False
        try:
            import cv2
            width = int(getattr(config, "VLM_SNAPSHOT_WIDTH", 384))
            h, w = frame_bgr.shape[:2]
            scale = width / max(1, w)
            small = cv2.resize(frame_bgr, (width, max(1, int(h * scale))))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                return False
            jpeg = bytes(buf)
        except Exception as e:
            config.log_debug(f"[vlm] snapshot encode note: {e}")
            return False

        job = VLMJob(jpeg_bytes=jpeg, yolo_labels=wanted,
                     known_names=sorted(known_people or []))
        with self._lock:
            if self._busy:  # raced another submit
                return False
            self._busy = True
            self._last_verify_time = now
        t = threading.Thread(target=self._run_job, args=(job, on_result),
                             daemon=True, name="vlm_verify")
        t.start()
        return True

    # -- worker --------------------------------------------------------
    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
            model_id = getattr(config, "VLM_MODEL_ID",
                               "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
            local = getattr(config, "VLM_MODEL_DIR", "")
            src = local if local and _has_snapshot(local) else model_id
            config.log_debug(f"[vlm] loading {src} on CPU...")
            self._processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                src, torch_dtype=torch.float32, low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self._model.eval()
            config.log_debug("[vlm] model ready")
            return True
        except Exception as e:
            with self._lock:
                self._load_error = str(e)
            config.log_debug(f"[vlm] model load note: {e}")
            return False

    def _run_job(self, job: VLMJob,
                 on_result: Optional[Callable[[Dict[str, Any], VLMJob], None]]) -> None:
        try:
            if not self._ensure_model():
                return
            result = self.describe(job.jpeg_bytes, job.yolo_labels, job.known_names)
            if result.get("corrections"):
                self.corrections.update(result["corrections"])
            if on_result:
                try:
                    on_result(result, job)
                except Exception as e:
                    config.log_debug(f"[vlm] result callback note: {e}")
        finally:
            with self._lock:
                self._busy = False

    def describe(self, jpeg_bytes: bytes, yolo_labels: List[str],
                 known_names: List[str]) -> Dict[str, Any]:
        """Synchronous single-photo description (worker thread only)."""
        import torch
        from PIL import Image
        import io as _io
        image = Image.open(_io.BytesIO(jpeg_bytes)).convert("RGB")
        prompt = build_prompt(yolo_labels, known_names)
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=int(getattr(config, "VLM_MAX_NEW_TOKENS", 128)),
                do_sample=False,
            )
        text = self._processor.decode(out[0], skip_special_tokens=True)
        # SmolVLM echoes the prompt; keep only what follows it.
        if prompt in text:
            text = text.split(prompt, 1)[1]
        return parse_result(text)


def _has_snapshot(path: str) -> bool:
    import os
    try:
        return os.path.isdir(path) and len(os.listdir(path)) > 0
    except Exception:
        return False


verifier = VLMVerifier()
