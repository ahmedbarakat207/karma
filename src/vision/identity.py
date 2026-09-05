"""Fuse YOLO person boxes with face-registry identities.

YOLO only outputs `person`. Face tracker outputs names + face boxes.
This module associates them spatially so downstream code (HUD, memory,
prompt context) sees `Sara` instead of `person`.
"""

from typing import List, Tuple

# (label, conf, (x1, y1, x2, y2))
BBox = Tuple[str, float, Tuple[int, int, int, int]]
# (x, y, w, h, name, emotion)
FaceBox = Tuple[int, int, int, int, str, str]

_UNKNOWN_FACE_LABELS = {"face", "unknown", "", "looking away"}


def _is_named(name: str) -> bool:
    return bool(name and name.strip() and name.strip().lower() not in _UNKNOWN_FACE_LABELS)


def point_in_box(px: float, py: float, box: Tuple[int, int, int, int], pad: int = 10) -> bool:
    x1, y1, x2, y2 = box
    return (x1 - pad) <= px <= (x2 + pad) and (y1 - pad) <= py <= (y2 + pad)


def _box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def fuse_person_identities(
    bboxes: List[BBox],
    faces: List[FaceBox],
) -> List[BBox]:
    """Return a copy of bboxes where `person` boxes containing a named face are relabeled.

    - Containment: face center inside person box (+pad).
    - One face claims at most one person box (smallest containing area wins).
    - One person box gets at most one name (first claimant wins).
    - Fallback: single unassigned person + single named face without
      containment (detector offset / partial overlap) still fuses. This
      covers the common 1-person-in-room case.
    - Faces named "Face"/unknown never rename anything.
    """
    if not bboxes:
        return []

    named_faces = [f for f in (faces or []) if _is_named(f[4])]
    if not named_faces:
        return list(bboxes)

    person_idx = [i for i, (lbl, _, _) in enumerate(bboxes) if lbl == "person"]
    if not person_idx:
        return list(bboxes)

    fused = list(bboxes)
    claimed_persons = set()
    claimed_faces = set()

    # Pass 1: spatial containment, smallest area wins.
    for fi, (fx, fy, fw, fh, fname, _emotion) in enumerate(named_faces):
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        best_pi = None
        best_area = None
        for pi in person_idx:
            if pi in claimed_persons:
                continue
            _, conf, pbox = fused[pi]
            if point_in_box(cx, cy, pbox):
                area = _box_area(pbox)
                if best_pi is None or area < best_area:
                    best_pi = pi
                    best_area = area
        if best_pi is not None:
            _, conf, pbox = fused[best_pi]
            fused[best_pi] = (fname, conf, pbox)
            claimed_persons.add(best_pi)
            claimed_faces.add(fi)

    # Pass 2: 1-to-1 fallback when geometry missed (e.g. cropped face box).
    unclaimed_persons = [pi for pi in person_idx if pi not in claimed_persons]
    unclaimed_faces = [fi for fi in range(len(named_faces)) if fi not in claimed_faces]
    if len(unclaimed_persons) == 1 and len(unclaimed_faces) == 1:
        pi = unclaimed_persons[0]
        fname = named_faces[unclaimed_faces[0]][4]
        _, conf, pbox = fused[pi]
        fused[pi] = (fname, conf, pbox)

    return fused


def positions_from_bboxes(bboxes: List[BBox]):
    """Build label -> list of centers from fused bboxes (multi-instance safe)."""
    positions = {}
    for lbl, _conf, (x1, y1, x2, y2) in bboxes:
        positions.setdefault(lbl, []).append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return positions
