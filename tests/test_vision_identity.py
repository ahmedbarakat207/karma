import numpy as np

from src.vision.identity import (
    fuse_person_identities,
    point_in_box,
    positions_from_bboxes,
)


def test_point_in_box_with_pad():
    assert point_in_box(50, 50, (0, 0, 100, 100))
    assert not point_in_box(500, 500, (0, 0, 100, 100))
    # pad extends bounds
    assert point_in_box(105, 50, (0, 0, 100, 100), pad=10)
    assert not point_in_box(115, 50, (0, 0, 100, 100), pad=10)


def test_fuse_single_person_single_face():
    bboxes = [("person", 0.9, (0, 0, 200, 400))]
    faces = [(50, 60, 80, 80, "Sara", "smiling")]
    fused = fuse_person_identities(bboxes, faces)
    assert fused[0][0] == "Sara"
    assert fused[0][1] == 0.9
    # input not mutated
    assert bboxes[0][0] == "person"


def test_fuse_ignores_unknown_face():
    bboxes = [("person", 0.9, (0, 0, 200, 400))]
    faces = [(50, 60, 80, 80, "Face", "neutral")]
    fused = fuse_person_identities(bboxes, faces)
    assert fused[0][0] == "person"


def test_fuse_two_persons_one_face_only_one_renamed():
    bboxes = [
        ("person", 0.9, (0, 0, 200, 400)),
        ("person", 0.85, (300, 0, 500, 400)),
    ]
    faces = [(50, 60, 80, 80, "Sara", "neutral")]
    fused = fuse_person_identities(bboxes, faces)
    names = [lbl for lbl, _, _ in fused]
    assert "Sara" in names
    assert "person" in names
    assert names.count("Sara") == 1


def test_fuse_fallback_single_single_without_containment():
    # Face box offset (calibration drift) but only one candidate each.
    bboxes = [("person", 0.7, (0, 0, 100, 200))]
    faces = [(400, 300, 60, 60, "Ahmed", "neutral")]
    fused = fuse_person_identities(bboxes, faces)
    assert fused[0][0] == "Ahmed"


def test_fuse_no_person_boxes_untouched():
    bboxes = [("chair", 0.8, (0, 0, 50, 50))]
    faces = [(5, 5, 20, 20, "Sara", "neutral")]
    assert fuse_person_identities(bboxes, faces) == bboxes


def test_positions_from_bboxes_multi_instance():
    bboxes = [
        ("person", 0.9, (0, 0, 100, 100)),
        ("person", 0.8, (200, 0, 300, 100)),
        ("chair", 0.7, (0, 0, 10, 10)),
    ]
    pos = positions_from_bboxes(bboxes)
    assert len(pos["person"]) == 2
    assert pos["person"][0] == (50.0, 50.0)
    assert pos["chair"][0] == (5.0, 5.0)


def test_draw_objects_renders_named_person():
    cv2 = __import__("cv2")
    from src.vision.render import VisionRenderer

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Should not raise, and should draw both unknown + named persons now.
    VisionRenderer.draw_objects(frame, [
        ("person", 0.9, (10, 10, 100, 200)),
        ("Sara", 0.9, (200, 10, 300, 200)),
    ])
    assert frame.sum() > 0
