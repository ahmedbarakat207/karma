import threading
import time

import numpy as np

from src.memory.working import WorkingMemory
from src.vision.vlm import (
    CorrectionCache,
    VLMVerifier,
    apply_corrections,
    build_prompt,
    parse_result,
    should_verify,
    yolo_origin_labels,
)


def test_yolo_origin_labels_filters_tracker_labels():
    labels = {"chair", "Sara looking at you (smiling)", "Face (looking away)",
              "person waving hand", "person", "laptop"}
    assert yolo_origin_labels(labels) == ["chair", "laptop", "person"]


def test_should_verify_triggers_on_novelty():
    got = should_verify(["chair"], set(), busy=False, now=100.0,
                        last_verify_time=0.0, cooldown=20.0)
    assert got == ["chair"]


def test_should_verify_drops_when_busy_or_cooldown_or_empty():
    assert should_verify(["chair"], set(), busy=True, now=100.0,
                         last_verify_time=0.0, cooldown=20.0) == []
    assert should_verify(["chair"], set(), busy=False, now=100.0,
                         last_verify_time=90.0, cooldown=20.0) == []
    assert should_verify([], set(), busy=False, now=100.0,
                         last_verify_time=0.0, cooldown=20.0) == []


def test_build_prompt_names_labels():
    p = build_prompt(["tv", "person"], ["Sara"])
    assert "tv" in p and "Sara" in p and "corrections" in p


def test_parse_result_clean_and_fenced():
    raw = '{"scene": "A cozy desk.", "people": [{"appearance": "red shirt", "name": "Sara"}], "objects": ["laptop"], "corrections": {"tv": "monitor"}}'
    r = parse_result(raw)
    assert r["scene"] == "A cozy desk."
    assert r["people"][0]["name"] == "Sara"
    assert r["corrections"] == {"tv": "monitor"}
    assert parse_result("```json\n" + raw + "\n```") == r


def test_parse_result_garbage_and_same_label_correction():
    assert parse_result("") == {}
    assert parse_result("no json here") == {}
    r = parse_result('{"scene": "x", "corrections": {"tv": "tv", "TV": "tv "}}')
    assert r["corrections"] == {}


def test_apply_corrections():
    assert apply_corrections(["tv", "chair"], {"tv": "microwave"}) == ["microwave", "chair"]
    assert apply_corrections(["tv"], {}) == ["tv"]


def test_correction_cache_ttl():
    cache = CorrectionCache(ttl=0.05)
    cache.update({"tv": "microwave"})
    assert cache.lookup("tv") == "microwave"
    assert cache.apply(["tv", "chair"]) == ["microwave", "chair"]
    time.sleep(0.08)
    assert cache.lookup("tv") == "tv"
    assert cache.active() == {}


def test_working_memory_recent_by_kind():
    mem = WorkingMemory()
    mem.add(kind="vlm_scene", text="A cat on the couch.", counts_as_activity=False)
    mem.add(kind="object", text="chair", counts_as_activity=False)
    assert mem.recent_by_kind("vlm_scene", 60) == ["A cat on the couch."]
    assert mem.recent_by_kind("object", 60) == ["chair"]
    assert mem.recent_by_kind("vlm_scene", 0.0) == []


def _fake_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_verifier_maybe_verify_disabled():
    import src.config as config
    old = config.VLM_ENABLED
    config.VLM_ENABLED = False
    try:
        v = VLMVerifier()
        assert v.maybe_verify(_fake_frame(), {"chair"}, set()) is False
    finally:
        config.VLM_ENABLED = old


def test_verifier_submit_and_callback_offline():
    v = VLMVerifier()
    v._ensure_model = lambda: True  # skip download
    seen = {}

    def fake_describe(jpeg_bytes, yolo_labels, known_names):
        assert jpeg_bytes[:2] == b"\xff\xd8"  # real JPEG snapshot
        assert yolo_labels == ["chair"]
        assert known_names == ["Sara"]
        return {"scene": "A chair by the desk.", "people": [],
                "objects": ["chair"], "corrections": {"chair": "armchair"}}

    v.describe = fake_describe
    done = threading.Event()

    def on_result(result, job):
        seen.update(result)
        done.set()

    assert v.maybe_verify(_fake_frame(), {"chair"}, {"Sara"},
                          now=time.time(), on_result=on_result) is True
    assert done.wait(timeout=10)
    assert seen["scene"] == "A chair by the desk."
    assert v.corrections.lookup("chair") == "armchair"
    assert v.busy is False


def test_store_vlm_result_writes_memory():
    from src.vision.pipeline import _store_vlm_result
    mem = WorkingMemory()

    class Job:
        yolo_labels = ["tv"]
        known_names = []

    _store_vlm_result(mem, {"scene": "A kitchen.",
                            "people": [{"appearance": "blue hoodie", "name": None}],
                            "objects": ["microwave"],
                            "corrections": {"tv": "microwave"}}, Job())
    scenes = mem.recent_by_kind("vlm_scene", 60)
    assert len(scenes) == 1
    assert "microwave" in scenes[0]

    before = len(mem.all_events())
    _store_vlm_result(mem, {}, Job())
    assert len(mem.all_events()) == before
