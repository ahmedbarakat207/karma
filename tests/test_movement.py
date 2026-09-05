import random
import time

import pytest

from src.cognition.movement_intents import check_move_intent
from src.hardware.drive import DriveBase
from src.memory.working import WorkingMemory
from src.navigation.explorer import (
    Explorer,
    choose_explore_action,
    is_blocked,
)
from src.state import InternalState


@pytest.fixture
def drive():
    d = DriveBase()
    d.clear_estop()
    d.stop()
    yield d
    d.stop()
    d.clear_estop()


@pytest.fixture
def explorer():
    e = Explorer()
    yield e
    e.stop()


def _mem():
    return WorkingMemory()


def _state():
    s = InternalState()
    s.energy = 0.8
    s.curiosity = 0.7
    return s


# -- drive ---------------------------------------------------------------

def test_drive_clamps_to_max_duty(drive):
    assert drive.set_velocity(5.0, 5.0, duration=0.2)
    import src.config as config
    assert abs(drive.left.last_duty) <= abs(config.DRIVE_MAX_DUTY) + 1e-6
    assert abs(drive.right.last_duty) <= abs(config.DRIVE_MAX_DUTY) + 1e-6


def test_drive_watchdog_auto_stops(drive):
    assert drive.forward(duration=0.3)
    assert drive.is_moving
    time.sleep(0.7)
    assert not drive.is_moving
    assert drive.left.last_duty == 0.0
    assert drive.right.last_duty == 0.0


def test_drive_estop_refuses_and_clears(drive):
    drive.estop("test")
    assert drive.is_estopped
    assert drive.forward(duration=0.2) is False
    drive.clear_estop()
    assert drive.forward(duration=0.2) is True


def test_drive_manual_moves_log_label(drive):
    assert drive.turn_left(duration=0.2)
    label, _ts = drive.last_command()
    assert "turn_left" in label


# -- obstacle ------------------------------------------------------------

def test_is_blocked_large_centered_person():
    bboxes = [("person", 0.9, (170, 60, 470, 420))]  # 300x360 of 640x480
    assert is_blocked(bboxes) == "person"


def test_is_blocked_ignores_small_or_side_boxes():
    small_center = [("person", 0.9, (300, 200, 340, 260))]
    assert is_blocked(small_center) is None
    big_side = [("chair", 0.9, (0, 0, 200, 400))]
    assert is_blocked(big_side) is None
    big_decor = [("book", 0.9, (170, 60, 470, 420))]
    assert is_blocked(big_decor) is None
    assert is_blocked([]) is None


# -- wander policy --------------------------------------------------------

def test_explore_turns_when_blocked():
    v, w, reason = choose_explore_action({}, True, 0.7, 0.8,
                                         rng=random.Random(0))
    assert v == 0.0 and w != 0.0 and reason.startswith("blocked_turn")


def test_explore_rests_when_tired():
    v, w, reason = choose_explore_action({}, False, 0.2, 0.1,
                                         rng=random.Random(0))
    assert (v, w) == (0.0, 0.0) and reason == "rest_low_energy"


def test_explore_approaches_person_when_curious():
    spatial = {"left": ["Sara"], "center": [], "right": []}
    v, w, reason = choose_explore_action(spatial, False, 0.8, 0.8,
                                         rng=random.Random(0))
    assert reason == "approach_left_person" and w > 0


# -- intents (EN + AR) ----------------------------------------------------

def test_move_intents_english():
    assert check_move_intent("follow me")[0] == "follow"
    assert check_move_intent("stop moving right now")[0] == "stop"
    assert check_move_intent("go explore the room")[0] == "explore"
    assert check_move_intent("go to the kitchen") == ("goto", "kitchen")
    assert check_move_intent("move forward a bit")[0] == "forward"
    assert check_move_intent("turn left")[0] == "left"
    assert check_move_intent("stop following me")[0] == "unfollow"


def test_move_intents_arabic():
    assert check_move_intent("اتبعني")[0] == "follow"
    assert check_move_intent("اقف مكانك")[0] == "stop"
    assert check_move_intent("استكشف الأوضة")[0] == "explore"
    assert check_move_intent("روح المطبخ") == ("goto", "kitchen")
    assert check_move_intent("لف شمال")[0] == "left"
    assert check_move_intent("امشي قدام شوية")[0] == "forward"


def test_move_intent_no_false_positives():
    assert check_move_intent("I left my keys on the table") is None
    assert check_move_intent("what's up karma") is None
    assert check_move_intent("") is None


# -- explorer tick ---------------------------------------------------------

def test_explorer_idle_ticks_silent(explorer, drive):
    explorer._last_tick = 0.0
    assert explorer.tick(_mem(), drive, _state()) is None
    assert not drive.is_moving


def test_explorer_explore_moves_and_logs(explorer, drive):
    explorer.start_explore()
    explorer._last_tick = 0.0
    line = explorer.tick(_mem(), drive, _state())
    assert line and "Exploring" in line
    assert drive.is_moving


def test_explorer_goto_completes_and_sets_location(explorer, drive):
    mem = _mem()
    assert explorer.goto("kitchen") == "kitchen"
    explorer._last_tick = 0.0
    line = explorer.tick(mem, drive, _state())
    assert line == "Going to kitchen"
    assert mem.consciousness.self_model["location"] == "kitchen"
    assert explorer.mode == Explorer.IDLE


def test_explorer_goto_rejects_unknown_beacon(explorer):
    assert explorer.goto("moon") == ""
    assert explorer.mode == Explorer.IDLE
