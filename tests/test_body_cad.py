"""
tests/test_body_cad.py
======================
Unit tests for the Karma robot 3D mechanical CAD generation and assembly integrity.
"""

import os
import struct
import pytest
from typing import Tuple

BODY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "body")

EXPECTED_STLS = [
    "base_front.stl",
    "base_back.stl",
    "column_front.stl",
    "column_back.stl",
    "neck_front.stl",
    "neck_back.stl",
    "steering_arm.stl",
    "head_window_half.stl",
    "head_cover_half.stl",
]

def read_stl_metadata(filepath: str) -> Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]:
    """Reads triangle count and bounding box (min_xyz, max_xyz) from binary STL."""
    assert os.path.exists(filepath), f"STL file not found: {filepath}"
    with open(filepath, "rb") as f:
        header = f.read(80)
        assert len(header) == 80, "Corrupted STL header"
        count_bytes = f.read(4)
        assert len(count_bytes) == 4, "Missing triangle count in STL"
        num_triangles = struct.unpack("<I", count_bytes)[0]
        assert num_triangles > 0, f"STL has 0 triangles: {filepath}"

        min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
        max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")

        for _ in range(num_triangles):
            record = f.read(50)
            assert len(record) == 50, "Incomplete triangle record in STL"
            floats = struct.unpack("<12fH", record)
            for i in range(3):
                vx, vy, vz = floats[3 * i + 3 : 3 * i + 6]
                min_x, max_x = min(min_x, vx), max(max_x, vx)
                min_y, max_y = min(min_y, vy), max(max_y, vy)
                min_z, max_z = min(min_z, vz), max(max_z, vz)

        return num_triangles, (min_x, min_y, min_z), (max_x, max_y, max_z)


@pytest.mark.parametrize("stl_filename", EXPECTED_STLS)
def test_stl_files_exist_and_valid(stl_filename):
    """Verifies that all 9 production STLs exist, are valid binary STLs, and have positive volume."""
    path = os.path.join(BODY_DIR, stl_filename)
    num_tris, min_xyz, max_xyz = read_stl_metadata(path)
    assert num_tris >= 100, f"{stl_filename} has suspiciously low triangle count: {num_tris}"

    dx = max_xyz[0] - min_xyz[0]
    dy = max_xyz[1] - min_xyz[1]
    dz = max_xyz[2] - min_xyz[2]

    assert dx > 0, f"{stl_filename} has zero X dimension"
    assert dy > 0, f"{stl_filename} has zero Y dimension"
    assert dz > 0, f"{stl_filename} has zero Z dimension"


def test_base_chassis_dimensions():
    """Verifies base footprint fits 320x400x220 mm specifications."""
    _, min_f, max_f = read_stl_metadata(os.path.join(BODY_DIR, "base_front.stl"))
    _, min_b, max_b = read_stl_metadata(os.path.join(BODY_DIR, "base_back.stl"))

    # Width: X should span -160 to +160 (320 mm)
    assert min_f[0] == pytest.approx(-160.0, abs=1.0)
    assert max_f[0] == pytest.approx(160.0, abs=1.0)
    assert min_b[0] == pytest.approx(-160.0, abs=1.0)
    assert max_b[0] == pytest.approx(160.0, abs=1.0)

    # Height: Z should be 220 mm
    assert max_f[2] - min_f[2] == pytest.approx(220.0, abs=1.0)
    assert max_b[2] - min_b[2] == pytest.approx(220.0, abs=1.0)


def test_column_spine_dimensions():
    """Verifies column torso height is 480 mm (spans Z=220 to Z=700 in assembly)."""
    _, min_f, max_f = read_stl_metadata(os.path.join(BODY_DIR, "column_front.stl"))
    _, min_b, max_b = read_stl_metadata(os.path.join(BODY_DIR, "column_back.stl"))

    assert max_f[2] - min_f[2] == pytest.approx(480.0, abs=1.0)
    assert max_b[2] - min_b[2] == pytest.approx(480.0, abs=1.0)


def test_neck_actuation_dimensions():
    """Verifies neck module height is 60 mm (spans Z=700 to Z=760 in assembly)."""
    _, min_f, max_f = read_stl_metadata(os.path.join(BODY_DIR, "neck_front.stl"))
    _, min_b, max_b = read_stl_metadata(os.path.join(BODY_DIR, "neck_back.stl"))

    assert min_f[0] == pytest.approx(-80.0, abs=1.0)
    assert max_f[0] == pytest.approx(80.0, abs=1.0)
    assert max_f[2] >= 60.0
    assert max_b[2] >= 60.0


def test_head_bezel_dimensions():
    """Verifies head front bezel is 300 mm wide to fit 7\" to 10.1\" displays."""
    _, min_f, max_f = read_stl_metadata(os.path.join(BODY_DIR, "head_window_half.stl"))
    assert max_f[0] - min_f[0] == pytest.approx(300.0, abs=1.0)


def test_cad_generation_script_execution():
    """Verifies that generate_robot_cad.py runs without error and outputs all STLs."""
    from body.generate_robot_cad import generate_all_production_cad
    generate_all_production_cad(BODY_DIR)
    for stl in EXPECTED_STLS:
        assert os.path.exists(os.path.join(BODY_DIR, stl))
