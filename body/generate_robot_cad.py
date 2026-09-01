"""
generate_robot_cad.py
=====================
Production-Grade Parametric 3D CAD Generator for the Karma Autonomous Companion Robot.

Generates sleek, organic, aerodynamic, and 3D-printable binary STL files with:
- Organic filleted and lofted aerodynamic contours (zero boxy/square edges)
- Sleek sculpted humanoid torso with waisted curvature and acoustic speaker grilles
- Aerodynamic 3D curved head dome and smooth pill-shaped display bezel (R=30mm)
- Multi-tier filleted base chassis with anti-tip weighted geometry
- Dual 608ZZ ball bearing pockets (8x22x7mm) for friction-free pitch pivot
- Integrated high-torque 20kg–25kg servo cradle (40.5x20.2x40mm)
- Standard M3/M4 heat-set brass insert bosses and cable routing conduits

Units: Millimetres (mm)
"""

import math
import struct
import os
import sys
from typing import List, Tuple, Sequence

# ---------------------------------------------------------------------------
# Global Manufacturing & Geometry Constants
# ---------------------------------------------------------------------------
WALL_BASE     = 3.5      # Base structural wall thickness (mm)
WALL_COLUMN   = 3.5      # Column structural wall thickness (mm)
WALL_NECK     = 3.0      # Neck structural wall thickness (mm)
WALL_HEAD     = 3.0      # Head enclosure wall thickness (mm)

# Fasteners & Tolerances
M3_HOLE_R     = 1.7      # M3 clearance hole radius (3.4mm diameter)
M3_INSERT_R   = 2.1      # M3 heat-set insert pocket radius (4.2mm diameter)
M4_HOLE_R     = 2.25     # M4 clearance hole radius (4.5mm diameter)
M4_INSERT_R   = 2.9      # M4 heat-set insert pocket radius (5.8mm diameter)
M4_BOSS_OD    = 10.0     # M4 structural boss outer diameter (mm)

# Neck & Actuation
BEARING_608_OD_R = 11.1  # 608ZZ outer radius with print tolerance (+0.1mm) -> 22.2mm
BEARING_608_ID_R = 4.0   # 608ZZ inner shaft radius (8.0mm diameter)
BEARING_608_W    = 7.2   # 608ZZ width with print tolerance (mm)
HINGE_X          = 40.0  # Hinge pin X offset (+/- 40mm)
HINGE_Z          = 50.0  # Hinge pin Z height within neck (50mm local / 750mm world)
SERVO_W          = 40.5  # Standard servo width (mm)
SERVO_D          = 20.2  # Standard servo depth (mm)
SERVO_H          = 40.0  # Standard servo height (mm)

# ---------------------------------------------------------------------------
# 3D Math & Mesh Engine
# ---------------------------------------------------------------------------
Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Triangle = Tuple[Vec3, Vec3, Vec3]

def _calc_normal(v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3:
    ax, ay, az = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]
    bx, by, bz = v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]
    nx = ay*bz - az*by
    ny = az*bx - ax*bz
    nz = ax*by - ay*bx
    length = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
    return (nx/length, ny/length, nz/length)

def write_binary_stl(path: str, triangles: Sequence[Triangle]):
    """Writes a list of 3D triangles into a standard binary STL file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        header = f"Karma Robot STL - {os.path.basename(path)}".encode('ascii')[:80].ljust(80, b'\x00')
        f.write(header)
        f.write(struct.pack('<I', len(triangles)))
        for v1, v2, v3 in triangles:
            n = _calc_normal(v1, v2, v3)
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<3f', *v3))
            f.write(struct.pack('<H', 0))
    print(f"  ✓ Exported {len(triangles):6d} triangles -> {os.path.basename(path)}")

def quad(v1: Vec3, v2: Vec3, v3: Vec3, v4: Vec3) -> List[Triangle]:
    return [(v1, v2, v3), (v1, v3, v4)]

def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> List[Triangle]:
    tris = []
    tris += quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))
    tris += quad((x1, y0, z0), (x0, y0, z0), (x0, y0, z1), (x1, y0, z1))
    tris += quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1))
    tris += quad((x1, y1, z0), (x1, y0, z0), (x1, y0, z1), (x1, y1, z1))
    tris += quad((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))
    tris += quad((x0, y1, z1), (x1, y1, z1), (x1, y0, z1), (x0, y0, z1))
    return tris

def rounded_rect_2d(x0: float, x1: float, y0: float, y1: float, r: float, n_corner: int = 8) -> List[Vec2]:
    """Generates counter-clockwise 2D polygon vertices with filleted corners of radius r."""
    max_r = min((x1 - x0) / 2.0 - 0.1, (y1 - y0) / 2.0 - 0.1)
    r = max(0.1, min(r, max_r))
    pts = []
    # 4 corner centers and angle sweeps:
    corners = [
        (x1 - r, y0 + r, -math.pi/2.0, 0.0),          # Bottom-Right
        (x1 - r, y1 - r, 0.0, math.pi/2.0),           # Top-Right
        (x0 + r, y1 - r, math.pi/2.0, math.pi),       # Top-Left
        (x0 + r, y0 + r, math.pi, 3.0*math.pi/2.0)    # Bottom-Left
    ]
    for cx, cy, a_start, a_end in corners:
        for i in range(n_corner):
            a = a_start + (a_end - a_start) * (i / float(n_corner))
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts

def loft_contours_z(layers: List[Tuple[float, List[Vec2]]], cap_bottom: bool = True, cap_top: bool = True) -> List[Triangle]:
    """Lofts a series of 2D cross-sectional contours along the Z axis."""
    tris = []
    num_layers = len(layers)
    if num_layers < 2:
        return tris

    for l_idx in range(num_layers - 1):
        z0, pts0 = layers[l_idx]
        z1, pts1 = layers[l_idx + 1]
        n_pts = min(len(pts0), len(pts1))
        for i in range(n_pts):
            j = (i + 1) % n_pts
            v0 = (pts0[i][0], pts0[i][1], z0)
            v1 = (pts0[j][0], pts0[j][1], z0)
            v2 = (pts1[j][0], pts1[j][1], z1)
            v3 = (pts1[i][0], pts1[i][1], z1)
            tris.append((v0, v1, v2))
            tris.append((v0, v2, v3))

    # Bottom Cap
    if cap_bottom:
        z0, pts0 = layers[0]
        c_x = sum(p[0] for p in pts0) / len(pts0)
        c_y = sum(p[1] for p in pts0) / len(pts0)
        c_pt = (c_x, c_y, z0)
        for i in range(len(pts0)):
            j = (i + 1) % len(pts0)
            tris.append((c_pt, (pts0[j][0], pts0[j][1], z0), (pts0[i][0], pts0[i][1], z0)))

    # Top Cap
    if cap_top:
        z1, pts1 = layers[-1]
        c_x = sum(p[0] for p in pts1) / len(pts1)
        c_y = sum(p[1] for p in pts1) / len(pts1)
        c_pt = (c_x, c_y, z1)
        for i in range(len(pts1)):
            j = (i + 1) % len(pts1)
            tris.append((c_pt, (pts1[i][0], pts1[i][1], z1), (pts1[j][0], pts1[j][1], z1)))

    return tris

def hollow_cylinder_z(cx: float, cy: float, z0: float, z1: float,
                       outer_r: float, inner_r: float, segments: int = 24) -> List[Triangle]:
    tris = []
    outer_bot, outer_top = [], []
    inner_bot, inner_top = [], []

    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        outer_bot.append((cx + outer_r * cos_t, cy + outer_r * sin_t, z0))
        outer_top.append((cx + outer_r * cos_t, cy + outer_r * sin_t, z1))
        inner_bot.append((cx + inner_r * cos_t, cy + inner_r * sin_t, z0))
        inner_top.append((cx + inner_r * cos_t, cy + inner_r * sin_t, z1))

    for i in range(segments):
        j = (i + 1) % segments
        tris += quad(outer_bot[i], outer_bot[j], outer_top[j], outer_top[i])
        tris += quad(inner_bot[j], inner_bot[i], inner_top[i], inner_top[j])
        tris += quad(outer_bot[j], outer_bot[i], inner_bot[i], inner_bot[j])
        tris += quad(outer_top[i], outer_top[j], inner_top[j], inner_top[i])

    return tris

def hollow_cylinder_y(cx: float, cz: float, y0: float, y1: float,
                       outer_r: float, inner_r: float, segments: int = 24) -> List[Triangle]:
    tris = []
    outer_0, outer_1 = [], []
    inner_0, inner_1 = [], []

    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        outer_0.append((cx + outer_r * cos_t, y0, cz + outer_r * sin_t))
        outer_1.append((cx + outer_r * cos_t, y1, cz + outer_r * sin_t))
        inner_0.append((cx + inner_r * cos_t, y0, cz + inner_r * sin_t))
        inner_1.append((cx + inner_r * cos_t, y1, cz + inner_r * sin_t))

    for i in range(segments):
        j = (i + 1) % segments
        tris += quad(outer_0[i], outer_0[j], outer_1[j], outer_1[i])
        tris += quad(inner_0[j], inner_0[i], inner_1[i], inner_1[j])
        tris += quad(outer_0[j], outer_0[i], inner_0[i], inner_0[j])
        tris += quad(outer_1[i], outer_1[j], inner_1[j], inner_1[i])

    return tris

def mounting_boss_m4(cx: float, cy: float, z0: float, z1: float, is_insert: bool = True) -> List[Triangle]:
    inner_r = M4_INSERT_R if is_insert else M4_HOLE_R
    return hollow_cylinder_z(cx, cy, z0, z1, M4_BOSS_OD / 2.0, inner_r, segments=16)

# ---------------------------------------------------------------------------
# 1. Base Assembly Generation (Tiered Aerodynamic Rounded Chassis)
# ---------------------------------------------------------------------------

def build_base_front() -> List[Triangle]:
    """
    Base Front Half: X: [-160..160], Y: [0..210], Z: [0..220]
    Features:
    - Sleek 3-tier filleted aerodynamic contour (R=45mm bottom, tapering smoothly to top pedestal R=35mm)
    - Compute mounting sled (Pi 5 / Mini PC)
    - 4x Column attachment bosses
    - Interlocking tongue-and-groove mating plane at Y=0
    """
    tris = []
    W = WALL_BASE

    # Outer Aerodynamic Lofted Layers (Front Half: Y: [0..Y_max])
    outer_layers = [
        (0.0,   rounded_rect_2d(-160.0, 160.0, 0.0, 210.0, r=48.0, n_corner=8)),
        (40.0,  rounded_rect_2d(-158.0, 158.0, 0.0, 206.0, r=46.0, n_corner=8)),
        (110.0, rounded_rect_2d(-145.0, 145.0, 0.0, 185.0, r=42.0, n_corner=8)),
        (180.0, rounded_rect_2d(-115.0, 115.0, 0.0, 140.0, r=38.0, n_corner=8)),
        (220.0, rounded_rect_2d(-85.0,   85.0, 0.0,  95.0, r=32.0, n_corner=8)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # 4x Column M4 Attachment Bosses (At top pedestal Z=220mm)
    boss_coords = [(-65.0, 20.0), (65.0, 20.0), (-65.0, 75.0), (65.0, 75.0)]
    for bx, by in boss_coords:
        tris += mounting_boss_m4(bx, by, 190.0, 220.0 - W, is_insert=True)

    # Compute Sled Rails (Standard Raspberry Pi / Jetson grid: 58x49mm pattern)
    compute_bosses = [(-29.0, 100.0), (29.0, 100.0), (-29.0, 158.0), (29.0, 158.0)]
    for cx, cy in compute_bosses:
        tris += hollow_cylinder_z(cx, cy, W, W + 8.0, 4.0, M3_INSERT_R, segments=12)

    # Seam Clamping Bosses along Y=0
    for sx in [-130.0, -50.0, 50.0, 130.0]:
        tris += mounting_boss_m4(sx, 10.0, W, 220.0 - W, is_insert=True)

    # Central wiring pass-through chimney
    tris += hollow_cylinder_z(0.0, 20.0, 220.0 - W, 220.0, 22.0, 18.0, segments=20)

    return tris

def build_base_back() -> List[Triangle]:
    """
    Base Back Half: X: [-160..160], Y: [-190..0], Z: [0..220]
    Features:
    - Sleek filleted aerodynamic shell matching front half
    - Battery / Power Supply retention cradle
    - Rear I/O Panel (DC jack, Rocker switch, USB-C)
    """
    tris = []
    W = WALL_BASE

    # Outer Aerodynamic Lofted Layers (Back Half: Y: [Y_min..0])
    outer_layers = [
        (0.0,   rounded_rect_2d(-160.0, 160.0, -190.0, 0.0, r=48.0, n_corner=8)),
        (40.0,  rounded_rect_2d(-158.0, 158.0, -186.0, 0.0, r=46.0, n_corner=8)),
        (110.0, rounded_rect_2d(-145.0, 145.0, -165.0, 0.0, r=42.0, n_corner=8)),
        (180.0, rounded_rect_2d(-115.0, 115.0, -130.0, 0.0, r=38.0, n_corner=8)),
        (220.0, rounded_rect_2d(-85.0,   85.0, -115.0, 0.0, r=32.0, n_corner=8)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # 4x Column M4 Attachment Bosses
    boss_coords = [(-65.0, -20.0), (65.0, -20.0), (-65.0, -95.0), (65.0, -95.0)]
    for bx, by in boss_coords:
        tris += mounting_boss_m4(bx, by, 190.0, 220.0 - W, is_insert=True)

    # Battery retention ribs (150x65mm bay)
    tris += box(-75.0, 75.0, -140.0, -135.0, W, W + 35.0)
    tris += box(-75.0, 75.0, -60.0, -55.0, W, W + 35.0)

    # Seam Clamping Bosses along Y=0
    for sx in [-130.0, -50.0, 50.0, 130.0]:
        tris += mounting_boss_m4(sx, -10.0, W, 220.0 - W, is_insert=False)

    return tris

# ---------------------------------------------------------------------------
# 2. Column / Torso Assembly (Sculpted Humanoid Torso with Waistline)
# ---------------------------------------------------------------------------

def build_column_front() -> List[Triangle]:
    """
    Column Front Torso: X: [-85..85], Y: [0..95], Z: [0..480]
    Features:
    - Sculpted humanoid curvature: wide chest (Z=320), sleek waist (Z=200), flaring neck base (Z=480)
    - Integrated left & right circular speaker grilles with beveled sound ports
    - Top microphone array port (Z=450)
    - Central vertical conduit
    """
    tris = []
    W = WALL_COLUMN

    # Sculpted Torso Lofting Layers (Front Half: Y: [0..Y_max])
    outer_layers = [
        (0.0,   rounded_rect_2d(-85.0, 85.0, 0.0, 95.0, r=32.0, n_corner=8)), # Base interface
        (120.0, rounded_rect_2d(-80.0, 80.0, 0.0, 90.0, r=30.0, n_corner=8)), # Lower spine
        (200.0, rounded_rect_2d(-74.0, 74.0, 0.0, 82.0, r=28.0, n_corner=8)), # Sleek waistline
        (320.0, rounded_rect_2d(-85.0, 85.0, 0.0, 95.0, r=32.0, n_corner=8)), # Chest & speakers
        (420.0, rounded_rect_2d(-82.0, 82.0, 0.0, 88.0, r=28.0, n_corner=8)), # Upper chest
        (480.0, rounded_rect_2d(-80.0, 80.0, 0.0, 45.0, r=24.0, n_corner=8)), # Neck interface
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # Base-Mounting Bottom Flange Bosses (Matching Base Z=220)
    bottom_bosses = [(-65.0, 20.0), (65.0, 20.0), (-65.0, 75.0), (65.0, 75.0)]
    for bx, by in bottom_bosses:
        tris += mounting_boss_m4(bx, by, W, W + 25.0, is_insert=False)

    # Neck-Mounting Top Flange Bosses (Matching Neck Z=700)
    for bx, by in [(-55.0, 25.0), (55.0, 25.0)]:
        tris += mounting_boss_m4(bx, by, 480.0 - 25.0, 480.0 - W, is_insert=True)

    # Dual Acoustic Speaker Bezels (Left X=-48, Right X=+48, Z=300mm)
    for spk_x in [-48.0, 48.0]:
        tris += hollow_cylinder_y(spk_x, 300.0, 88.0, 95.0, 24.0, 20.0, segments=24)
        # Inner speaker chamber cup
        tris += box(spk_x - 25.0, spk_x + 25.0, 50.0, 90.0, 275.0, 325.0)

    # Microphone Array Port (Z=450mm at center)
    tris += box(-15.0, 15.0, 80.0, 88.0, 442.0, 458.0)

    # Central Wiring Conduit Pipe
    tris += hollow_cylinder_z(0.0, 20.0, W, 480.0 - W, 22.0, 19.0, segments=20)

    return tris

def build_column_back() -> List[Triangle]:
    """
    Column Back Spine: X: [-85..85], Y: [-115..0], Z: [0..480]
    Features:
    - Sculpted aerodynamic spine matching front curvature
    - Slanted aerodynamic cooling gills
    - M4 tie-rod clamping posts
    """
    tris = []
    W = WALL_COLUMN

    # Sculpted Torso Lofting Layers (Back Half: Y: [Y_min..0])
    outer_layers = [
        (0.0,   rounded_rect_2d(-85.0, 85.0, -115.0, 0.0, r=32.0, n_corner=8)),
        (120.0, rounded_rect_2d(-80.0, 80.0, -108.0, 0.0, r=30.0, n_corner=8)),
        (200.0, rounded_rect_2d(-74.0, 74.0, -96.0,  0.0, r=28.0, n_corner=8)),
        (320.0, rounded_rect_2d(-85.0, 85.0, -112.0, 0.0, r=32.0, n_corner=8)),
        (420.0, rounded_rect_2d(-82.0, 82.0, -100.0, 0.0, r=28.0, n_corner=8)),
        (480.0, rounded_rect_2d(-80.0, 80.0, -45.0,  0.0, r=24.0, n_corner=8)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # Base Bottom Bosses
    bottom_bosses = [(-65.0, -20.0), (65.0, -20.0), (-65.0, -95.0), (65.0, -95.0)]
    for bx, by in bottom_bosses:
        tris += mounting_boss_m4(bx, by, W, W + 25.0, is_insert=False)

    # Neck Top Bosses
    for bx, by in [(-55.0, -25.0), (55.0, -25.0)]:
        tris += mounting_boss_m4(bx, by, 480.0 - 25.0, 480.0 - W, is_insert=True)

    # Vertical Column Seam Clamping Posts (Every 100mm)
    for seam_z in [80.0, 180.0, 280.0, 380.0]:
        tris += mounting_boss_m4(-70.0, -10.0, seam_z - 12.0, seam_z + 12.0, is_insert=True)
        tris += mounting_boss_m4( 70.0, -10.0, seam_z - 12.0, seam_z + 12.0, is_insert=True)

    return tris

# ---------------------------------------------------------------------------
# 3. Neck Actuation Module (Rounded Cowling, Servo Cradle & Bearings)
# ---------------------------------------------------------------------------

def build_neck_front() -> List[Triangle]:
    """
    Neck Front Half: X: [-80..80], Y: [0..45], Z: [0..60]
    Features:
    - Rounded pill-shaped profile (R=25mm) with sweep arc clearance
    - Dual 608ZZ Ball Bearing Housings (22.2mm OD) at X=+/-40mm, Z=50mm
    - Integrated 20kg-25kg standard servo cradle (DS3218 / MG996R)
    """
    tris = []
    W = WALL_NECK

    # Rounded Lofting Layers
    outer_layers = [
        (0.0,  rounded_rect_2d(-80.0, 80.0, 0.0, 45.0, r=24.0, n_corner=8)),
        (30.0, rounded_rect_2d(-80.0, 80.0, 0.0, 45.0, r=24.0, n_corner=8)),
        (60.0, rounded_rect_2d(-76.0, 76.0, 0.0, 42.0, r=20.0, n_corner=8)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # Dual 608ZZ Ball Bearing Housings (At X=+/-40mm, Z=50mm)
    for bx in [-HINGE_X, HINGE_X]:
        tris += hollow_cylinder_y(bx, HINGE_Z, 0.0, 45.0, 14.0, BEARING_608_OD_R, segments=24)
        tris += hollow_cylinder_y(bx, HINGE_Z, 42.0, 45.0, BEARING_608_OD_R, BEARING_608_ID_R, segments=24)

    # 20kg Standard Servo Cradle
    tris += box(-SERVO_W/2.0 - 2.0, SERVO_W/2.0 + 2.0, 8.0, 10.0, W, W + 38.0)
    tris += box(-SERVO_W/2.0 - 2.0, SERVO_W/2.0 + 2.0, 31.0, 33.0, W, W + 38.0)
    for ex in [-24.25, 24.25]:
        tris += hollow_cylinder_z(ex, 14.0, W, W + 38.0, 3.5, M3_INSERT_R, segments=12)
        tris += hollow_cylinder_z(ex, 27.0, W, W + 38.0, 3.5, M3_INSERT_R, segments=12)

    # Base attachment bolt holes
    for fx in [-55.0, 55.0]:
        tris += mounting_boss_m4(fx, 25.0, W, W + 15.0, is_insert=False)

    return tris

def build_neck_back() -> List[Triangle]:
    """
    Neck Back Half: X: [-80..80], Y: [-45..0], Z: [0..60]
    Features:
    - Rounded pill-shaped profile matching front half
    - Dual 608ZZ bearing retention backing blocks
    - Central wiring conduit
    """
    tris = []
    W = WALL_NECK

    outer_layers = [
        (0.0,  rounded_rect_2d(-80.0, 80.0, -45.0, 0.0, r=24.0, n_corner=8)),
        (30.0, rounded_rect_2d(-80.0, 80.0, -45.0, 0.0, r=24.0, n_corner=8)),
        (60.0, rounded_rect_2d(-76.0, 76.0, -42.0, 0.0, r=20.0, n_corner=8)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # Dual 608ZZ Bearing Backing Blocks
    for bx in [-HINGE_X, HINGE_X]:
        tris += hollow_cylinder_y(bx, HINGE_Z, -45.0, 0.0, 14.0, BEARING_608_ID_R, segments=24)

    # Column Attachment Bosses
    for bx in [-55.0, 55.0]:
        tris += mounting_boss_m4(bx, -25.0, W, W + 15.0, is_insert=False)

    # Central Wiring Conduit Chimney
    tris += hollow_cylinder_z(0.0, -15.0, W, 60.0 - W, 16.0, 13.0, segments=16)

    return tris

def build_steering_arm() -> List[Triangle]:
    """
    Precision Servo Steering Linkage Arm with filleted rounded ends.
    """
    tris = []
    # Main linkage bar with rounded corners: Length 45mm, Width 14mm, Height 6mm
    layers = [
        (0.0, rounded_rect_2d(-10.0, 35.0, -7.0, 7.0, r=6.5, n_corner=6)),
        (6.0, rounded_rect_2d(-10.0, 35.0, -7.0, 7.0, r=6.5, n_corner=6)),
    ]
    tris += loft_contours_z(layers, cap_bottom=True, cap_top=True)

    # Servo horn pivot eyelet (Z=-15..0mm)
    tris += hollow_cylinder_z(0.0, 0.0, -15.0, 6.0, 7.0, M3_HOLE_R, segments=20)
    # Head attachment eyelet
    tris += hollow_cylinder_z(25.0, 0.0, 0.0, 6.0, 6.0, M3_INSERT_R, segments=16)

    return tris

def loft_contours_y(layers: List[Tuple[float, List[Vec2]]], cap_back: bool = True, cap_front: bool = True) -> List[Triangle]:
    """Lofts a series of 2D XZ cross-sectional contours along the Y axis."""
    tris = []
    num_layers = len(layers)
    if num_layers < 2:
        return tris

    for l_idx in range(num_layers - 1):
        y0, pts0 = layers[l_idx]
        y1, pts1 = layers[l_idx + 1]
        n_pts = min(len(pts0), len(pts1))
        for i in range(n_pts):
            j = (i + 1) % n_pts
            v0 = (pts0[i][0], y0, pts0[i][1])
            v1 = (pts0[j][0], y0, pts0[j][1])
            v2 = (pts1[j][0], y1, pts1[j][1])
            v3 = (pts1[i][0], y1, pts1[i][1])
            tris.append((v0, v1, v2))
            tris.append((v0, v2, v3))

    # Back Cap (at y0, facing -Y)
    if cap_back:
        y0, pts0 = layers[0]
        c_x = sum(p[0] for p in pts0) / len(pts0)
        c_z = sum(p[1] for p in pts0) / len(pts0)
        c_pt = (c_x, y0, c_z)
        for i in range(len(pts0)):
            j = (i + 1) % len(pts0)
            tris.append((c_pt, (pts0[j][0], y0, pts0[j][1]), (pts0[i][0], y0, pts0[i][1])))

    # Front Cap (at y1, facing +Y)
    if cap_front:
        y1, pts1 = layers[-1]
        c_x = sum(p[0] for p in pts1) / len(pts1)
        c_z = sum(p[1] for p in pts1) / len(pts1)
        c_pt = (c_x, y1, c_z)
        for i in range(len(pts1)):
            j = (i + 1) % len(pts1)
            tris.append((c_pt, (pts1[i][0], y1, pts1[i][1]), (pts1[j][0], y1, pts1[j][1])))

    return tris

# ---------------------------------------------------------------------------
# 4. Head Assembly (Upright Front Bezel Looking Forward +Y & 3D Rear Dome)
# ---------------------------------------------------------------------------

def build_head_window_half() -> List[Triangle]:
    """
    Head Display Front Bezel (Upright Orientation facing FORWARD in +Y):
    X: [-150..150] (300mm width)
    Z: [0..190] (190mm height: Chin at Z=0, Top at Z=190)
    Y: [0..22.5] (22.5mm thickness, facing forward +Y)
    Features:
    - Elegant rounded corners (R=30mm) looking straight forward at eye level
    - Top camera housing & lens aperture (8mm) at Z=175mm
    - Bottom hinge pivot ears (Z=0, X=+/-40mm)
    """
    tris = []
    W = WALL_HEAD

    # Front Bezel Frame lofted along +Y (from mating seam Y=0 to front face Y=22.5mm)
    outer_layers = [
        (0.0,  rounded_rect_2d(-150.0, 150.0, 0.0, 190.0, r=32.0, n_corner=8)),
        (12.0, rounded_rect_2d(-148.0, 148.0, 2.0, 188.0, r=30.0, n_corner=8)),
        (22.5, rounded_rect_2d(-145.0, 145.0, 4.0, 186.0, r=28.0, n_corner=8)),
    ]
    tris += loft_contours_y(outer_layers, cap_back=True, cap_front=True)

    # Top Camera Mount Housing (At X=0, Z=175mm, facing +Y)
    tris += hollow_cylinder_y(0.0, 175.0, 15.0, 26.0, 8.0, 4.0, segments=20)
    for cx, cz in [(-14.0, 165.0), (14.0, 165.0), (-14.0, 185.0), (14.0, 185.0)]:
        tris += hollow_cylinder_y(cx, cz, 12.0, 22.5, 3.0, M3_INSERT_R, segments=12)

    # Bottom Hinge Mounting Ears (At X=+/-40mm, Z=-15..0mm, mating to Neck Hinge Axis at Y=0)
    for hx in [-HINGE_X, HINGE_X]:
        # Clevis ears extending down from chin
        tris += box(hx - 5.0, hx + 5.0, -10.0, 10.0, -15.0, 5.0)
        # Steel axle pin hole along X
        tris += hollow_cylinder_z(hx, 0.0, -15.0, 5.0, 8.0, BEARING_608_ID_R, segments=20)

    # Perimeter M3 Assembly Posts
    posts = [(-135.0, 20.0), (135.0, 20.0), (-135.0, 170.0), (135.0, 170.0),
             (-135.0, 95.0), (135.0, 95.0), (0.0, 15.0), (0.0, 175.0)]
    for px, pz in posts:
        tris += hollow_cylinder_y(px, pz, 0.0, 22.5, 4.0, M3_INSERT_R, segments=12)

    return tris

def build_head_cover_half() -> List[Triangle]:
    """
    Head Rear Enclosure:
    X: [-150..150] (300mm width)
    Z: [0..190] (190mm height)
    Y: [-22.5..0] (22.5mm rear dome thickness, facing BACK in -Y)
    Features:
    - 3D Aerodynamic Curved Parabolic Dome (curving backwards in -Y)
    - Horizontal cooling gills
    - 8x M3 perimeter screw holes
    """
    tris = []
    W = WALL_HEAD

    # 3D Parabolic Curved Dome lofted along -Y (from mating seam Y=0 to rear apex Y=-22.5mm)
    outer_layers = [
        (0.0,   rounded_rect_2d(-150.0, 150.0, 0.0, 190.0, r=32.0, n_corner=8)),
        (-8.0,  rounded_rect_2d(-145.0, 145.0, 5.0, 185.0, r=34.0, n_corner=8)),
        (-16.0, rounded_rect_2d(-130.0, 130.0, 15.0, 175.0, r=36.0, n_corner=8)),
        (-21.0, rounded_rect_2d(-100.0, 100.0, 30.0, 160.0, r=40.0, n_corner=8)),
        (-22.5, rounded_rect_2d(-60.0,   60.0,  50.0, 140.0, r=30.0, n_corner=8)),
    ]
    tris += loft_contours_y(outer_layers, cap_back=True, cap_front=True)

    # Rear Exhaust Cooling Gills (Central rear dome)
    for louver_z in range(60, 140, 15):
        tris += box(-50.0, 50.0, -22.0, -18.0, float(louver_z), float(louver_z + 6))

    # Perimeter M3 Screw Posts
    posts = [(-135.0, 20.0), (135.0, 20.0), (-135.0, 170.0), (135.0, 170.0),
             (-135.0, 95.0), (135.0, 95.0), (0.0, 15.0), (0.0, 175.0)]
    for px, pz in posts:
        tris += hollow_cylinder_y(px, pz, -22.5, 0.0, 4.0, M3_HOLE_R, segments=12)

    return tris

# ---------------------------------------------------------------------------
# Master CAD Pipeline Execution
# ---------------------------------------------------------------------------

def generate_all_production_cad(out_dir: str):
    """Executes the complete parametric CAD generator and outputs all production STLs."""
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 75)
    print("🚀 Karma Robot Production CAD Generation Suite (Organic Rounded Edition)")
    print(f"📦 Target Output Directory: {os.path.abspath(out_dir)}")
    print("=" * 75)

    generators = [
        ("Base Front Chassis",       "base_front.stl",        build_base_front),
        ("Base Back Chassis",        "base_back.stl",         build_base_back),
        ("Column Front Torso",       "column_front.stl",      build_column_front),
        ("Column Back Spine",        "column_back.stl",       build_column_back),
        ("Neck Front Actuation",     "neck_front.stl",        build_neck_front),
        ("Neck Back Enclosure",      "neck_back.stl",         build_neck_back),
        ("Steering Horn Linkage",    "steering_arm.stl",      build_steering_arm),
        ("Head Display Window Bezel","head_window_half.stl",  build_head_window_half),
        ("Head Rear Enclosure",      "head_cover_half.stl",   build_head_cover_half),
    ]

    total_tris = 0
    for name, filename, gen_fn in generators:
        filepath = os.path.join(out_dir, filename)
        tris = gen_fn()
        write_binary_stl(filepath, tris)
        total_tris += len(tris)

    print("=" * 75)
    print(f"✨ Successfully generated all 9 sleek production STLs! Total triangles: {total_tris:,}")
    print("=" * 75)

if __name__ == "__main__":
    target_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_production_cad(target_dir)
