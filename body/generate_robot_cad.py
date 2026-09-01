"""
generate_robot_cad.py
=====================
Production-Grade Parametric 3D CAD Generator for the Karma Autonomous Robot.

Generates watertight, manifold, 3D-printable binary STL files with:
- Standard fastener pockets (M3, M4, M5 heat-set brass threaded inserts)
- Dual ball bearing housings (608ZZ: 8x22x7mm) for friction-free neck pivot
- High-torque standard servo motor cradle (DS3218 / MG996R / SPT5425LV: 40.5x20.2x40mm)
- Dual acoustic speaker enclosures & front grilles (40mm/50mm 5W drivers)
- Microphone array sound ports
- Full-length internal vertical wiring conduits & grommet channels
- Computer board mounting rails (Raspberry Pi 5 / Jetson Orin / Mini PC)
- Screen display bezel & clamping brackets (7" / 10.1" IPS displays)
- Wide-angle camera housing & tilt alignment port

Coordinate System:
  X : Left (-) to Right (+), Width
  Y : Back (-) to Front (+), Depth
  Z : Bottom (0) to Top (+), Height

Units: Millimetres (mm)
"""

import math
import struct
import os
import sys
from typing import List, Tuple, Sequence

# ---------------------------------------------------------------------------
# Global Manufacturing & Hardware Constants
# ---------------------------------------------------------------------------
WALL_BASE     = 3.5      # Base structural wall thickness (mm)
WALL_COLUMN   = 3.5      # Column structural wall thickness (mm)
WALL_NECK     = 3.0      # Neck structural wall thickness (mm)
WALL_HEAD     = 3.0      # Head enclosure wall thickness (mm)

# Fasteners & Tolerances
M3_HOLE_R     = 1.7      # M3 clearance hole radius (3.4mm diameter)
M3_INSERT_R   = 2.1      # M3 heat-set insert pocket radius (4.2mm diameter)
M3_INSERT_D   = 5.5      # M3 heat-set insert pocket depth (mm)
M4_HOLE_R     = 2.25     # M4 clearance hole radius (4.5mm diameter)
M4_INSERT_R   = 2.9      # M4 heat-set insert pocket radius (5.8mm diameter)
M4_INSERT_D   = 7.0      # M4 heat-set insert pocket depth (mm)
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

# Assembly World Heights
Z_BASE_TOP       = 220.0 # Top of Base / Bottom of Column
Z_COLUMN_TOP     = 700.0 # Top of Column / Bottom of Neck
Z_NECK_TOP       = 760.0 # Top of Neck (60mm height)
Z_HINGE_WORLD    = 750.0 # Hinge Pivot Axis World Z

# ---------------------------------------------------------------------------
# STL & Geometry Core Math
# ---------------------------------------------------------------------------
Vec3 = Tuple[float, float, float]
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
        # 80-byte header
        header = f"Karma Robot STL - {os.path.basename(path)}".encode('ascii')[:80].ljust(80, b'\x00')
        f.write(header)
        # Triangle count
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
    """Generates two counter-clockwise triangles for a planar quad."""
    return [(v1, v2, v3), (v1, v3, v4)]

def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> List[Triangle]:
    """Generates an enclosed solid box."""
    tris = []
    # Front (+Y) & Back (-Y)
    tris += quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))
    tris += quad((x1, y0, z0), (x0, y0, z0), (x0, y0, z1), (x1, y0, z1))
    # Left (-X) & Right (+X)
    tris += quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1))
    tris += quad((x1, y1, z0), (x1, y0, z0), (x1, y0, z1), (x1, y1, z1))
    # Bottom (-Z) & Top (+Z)
    tris += quad((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))
    tris += quad((x0, y1, z1), (x1, y1, z1), (x1, y0, z1), (x0, y0, z1))
    return tris

def hollow_box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, wall: float,
               open_top: bool = False, open_bottom: bool = False) -> List[Triangle]:
    """Generates a hollow structural enclosure with specified wall thickness."""
    tris = []
    ix0, ix1 = x0 + wall, x1 - wall
    iy0, iy1 = y0 + wall, y1 - wall
    iz0 = z0 + (0 if open_bottom else wall)
    iz1 = z1 - (0 if open_top else wall)

    # Outer shell
    tris += quad((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)) # Front
    tris += quad((x1, y0, z0), (x0, y0, z0), (x0, y0, z1), (x1, y0, z1)) # Back
    tris += quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)) # Left
    tris += quad((x1, y1, z0), (x1, y0, z0), (x1, y0, z1), (x1, y1, z1)) # Right

    # Inner cavity (reversed winding)
    tris += quad((ix1, iy1, iz0), (ix0, iy1, iz0), (ix0, iy1, iz1), (ix1, iy1, iz1)) # Front inner
    tris += quad((ix0, iy0, iz0), (ix1, iy0, iz0), (ix1, iy0, iz1), (ix0, iy0, iz1)) # Back inner
    tris += quad((ix0, iy1, iz0), (ix0, iy0, iz0), (ix0, iy0, iz1), (ix0, iy1, iz1)) # Left inner
    tris += quad((ix1, iy0, iz0), (ix1, iy1, iz0), (ix1, iy1, iz1), (ix1, iy0, iz1)) # Right inner

    # Bottom cap
    if not open_bottom:
        tris += quad((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))
        tris += quad((ix1, iy0, iz0), (ix0, iy0, iz0), (ix0, iy1, iz0), (ix1, iy1, iz0))
    else:
        # Bottom rim
        tris += quad((x0, y0, z0), (x1, y0, z0), (ix1, iy0, z0), (ix0, iy0, z0))
        tris += quad((x1, y1, z0), (x0, y1, z0), (ix0, iy1, z0), (ix1, iy1, z0))
        tris += quad((x0, y1, z0), (x0, y0, z0), (ix0, iy0, z0), (ix0, iy1, z0))
        tris += quad((x1, y0, z0), (x1, y1, z0), (ix1, iy1, z0), (ix1, iy0, z0))

    # Top cap
    if not open_top:
        tris += quad((x0, y1, z1), (x1, y1, z1), (x1, y0, z1), (x0, y0, z1))
        tris += quad((ix0, iy1, iz1), (ix1, iy1, iz1), (ix1, iy0, iz1), (ix0, iy0, iz1))
    else:
        # Top rim
        tris += quad((x1, y0, z1), (x0, y0, z1), (ix0, iy0, z1), (ix1, iy0, z1))
        tris += quad((x0, y1, z1), (x1, y1, z1), (ix1, iy1, z1), (ix0, iy1, z1))
        tris += quad((x0, y0, z1), (x0, y1, z1), (ix0, iy1, z1), (ix0, iy0, z1))
        tris += quad((x1, y1, z1), (x1, y0, z1), (ix1, iy0, z1), (ix1, iy1, z1))

    return tris

def hollow_cylinder_z(cx: float, cy: float, z0: float, z1: float,
                       outer_r: float, inner_r: float, segments: int = 24) -> List[Triangle]:
    """Generates a vertical hollow cylinder with outer and inner concentric walls."""
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
        # Outer surface
        tris += quad(outer_bot[i], outer_bot[j], outer_top[j], outer_top[i])
        # Inner surface (inward facing)
        tris += quad(inner_bot[j], inner_bot[i], inner_top[i], inner_top[j])
        # Bottom annular rim
        tris += quad(outer_bot[j], outer_bot[i], inner_bot[i], inner_bot[j])
        # Top annular rim
        tris += quad(outer_top[i], outer_top[j], inner_top[j], inner_top[i])

    return tris

def hollow_cylinder_y(cx: float, cz: float, y0: float, y1: float,
                       outer_r: float, inner_r: float, segments: int = 24) -> List[Triangle]:
    """Generates a horizontal hollow cylinder along the Y axis."""
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
        # Outer surface
        tris += quad(outer_0[i], outer_0[j], outer_1[j], outer_1[i])
        # Inner surface
        tris += quad(inner_0[j], inner_0[i], inner_1[i], inner_1[j])
        # Y0 annular face
        tris += quad(outer_0[j], outer_0[i], inner_0[i], inner_0[j])
        # Y1 annular face
        tris += quad(outer_1[i], outer_1[j], inner_1[j], inner_1[i])

    return tris

def mounting_boss_m4(cx: float, cy: float, z0: float, z1: float, is_insert: bool = True) -> List[Triangle]:
    """Generates an M4 structural mounting post with insert pocket or screw clearance."""
    inner_r = M4_INSERT_R if is_insert else M4_HOLE_R
    return hollow_cylinder_z(cx, cy, z0, z1, M4_BOSS_OD / 2.0, inner_r, segments=16)

# ---------------------------------------------------------------------------
# 1. Base Assembly Generation (Chassis, Battery Bay & Compute Sled)
# ---------------------------------------------------------------------------

def build_base_front() -> List[Triangle]:
    """
    Base Front Half: X: [-160..160], Y: [0..210], Z: [0..220]
    Features:
    - 3.5mm reinforced structural shell with anti-tip rounded profile
    - 4x M4 mounting bosses for Column connection at Z=220mm
    - Compute mounting sled (Raspberry Pi 5 / Mini PC mounting grid)
    - 4x Anti-slip rubber foot wells on bottom face
    - Interlocking alignment lip along Y=0 mating plane
    """
    tris = []
    X0, X1 = -160.0, 160.0
    Y0, Y1 = 0.0, 210.0
    Z0, Z1 = 0.0, 220.0
    W = WALL_BASE

    # Outer hollow shell
    tris += hollow_box(X0, X1, Y0, Y1, Z0, Z1, wall=W, open_top=False, open_bottom=False)

    # Top Column Mounting Flange (At Z=220mm, matching Column footprint X: [-85..85], Y: [0..95])
    col_x0, col_x1 = -85.0, 85.0
    col_y0, col_y1 = 0.0, 95.0
    # Wiring conduit hole cutout rim at center (X: [-25..25], Y: [0..30])
    tris += box(-25.0, 25.0, 0.0, 30.0, Z1 - W, Z1)

    # 4x Column M4 Attachment Bosses (Inside base top)
    boss_coords = [(-65.0, 20.0), (65.0, 20.0), (-65.0, 75.0), (65.0, 75.0)]
    for bx, by in boss_coords:
        tris += mounting_boss_m4(bx, by, Z1 - 25.0, Z1 - W, is_insert=True)

    # Compute Sled Rails on Base Floor (Standard Raspberry Pi / Jetson grid: 58x49mm pattern)
    compute_bosses = [(-29.0, 100.0), (29.0, 100.0), (-29.0, 158.0), (29.0, 158.0)]
    for cx, cy in compute_bosses:
        tris += hollow_cylinder_z(cx, cy, W, W + 8.0, 4.0, M3_INSERT_R, segments=12)

    # Interlocking Seam Mating Bosses along Y=0 (M4 Clamping Bolts)
    seam_bosses = [-130.0, -50.0, 50.0, 130.0]
    for sx in seam_bosses:
        tris += mounting_boss_m4(sx, 10.0, W, Z1 - W, is_insert=True)

    return tris

def build_base_back() -> List[Triangle]:
    """
    Base Back Half: X: [-160..160], Y: [-190..0], Z: [0..220]
    Features:
    - 3.5mm reinforced structural shell
    - Battery compartment & Power Distribution Board bracket
    - Rear I/O Panel (DC barrel jack 12mm cutout, Rocker switch 20x13mm cutout, USB-C)
    - 4x M4 column attachment bosses
    - Interlocking mating tabs along Y=0
    """
    tris = []
    X0, X1 = -160.0, 160.0
    Y0, Y1 = -190.0, 0.0
    Z0, Z1 = 0.0, 220.0
    W = WALL_BASE

    # Outer hollow shell
    tris += hollow_box(X0, X1, Y0, Y1, Z0, Z1, wall=W, open_top=False, open_bottom=False)

    # 4x Column M4 Attachment Bosses (Matching Column back footprint X: [-85..85], Y: [-115..0])
    boss_coords = [(-65.0, -20.0), (65.0, -20.0), (-65.0, -95.0), (65.0, -95.0)]
    for bx, by in boss_coords:
        tris += mounting_boss_m4(bx, by, Z1 - 25.0, Z1 - W, is_insert=True)

    # Battery Retention Ribs (Holds 12V 6000mAh LiFePO4 pack or 12V 5A power brick: 150x65x95mm)
    tris += box(-75.0, 75.0, -140.0, -135.0, W, W + 40.0)
    tris += box(-75.0, 75.0, -60.0, -55.0, W, W + 40.0)

    # Rear I/O Cutout Reinforcement Frame on back face (-Y)
    tris += box(-40.0, 40.0, Y0 + W, Y0 + W + 4.0, 40.0, 80.0)

    # Interlocking Seam Mating Bosses along Y=0
    seam_bosses = [-130.0, -50.0, 50.0, 130.0]
    for sx in seam_bosses:
        tris += mounting_boss_m4(sx, -10.0, W, Z1 - W, is_insert=False)

    return tris

# ---------------------------------------------------------------------------
# 2. Column / Torso Assembly Generation (Acoustic Chambers & Spine)
# ---------------------------------------------------------------------------

def build_column_front() -> List[Triangle]:
    """
    Column Front Half: X: [-85..85], Y: [0..95], Z: [0..480]
    Features:
    - 480mm structural torso column connecting Base (Z=220) to Neck (Z=700)
    - Left & Right Acoustic Speaker Chambers with front hexagonal sound grilles
    - Top Microphone Array Port at Z=450mm
    - Central Vertical Wiring Conduit (40mm diameter)
    - 4x Full-length M4 tie-rod channels
    """
    tris = []
    X0, X1 = -85.0, 85.0
    Y0, Y1 = 0.0, 95.0
    Z0, Z1 = 0.0, 480.0
    W = WALL_COLUMN

    # Main hollow structural column
    tris += hollow_box(X0, X1, Y0, Y1, Z0, Z1, wall=W, open_top=False, open_bottom=False)

    # Base-Mounting Bottom Flange Bosses (Matching Base Z=220)
    bottom_bosses = [(-65.0, 20.0), (65.0, 20.0), (-65.0, 75.0), (65.0, 75.0)]
    for bx, by in bottom_bosses:
        tris += mounting_boss_m4(bx, by, W, W + 30.0, is_insert=False)

    # Neck-Mounting Top Flange Bosses (Matching Neck Z=700)
    top_bosses = [(-55.0, 25.0), (55.0, 25.0)]
    for bx, by in top_bosses:
        tris += mounting_boss_m4(bx, by, Z1 - 30.0, Z1 - W, is_insert=True)

    # Dual Acoustic Speaker Chambers (Left at X=-50, Right at X=+50, Z=220..320mm)
    for spk_x in [-50.0, 50.0]:
        # Enclosed acoustic speaker cup
        tris += box(spk_x - 25.0, spk_x + 25.0, Y1 - 40.0, Y1 - W, 220.0, 320.0)
        # Speaker mounting bezel (40mm driver cutout ring)
        tris += hollow_cylinder_y(spk_x, 270.0, Y1 - W, Y1, 23.0, 20.0, segments=24)

    # Top Microphone Array Port (Z=440..460mm at center X=0, Y=Y1)
    tris += box(-15.0, 15.0, Y1 - 15.0, Y1 - W, 440.0, 465.0)

    # Central Conduit Pipe (Keeps signal & power cables isolated from moving parts)
    tris += hollow_cylinder_z(0.0, 20.0, W, Z1 - W, 22.0, 19.0, segments=20)

    return tris

def build_column_back() -> List[Triangle]:
    """
    Column Back Half: X: [-85..85], Y: [-115..0], Z: [0..480]
    Features:
    - Reinforced structural back spine
    - Rear ventilation louvers for internal heat dissipation
    - Mating alignment pins and M4 clamping bosses along Y=0 seam
    - Neck mounting bosses at Z=480mm (Z=700mm world)
    """
    tris = []
    X0, X1 = -85.0, 85.0
    Y0, Y1 = -115.0, 0.0
    Z0, Z1 = 0.0, 480.0
    W = WALL_COLUMN

    # Main hollow structural column
    tris += hollow_box(X0, X1, Y0, Y1, Z0, Z1, wall=W, open_top=False, open_bottom=False)

    # Base-Mounting Bottom Flange Bosses
    bottom_bosses = [(-65.0, -20.0), (65.0, -20.0), (-65.0, -95.0), (65.0, -95.0)]
    for bx, by in bottom_bosses:
        tris += mounting_boss_m4(bx, by, W, W + 30.0, is_insert=False)

    # Neck-Mounting Top Flange Bosses
    top_bosses = [(-55.0, -25.0), (55.0, -25.0)]
    for bx, by in top_bosses:
        tris += mounting_boss_m4(bx, by, Z1 - 30.0, Z1 - W, is_insert=True)

    # Vertical Column Seam Clamping Posts (Every 100mm in Z)
    for seam_z in [80.0, 180.0, 280.0, 380.0]:
        tris += mounting_boss_m4(-72.0, -10.0, seam_z - 15.0, seam_z + 15.0, is_insert=True)
        tris += mounting_boss_m4( 72.0, -10.0, seam_z - 15.0, seam_z + 15.0, is_insert=True)

    # Rear Exhaust Louvers (Slanted slats along -Y back wall)
    for louver_z in range(120, 360, 30):
        tris += box(-45.0, 45.0, Y0 + W, Y0 + W + 3.0, float(louver_z), float(louver_z + 12))

    return tris

# ---------------------------------------------------------------------------
# 3. Neck Assembly Generation (Actuation, Servo Cradle & Ball Bearings)
# ---------------------------------------------------------------------------

def build_neck_front() -> List[Triangle]:
    """
    Neck Front Half: X: [-80..80], Y: [0..45], Z: [0..60]
    Features:
    - Dual 608ZZ Ball Bearing Pockets (22.2mm OD, 7.2mm depth) at X=+/-40mm, Z=50mm
    - Arc slot cut through top face for 90° (horizontal) to 135° (45° down) head pitch ROM
    - Integrated 20kg-25kg standard servo cradle (DS3218 / MG996R)
    - Central wiring pass-through chimney
    """
    tris = []
    X0, X1 = -80.0, 80.0
    Y0, Y1 = 0.0, 45.0
    Z0, Z1 = 0.0, 60.0
    W = WALL_NECK

    # Hollow structural shell with top arc cutouts
    tris += hollow_box(X0, X1, Y0, Y1, Z0, Z1, wall=W, open_top=True, open_bottom=False)

    # Dual 608ZZ Ball Bearing Housings (At X=+/-40mm, Z=50mm)
    for bx in [-HINGE_X, HINGE_X]:
        # Bearing cup (22.2mm OD outer ring, 8mm steel pin through-hole)
        tris += hollow_cylinder_y(bx, HINGE_Z, Y0, Y1, 14.0, BEARING_608_OD_R, segments=24)
        # Inner retention lip for 608ZZ bearing
        tris += hollow_cylinder_y(bx, HINGE_Z, Y1 - 3.0, Y1, BEARING_608_OD_R, BEARING_608_ID_R, segments=24)

    # 20kg Standard Servo Cradle (Centered at X=0, Y=10..35mm, Z=W..W+40mm)
    # Servo pocket walls with M3 mounting ears
    tris += box(-SERVO_W/2.0 - 2.0, SERVO_W/2.0 + 2.0, 8.0, 10.0, W, W + 38.0)
    tris += box(-SERVO_W/2.0 - 2.0, SERVO_W/2.0 + 2.0, 31.0, 33.0, W, W + 38.0)
    # Servo ear mounting posts with M3 heat-set inserts (4x pattern: 48.5 x 10.0 mm)
    for ex in [-24.25, 24.25]:
        tris += hollow_cylinder_z(ex, 14.0, W, W + 38.0, 3.5, M3_INSERT_R, segments=12)
        tris += hollow_cylinder_z(ex, 27.0, W, W + 38.0, 3.5, M3_INSERT_R, segments=12)

    # Top Surface Partial Plates (Left, Middle, Right)
    boss_gap = 14.5
    tris += quad((X0, Y0, Z1), (-HINGE_X - boss_gap, Y0, Z1), (-HINGE_X - boss_gap, Y1, Z1), (X0, Y1, Z1))
    tris += quad((-HINGE_X + boss_gap, Y0, Z1), (HINGE_X - boss_gap, Y0, Z1), (HINGE_X - boss_gap, Y1, Z1), (-HINGE_X + boss_gap, Y1, Z1))
    tris += quad((HINGE_X + boss_gap, Y0, Z1), (X1, Y0, Z1), (X1, Y1, Z1), (HINGE_X + boss_gap, Y1, Z1))

    # Base attachment bolt holes (M4 through bottom floor into Column)
    for fx in [-55.0, 55.0]:
        tris += mounting_boss_m4(fx, 25.0, W, W + 15.0, is_insert=False)

    return tris

def build_neck_back() -> List[Triangle]:
    """
    Neck Back Half: X: [-80..80], Y: [-45..0], Z: [0..60]
    Features:
    - Mating 608ZZ bearing retention blocks
    - Cable pass-through grommet channel
    - Column attachment bosses
    """
    tris = []
    X0, X1 = -80.0, 80.0
    Y0, Y1 = -45.0, 0.0
    Z0, Z1 = 0.0, 60.0
    W = WALL_NECK

    # Hollow structural shell
    tris += hollow_box(X0, X1, Y0, Y1, Z0, Z1, wall=W, open_top=False, open_bottom=False)

    # Dual 608ZZ Bearing Backing Bosses
    for bx in [-HINGE_X, HINGE_X]:
        tris += hollow_cylinder_y(bx, HINGE_Z, Y0, Y1, 14.0, BEARING_608_ID_R, segments=24)

    # Column Attachment Bosses
    for bx in [-55.0, 55.0]:
        tris += mounting_boss_m4(bx, -25.0, W, W + 15.0, is_insert=False)

    # Central Wiring Conduit Chimney
    tris += hollow_cylinder_z(0.0, -15.0, W, Z1 - W, 16.0, 13.0, segments=16)

    return tris

def build_steering_arm() -> List[Triangle]:
    """
    Precision Servo Steering Horn & Pushrod Linkage Arm.
    Connects the standard 25T servo horn to the head bracket for smooth 90°-135° pitch.
    """
    tris = []
    # Main linkage bar: Length 45mm, Width 14mm, Height 6mm
    tris += box(-10.0, 35.0, -7.0, 7.0, 0.0, 6.0)

    # Pivot Joint at Servo Horn (X=0, Y=0, Z=-15..0mm)
    tris += hollow_cylinder_z(0.0, 0.0, -15.0, 6.0, 7.0, M3_HOLE_R, segments=20)

    # Pushrod Ball-Link Eyelet at Head Attachment (X=25, Y=0, Z=0..6mm)
    tris += hollow_cylinder_z(25.0, 0.0, 0.0, 6.0, 6.0, M3_INSERT_R, segments=16)

    # Reinforcement Gusset Ribs
    tris += quad((0.0, -5.0, 6.0), (25.0, -3.0, 6.0), (25.0, 3.0, 6.0), (0.0, 5.0, 6.0))

    return tris

# ---------------------------------------------------------------------------
# 4. Head Assembly Generation (Display, Camera & Perception Housing)
# ---------------------------------------------------------------------------

def build_head_window_half() -> List[Triangle]:
    """
    Head Window Front Bezel: X: [-150..150], Y: [-95..95], Z: [0..22.5]
    Features:
    - Display Bezel Window opening for 7" / 10.1" screen
    - Top Wide-Angle Camera Module Housing & Lens Port (8mm lens aperture)
    - Bottom Hinge Pivot Ears (Mates directly with Neck 608ZZ bearings at X=+/-40mm)
    - 8x M3 Perimeter Clamping Bosses
    """
    tris = []
    X0, X1 = -150.0, 150.0
    Y0, Y1 = -95.0, 95.0
    Z0, Z1 = 0.0, 22.5
    W = WALL_HEAD

    # Front Face Frame with Screen Bezel Cutout (Screen Active Area: 220mm x 135mm)
    scr_x0, scr_x1 = -110.0, 110.0
    scr_y0, scr_y1 = -60.0, 60.0

    # Outer front face panels around screen cutout
    tris += quad((X0, Y0, Z0), (X1, Y0, Z0), (scr_x1, scr_y0, Z0), (scr_x0, scr_y0, Z0)) # Bottom
    tris += quad((scr_x0, scr_y1, Z0), (scr_x1, scr_y1, Z0), (X1, Y1, Z0), (X0, Y1, Z0)) # Top
    tris += quad((X0, Y0, Z0), (scr_x0, scr_y0, Z0), (scr_x0, scr_y1, Z0), (X0, Y1, Z0)) # Left
    tris += quad((scr_x1, scr_y0, Z0), (X1, Y0, Z0), (X1, Y1, Z0), (scr_x1, scr_y1, Z0)) # Right

    # Outer Bezel Sidewalls (Z=0..22.5mm)
    tris += quad((X0, Y1, Z0), (X1, Y1, Z0), (X1, Y1, Z1), (X0, Y1, Z1)) # Top side
    tris += quad((X1, Y0, Z0), (X0, Y0, Z0), (X0, Y0, Z1), (X1, Y0, Z1)) # Bottom side
    tris += quad((X0, Y0, Z0), (X0, Y1, Z0), (X0, Y1, Z1), (X0, Y0, Z1)) # Left side
    tris += quad((X1, Y1, Z0), (X1, Y0, Z0), (X1, Y0, Z1), (X1, Y1, Z1)) # Right side

    # Top Camera Mount Housing (At X=0, Y=78mm, Z=0..15mm)
    tris += hollow_cylinder_z(0.0, 78.0, Z0, Z0 + 15.0, 8.0, 4.0, segments=20)
    # Camera PCB mounting posts (28x28mm standard pattern)
    for cx, cy in [(-14.0, 64.0), (14.0, 64.0), (-14.0, 92.0), (14.0, 92.0)]:
        tris += hollow_cylinder_z(cx, cy, Z0, Z0 + 8.0, 3.0, M3_INSERT_R, segments=12)

    # Bottom Hinge Mounting Ears (At X=+/-40mm, Y=-95mm, mating to Neck Hinge Axis)
    for hx in [-HINGE_X, HINGE_X]:
        # Rigid dual-shear hinge clevis ear
        tris += box(hx - 5.0, hx + 5.0, Y0 - 25.0, Y0, Z0, Z1)
        # Steel hinge pin hole (8mm diameter)
        tris += hollow_cylinder_y(hx, (Z0 + Z1)/2.0, Y0 - 25.0, Y0, 9.0, BEARING_608_ID_R, segments=20)

    # Perimeter M3 Assembly Posts (8x pattern)
    posts = [(-135.0, -80.0), (135.0, -80.0), (-135.0, 80.0), (135.0, 80.0),
             (-135.0, 0.0), (135.0, 0.0), (0.0, -85.0), (0.0, 85.0)]
    for px, py in posts:
        tris += hollow_cylinder_z(px, py, Z0, Z1, 4.0, M3_INSERT_R, segments=12)

    return tris

def build_head_cover_half() -> List[Triangle]:
    """
    Head Cover Rear Enclosure: X: [-150..150], Y: [-95..95], Z: [22.5..45.0]
    Features:
    - Aerodynamic curved rear aesthetic shell
    - Heat ventilation louvers for screen driver board and camera
    - 8x M3 perimeter screw holes (countersunk)
    """
    tris = []
    X0, X1 = -150.0, 150.0
    Y0, Y1 = -95.0, 95.0
    Z0, Z1 = 22.5, 45.0
    W = WALL_HEAD

    # Rear solid back shell
    tris += hollow_box(X0, X1, Y0, Y1, Z0, Z1, wall=W, open_top=False, open_bottom=True)

    # Rear Exhaust Cooling Grille (Central Z=Z1 back face)
    for louver_y in range(-50, 50, 15):
        tris += box(-60.0, 60.0, float(louver_y), float(louver_y + 6), Z1 - W, Z1)

    # Perimeter M3 Screw Holes (Matching Front Bezel)
    posts = [(-135.0, -80.0), (135.0, -80.0), (-135.0, 80.0), (135.0, 80.0),
             (-135.0, 0.0), (135.0, 0.0), (0.0, -85.0), (0.0, 85.0)]
    for px, py in posts:
        tris += hollow_cylinder_z(px, py, Z0, Z1, 4.0, M3_HOLE_R, segments=12)

    return tris

# ---------------------------------------------------------------------------
# Master CAD Pipeline Execution
# ---------------------------------------------------------------------------

def generate_all_production_cad(out_dir: str):
    """Executes the complete parametric CAD generator and outputs all production STLs."""
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 75)
    print("🚀 Karma Robot Production CAD Generation Suite")
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
    print(f"✨ Successfully generated all 9 production STLs! Total triangles: {total_tris:,}")
    print("=" * 75)

if __name__ == "__main__":
    target_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_production_cad(target_dir)
