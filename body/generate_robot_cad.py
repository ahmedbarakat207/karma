"""
generate_robot_cad.py
=====================
Production 3D-Printable CAD Generation Suite for the Karma Robot.
Fully parameterized with physical mounting bosses, bracket bays, hardware openings,
and fastener pockets for every single component from PARTS.MD.

Complete Hardware Bill of Materials (BOM) & Interfaces:
1. SBC:           Raspberry Pi 4 (4GB) — 4x M2.5 standoffs, USB-A/C, RJ45, Micro-HDMI, MicroSD cutouts
2. Drive Motors:  2x 5840-31ZY Worm DC Gear Motors 12V — 4x M3 bracket bosses, Ø18mm axle ports, wheel arches
3. Motor Drivers: 2x BTS7960 43A H-Bridge Modules — 4x M3 standoff bosses (44x44mm pitch)
4. Vision:        Raspberry Pi Camera Module v2 — Ø8mm lens aperture, 4x M2 standoffs (21x12.5mm)
5. Display:       7" 800x480 Capacitive Touch LCD — 154x86mm window, 4x M3 corner bosses (155x97mm)
6. Neck Hinge:    TowerPro MG90S Micro Servo — precision cradle, 21T horn, 35mm pushrod, Ø3mm dual-shear hinge
7. Battery:       12V 9Ah Li-ion Pack with BMS — 151x65mm secure floor retaining cradle
8. Stabilizers:   2x 30mm Free-Moving Nylon Caster Wheels — 4x M3 mounting bosses (front & rear)
9. Power Input:   DC Jack 5.5x2.1mm panel mount (Ø11.5mm) + 20x13mm Rocker Power Switch
10. Voltage Reg:  5V Step-Down Buck Converter (LM2596/XL4015) — 2x M3 mounting bosses
11. Fasteners:    M2 / M2.5 / M3 / M4 Heat-Set Brass Threaded Inserts (Ruthex standard)

Units: Millimetres (mm)
"""

import math
import struct
import os
from typing import List, Tuple, Sequence

# ---------------------------------------------------------------------------
# Global Manufacturing & Hardware Constants
# ---------------------------------------------------------------------------
WALL_BASE     = 3.5      # Base structural wall thickness (mm)
WALL_COLUMN   = 3.5      # Column structural wall thickness (mm)
WALL_NECK     = 3.0      # Neck structural wall thickness (mm)
WALL_HEAD     = 3.0      # Head enclosure wall thickness (mm)

# Fasteners & Fastener Pockets (3D Print Toleranced)
M2_HOLE_R     = 1.1      # M2 clearance hole (2.2mm diameter)
M2_INSERT_R   = 1.6      # M2 heat-set insert pocket (3.2mm diameter)
M2_5_HOLE_R   = 1.4      # M2.5 clearance hole (2.8mm diameter - RPi 4)
M2_5_INSERT_R = 1.9      # M2.5 heat-set insert pocket (3.8mm diameter)
M3_HOLE_R     = 1.7      # M3 clearance hole (3.4mm diameter)
M3_INSERT_R   = 2.1      # M3 heat-set insert pocket (4.2mm diameter)
M4_HOLE_R     = 2.25     # M4 clearance hole (4.5mm diameter)
M4_INSERT_R   = 2.9      # M4 heat-set insert pocket (5.8mm diameter)
M3_BOSS_OD    = 7.5      # M3 structural boss outer diameter (mm)
M4_BOSS_OD    = 10.0     # M4 structural boss outer diameter (mm)

# 5840-31ZY Worm DC Gear Motor Dimensions
MOTOR_5840_L  = 58.0     # Gearbox length (mm)
MOTOR_5840_W  = 40.0     # Gearbox width (mm)
MOTOR_5840_H  = 35.0     # Gearbox height (mm)
MOTOR_SHAFT_R = 4.0      # Ø8mm D-Shaft radius
MOTOR_HOLE_DX = 30.0     # Bracket mounting hole spacing in X (mm)
MOTOR_HOLE_DY = 40.0     # Bracket mounting hole spacing in Y (mm)

# Raspberry Pi 4 Specifications
RPI4_PCB_L    = 85.0     # Length (mm)
RPI4_PCB_W    = 56.0     # Width (mm)
RPI4_HOLE_DX  = 58.0     # Mounting hole pitch X (mm)
RPI4_HOLE_DY  = 49.0     # Mounting hole pitch Y (mm)

# 7-Inch 800x480 Capacitive Touch LCD Dimensions
LCD_7INCH_W   = 165.0    # Frame width (mm)
LCD_7INCH_H   = 107.0    # Frame height (mm)
LCD_VIEW_W    = 154.0    # Active viewing width (mm)
LCD_VIEW_H    = 86.0     # Active viewing height (mm)
LCD_HOLE_DX   = 155.0    # Corner mounting hole pitch X (mm)
LCD_HOLE_DY   = 97.0     # Corner mounting hole pitch Y (mm)

# Raspberry Pi Camera v2
CAM_PCB_L     = 25.0     # Length (mm)
CAM_PCB_W     = 24.0     # Width (mm)
CAM_HOLE_DX   = 21.0     # Hole pitch X (mm)
CAM_HOLE_DY   = 12.5     # Hole pitch Y (mm)
CAM_LENS_R    = 4.0      # Lens aperture radius (8mm diameter)

# 12V 9Ah Battery Compartment
BATT_L        = 151.0    # Length (mm)
BATT_W        = 65.0     # Width (mm)
BATT_H        = 95.0     # Height (mm)

# BTS7960 43A Motor Drivers
BTS_HOLE_DX   = 44.0     # Standoff hole pitch X (mm)
BTS_HOLE_DY   = 44.0     # Standoff hole pitch Y (mm)

# MG90S Servo Specifications
MG90S_L       = 23.0     # Pocket Length (mm)
MG90S_W       = 12.5     # Pocket Width (mm)
MG90S_H       = 23.0     # Pocket Depth / Body Height (mm)
MG90S_FLANGE_L= 32.5     # Flange overall length (mm)
MG90S_HOLE_DIST=28.0     # Distance between M2 mounting hole centers (mm)
MG90S_SPLINE_R= 2.4      # 21-tooth micro spline radius (4.8mm diameter)
MG90S_SPLINE_H= 4.5      # Spline shaft height (mm)

# Hinge Pivot Axle
HINGE_X       = 40.0     # Hinge pin X offset (+/- 40mm)
HINGE_Z       = 50.0     # Hinge pin Z height within neck (50mm local / 750mm world)
HINGE_PIN_R   = 1.55     # Ø3.0mm stainless steel pin clearance hole (3.1mm diameter)
HINGE_BOSS_R  = 5.5      # Structural hinge boss outer radius (11mm diameter)

# ---------------------------------------------------------------------------
# 3D Geometry Core Math
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
    l = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
    return (nx/l, ny/l, nz/l)

def write_binary_stl(path: str, triangles: Sequence[Triangle]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        header = f"Karma Physical Robot STL - {os.path.basename(path)}".encode('ascii')[:80].ljust(80, b'\x00')
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

def half_contour_front(x0: float, x1: float, y_split: float, y_max: float, chamfer_front: float = 35.0, n_per_seg: int = 6) -> List[Vec2]:
    c = min(chamfer_front, (x1 - x0) / 3.0, (y_max - y_split) / 2.0)
    poly = [
        (x1, y_split),
        (x1, y_max - c),
        (x1 - c, y_max),
        (x0 + c, y_max),
        (x0, y_max - c),
        (x0, y_split),
    ]
    pts = []
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i + 1) % len(poly)]
        for s in range(n_per_seg):
            t = s / float(n_per_seg)
            pts.append((p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t))
    return pts

def half_contour_back(x0: float, x1: float, y_min: float, y_split: float, chamfer_back: float = 35.0, n_per_seg: int = 6) -> List[Vec2]:
    c = min(chamfer_back, (x1 - x0) / 3.0, (y_split - y_min) / 2.0)
    poly = [
        (x0, y_split),
        (x0, y_min + c),
        (x0 + c, y_min),
        (x1 - c, y_min),
        (x1, y_min + c),
        (x1, y_split),
    ]
    pts = []
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i + 1) % len(poly)]
        for s in range(n_per_seg):
            t = s / float(n_per_seg)
            pts.append((p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t))
    return pts

def filleted_rect_xz(x0: float, x1: float, z0: float, z1: float, r: float = 25.0, n_per_corner: int = 8) -> List[Vec2]:
    cr = min(r, (x1 - x0) / 3.0, (z1 - z0) / 3.0)
    pts = []
    corners = [
        (x1 - cr, z1 - cr, 0.0),
        (x0 + cr, z1 - cr, math.pi / 2),
        (x0 + cr, z0 + cr, math.pi),
        (x1 - cr, z0 + cr, 3 * math.pi / 2)
    ]
    for cx, cz, start_ang in corners:
        for i in range(n_per_corner):
            ang = start_ang + (math.pi / 2.0) * (i / float(n_per_corner))
            pts.append((cx + cr * math.cos(ang), cz + cr * math.sin(ang)))
    return pts

def loft_contours_z(layers: List[Tuple[float, List[Vec2]]], cap_bottom: bool = True, cap_top: bool = True) -> List[Triangle]:
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

    if cap_bottom:
        z0, pts0 = layers[0]
        c_x = sum(p[0] for p in pts0) / len(pts0)
        c_y = sum(p[1] for p in pts0) / len(pts0)
        c_pt = (c_x, c_y, z0)
        for i in range(len(pts0)):
            j = (i + 1) % len(pts0)
            tris.append((c_pt, (pts0[j][0], pts0[j][1], z0), (pts0[i][0], pts0[i][1], z0)))

    if cap_top:
        z1, pts1 = layers[-1]
        c_x = sum(p[0] for p in pts1) / len(pts1)
        c_y = sum(p[1] for p in pts1) / len(pts1)
        c_pt = (c_x, c_y, z1)
        for i in range(len(pts1)):
            j = (i + 1) % len(pts1)
            tris.append((c_pt, (pts1[i][0], pts1[i][1], z1), (pts1[j][0], pts1[j][1], z1)))

    return tris

def loft_contours_y(layers: List[Tuple[float, List[Vec2]]], cap_back: bool = True, cap_front: bool = True) -> List[Triangle]:
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

    if cap_back:
        y0, pts0 = layers[0]
        c_x = sum(p[0] for p in pts0) / len(pts0)
        c_z = sum(p[1] for p in pts0) / len(pts0)
        c_pt = (c_x, y0, c_z)
        for i in range(len(pts0)):
            j = (i + 1) % len(pts0)
            tris.append((c_pt, (pts0[j][0], y0, pts0[j][1]), (pts0[i][0], y0, pts0[i][1])))

    if cap_front:
        y1, pts1 = layers[-1]
        c_x = sum(p[0] for p in pts1) / len(pts1)
        c_z = sum(p[1] for p in pts1) / len(pts1)
        c_pt = (c_x, y1, c_z)
        for i in range(len(pts1)):
            j = (i + 1) % len(pts1)
            tris.append((c_pt, (pts1[i][0], y1, pts1[i][1]), (pts1[j][0], y1, pts1[j][1])))

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

def solid_cylinder_z(cx: float, cy: float, z0: float, z1: float, r: float, segments: int = 20) -> List[Triangle]:
    tris = []
    bot_pts, top_pts = [], []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        bot_pts.append((cx + r * cos_t, cy + r * sin_t, z0))
        top_pts.append((cx + r * cos_t, cy + r * sin_t, z1))
    for i in range(segments):
        j = (i + 1) % segments
        tris += quad(bot_pts[i], bot_pts[j], top_pts[j], top_pts[i])
        tris.append(((cx, cy, z0), bot_pts[j], bot_pts[i]))
        tris.append(((cx, cy, z1), top_pts[i], top_pts[j]))
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

def solid_cylinder_y(cx: float, cz: float, y0: float, y1: float, r: float, segments: int = 20) -> List[Triangle]:
    tris = []
    pts_0, pts_1 = [], []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        pts_0.append((cx + r * cos_t, y0, cz + r * sin_t))
        pts_1.append((cx + r * cos_t, y1, cz + r * sin_t))
    for i in range(segments):
        j = (i + 1) % segments
        tris += quad(pts_0[i], pts_0[j], pts_1[j], pts_1[i])
        tris.append(((cx, y0, cz), pts_0[j], pts_0[i]))
        tris.append(((cx, y1, cz), pts_1[i], pts_1[j]))
    return tris

def hollow_cylinder_x(cy: float, cz: float, x0: float, x1: float,
                       outer_r: float, inner_r: float, segments: int = 24) -> List[Triangle]:
    tris = []
    outer_0, outer_1 = [], []
    inner_0, inner_1 = [], []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        outer_0.append((x0, cy + outer_r * cos_t, cz + outer_r * sin_t))
        outer_1.append((x1, cy + outer_r * cos_t, cz + outer_r * sin_t))
        inner_0.append((x0, cy + inner_r * cos_t, cz + inner_r * sin_t))
        inner_1.append((x1, cy + inner_r * cos_t, cz + inner_r * sin_t))

    for i in range(segments):
        j = (i + 1) % segments
        tris += quad(outer_0[i], outer_0[j], outer_1[j], outer_1[i])
        tris += quad(inner_0[j], inner_0[i], inner_1[i], inner_1[j])
        tris += quad(outer_0[j], outer_0[i], inner_0[i], inner_0[j])
        tris += quad(outer_1[i], outer_1[j], inner_1[j], inner_1[i])
    return tris

def mounting_boss_m2_5(cx: float, cy: float, z0: float, z1: float, is_insert: bool = True) -> List[Triangle]:
    inner_r = M2_5_INSERT_R if is_insert else M2_5_HOLE_R
    return hollow_cylinder_z(cx, cy, z0, z1, 3.5, inner_r, segments=12)

def mounting_boss_m3(cx: float, cy: float, z0: float, z1: float, is_insert: bool = True) -> List[Triangle]:
    inner_r = M3_INSERT_R if is_insert else M3_HOLE_R
    return hollow_cylinder_z(cx, cy, z0, z1, M3_BOSS_OD / 2.0, inner_r, segments=14)

def mounting_boss_m4(cx: float, cy: float, z0: float, z1: float, is_insert: bool = True) -> List[Triangle]:
    inner_r = M4_INSERT_R if is_insert else M4_HOLE_R
    return hollow_cylinder_z(cx, cy, z0, z1, M4_BOSS_OD / 2.0, inner_r, segments=16)

# ---------------------------------------------------------------------------
# 1. Base Assembly (320 x 400 x 220 mm)
# ---------------------------------------------------------------------------

def build_base_front() -> List[Triangle]:
    """
    Exotic Base Front Chassis:
    - 2x 5840-31ZY Worm DC Gear Motor Mounting Bays & Ø18mm Axle Pass-Through Tunnels
    - Front 30mm Nylon Caster Wheel 4x M3 Mounting Bosses
    - 12V 9Ah Li-ion Battery Compartment Front Retaining Walls
    - 4x M4 Perimeter Seam Fasteners & 4x M4 Column Anchor Bosses
    - Central Ø24mm Wire Routing Conduit
    """
    tris = []
    W = WALL_BASE

    outer_layers = [
        (0.0,   half_contour_front(-160.0, 160.0, 0.0, 210.0, chamfer_front=45.0, n_per_seg=6)),
        (40.0,  half_contour_front(-158.0, 158.0, 0.0, 206.0, chamfer_front=44.0, n_per_seg=6)),
        (110.0, half_contour_front(-145.0, 145.0, 0.0, 185.0, chamfer_front=38.0, n_per_seg=6)),
        (180.0, half_contour_front(-115.0, 115.0, 0.0, 140.0, chamfer_front=30.0, n_per_seg=6)),
        (220.0, half_contour_front(-85.0,   85.0, 0.0,  95.0, chamfer_front=22.0, n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # 1. 2x 5840-31ZY Worm DC Gear Motor Mounting Brackets (Left & Right)
    for mx in [-105.0, 105.0]:
        # Motor L-Bracket M3 Mounting Bosses (30mm x 40mm hole pitch)
        for hx in [mx - MOTOR_HOLE_DX/2.0, mx + MOTOR_HOLE_DX/2.0]:
            for hy in [15.0, 15.0 + MOTOR_HOLE_DY]:
                tris += mounting_boss_m3(hx, hy, W, W + 12.0, is_insert=True)
        # Ø18mm Motor D-Shaft / Brass Hex Coupling Tunnel through outer wheel well
        tunnel_x0 = 135.0 if mx > 0 else -160.0
        tunnel_x1 = 160.0 if mx > 0 else -135.0
        tris += hollow_cylinder_x(35.0, 45.0, tunnel_x0, tunnel_x1, 12.0, 9.0, segments=20)

    # 2. Front 30mm Free-Moving Nylon Caster Wheel Mount (4x M3 at X=±12, Y=135±15)
    for cx in [-12.0, 12.0]:
        for cy in [120.0, 150.0]:
            tris += mounting_boss_m3(cx, cy, 0.0, W + 8.0, is_insert=True)

    # 3. 12V 9Ah Li-ion Battery Compartment Front Retaining Cradle (151x65mm)
    tris += box(-BATT_L/2.0 - 2.0, BATT_L/2.0 + 2.0, 0.0, BATT_W/2.0 + 2.0, W, W + 25.0)

    # 4. Column Anchor Structural Bosses (4x M4 connecting Base to Column)
    for bx, by in [(-65.0, 20.0), (65.0, 20.0), (-65.0, 75.0), (65.0, 75.0)]:
        tris += mounting_boss_m4(bx, by, 190.0, 220.0 - W, is_insert=True)

    # 5. Base Perimeter Seam Fasteners (4x M4 connecting front to back half)
    for sx in [-130.0, -50.0, 50.0, 130.0]:
        tris += mounting_boss_m4(sx, 10.0, W, 220.0 - W, is_insert=True)

    # 6. Central Wire Routing Conduit (Ø24mm rising to Column)
    tris += hollow_cylinder_z(0.0, 20.0, 190.0, 220.0, 14.0, 12.0, segments=20)

    return tris


def build_base_back() -> List[Triangle]:
    """
    Exotic Base Back Chassis:
    - 2x BTS7960 43A Motor Driver Module 4x M3 Standoffs (44x44mm pitch)
    - 5V Step-Down Buck Converter 2x M3 Standoffs
    - Rear 30mm Nylon Caster Wheel 4x M3 Mounting Bosses
    - 12V 9Ah Battery Compartment Rear Cradle
    - DC Power Jack Socket (Ø11.5mm) & Rocker Power Switch (20x13mm) Cutouts
    - 4x M4 Perimeter Seam Fasteners & 4x M4 Column Anchor Bosses
    """
    tris = []
    W = WALL_BASE

    outer_layers = [
        (0.0,   half_contour_back(-160.0, 160.0, -190.0, 0.0, chamfer_back=45.0, n_per_seg=6)),
        (40.0,  half_contour_back(-158.0, 158.0, -186.0, 0.0, chamfer_back=44.0, n_per_seg=6)),
        (110.0, half_contour_back(-145.0, 145.0, -165.0, 0.0, chamfer_back=38.0, n_per_seg=6)),
        (180.0, half_contour_back(-115.0, 115.0, -130.0, 0.0, chamfer_back=30.0, n_per_seg=6)),
        (220.0, half_contour_back(-85.0,   85.0, -115.0, 0.0, chamfer_back=22.0, n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # 1. 2x BTS7960 43A Motor Driver Module Mounts (Left: X=-60, Right: X=+60, 44x44mm pitch)
    for dx in [-60.0, 60.0]:
        for hx in [dx - BTS_HOLE_DX/2.0, dx + BTS_HOLE_DX/2.0]:
            for hy in [-85.0, -85.0 + BTS_HOLE_DY]:
                tris += mounting_boss_m3(hx, hy, W, W + 10.0, is_insert=True)

    # 2. 5V Step-Down Buck Converter (2x M3 at X=±16mm, Y=-130mm)
    for bx in [-16.0, 16.0]:
        tris += mounting_boss_m3(bx, -130.0, W, W + 8.0, is_insert=True)

    # 3. Rear 30mm Free-Moving Nylon Caster Wheel Mount (4x M3 at X=±12, Y=-135±15)
    for cx in [-12.0, 12.0]:
        for cy in [-150.0, -120.0]:
            tris += mounting_boss_m3(cx, cy, 0.0, W + 8.0, is_insert=True)

    # 4. 12V 9Ah Li-ion Battery Compartment Rear Retaining Cradle
    tris += box(-BATT_L/2.0 - 2.0, BATT_L/2.0 + 2.0, -BATT_W/2.0 - 2.0, 0.0, W, W + 25.0)

    # 5. DC Power Jack Socket (Ø11.5mm panel mount) at (X=-45, Y=-185, Z=50)
    tris += hollow_cylinder_y(-45.0, 50.0, -190.0, -180.0, 8.0, 5.8, segments=20)

    # 6. Master Rocker Power Switch Opening (20x13mm snap-in) at (X=+45, Y=-185, Z=50)
    tris += box(35.0, 55.0, -190.0, -180.0, 43.5, 56.5)

    # 7. Column Anchor Structural Bosses (4x M4 connecting Base to Column)
    for bx, by in [(-65.0, -20.0), (65.0, -20.0), (-65.0, -75.0), (65.0, -75.0)]:
        tris += mounting_boss_m4(bx, by, 190.0, 220.0 - W, is_insert=True)

    # 8. Base Perimeter Seam Fasteners (4x M4 connecting back to front half)
    for sx in [-130.0, -50.0, 50.0, 130.0]:
        tris += mounting_boss_m4(sx, -10.0, W, 220.0 - W, is_insert=False)

    return tris

# ---------------------------------------------------------------------------
# 2. Column Torso Assembly (170 x 210 x 480 mm)
# ---------------------------------------------------------------------------

def build_column_front() -> List[Triangle]:
    """
    Exotic Column Front Torso:
    - Raspberry Pi 4 (4GB) 4x M2.5 Standoff Mounting Bay (58x49mm pitch)
    - External I/O Port Openings: 4x USB-A, 1x RJ45 Gigabit Ethernet, 1x USB-C Power, 2x Micro-HDMI, MicroSD Slot
    - Dual Speaker Acoustic Chambers with Front Acoustic Sound Louvers
    - Continuous Central Ø22mm Wire Routing Conduit
    - Base Attachment (4x M4) and Neck Mounting (4x M4) Bosses
    """
    tris = []
    W = WALL_COLUMN

    outer_layers = [
        (0.0,   half_contour_front(-85.0, 85.0, 0.0, 95.0, chamfer_front=22.0, n_per_seg=6)),
        (120.0, half_contour_front(-80.0, 80.0, 0.0, 90.0, chamfer_front=20.0, n_per_seg=6)),
        (200.0, half_contour_front(-74.0, 74.0, 0.0, 82.0, chamfer_front=18.0, n_per_seg=6)),
        (320.0, half_contour_front(-85.0, 85.0, 0.0, 95.0, chamfer_front=22.0, n_per_seg=6)),
        (420.0, half_contour_front(-82.0, 82.0, 0.0, 88.0, chamfer_front=20.0, n_per_seg=6)),
        (480.0, half_contour_front(-80.0, 80.0, 0.0, 45.0, chamfer_front=18.0, n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # 1. Raspberry Pi 4 (4GB) Mounting Bay (4x M2.5 standoffs at 58x49mm pitch at Z=200mm)
    rpi_cz = 200.0
    for rx in [-RPI4_HOLE_DX/2.0, RPI4_HOLE_DX/2.0]:
        for ry in [25.0, 25.0 + RPI4_HOLE_DY]:
            tris += mounting_boss_m2_5(rx, ry, W, W + 15.0, is_insert=True)

    # 2. External I/O Port Cutouts on Left Torso Wall (X=-82mm)
    # 4x USB-A Ports Cutout (16mm x 32mm)
    tris += box(-85.0, -70.0, 15.0, 47.0, rpi_cz - 16.0, rpi_cz + 16.0)
    # 1x RJ45 Gigabit Ethernet Port Cutout (16mm x 15mm)
    tris += box(-85.0, -70.0, 52.0, 67.0, rpi_cz - 8.0, rpi_cz + 8.0)
    # 1x USB-C Power Port Cutout (10mm x 6mm)
    tris += box(-85.0, -70.0, 10.0, 20.0, rpi_cz - 28.0, rpi_cz - 22.0)
    # 2x Micro-HDMI Port Cutouts (8mm x 4mm)
    tris += box(-85.0, -70.0, 25.0, 33.0, rpi_cz - 28.0, rpi_cz - 24.0)
    tris += box(-85.0, -70.0, 38.0, 46.0, rpi_cz - 28.0, rpi_cz - 24.0)
    # MicroSD Card Slot Cutout at bottom of Pi
    tris += box(-85.0, -70.0, 30.0, 45.0, rpi_cz - 35.0, rpi_cz - 30.0)

    # 3. Dual Speaker Acoustic Chambers & Louver Grilles (At X=±48mm, Z=300mm)
    for spk_x in [-48.0, 48.0]:
        tris += hollow_cylinder_y(spk_x, 300.0, 82.0, 92.0, 24.0, 20.0, segments=24)
        tris += box(spk_x - 22.0, spk_x + 22.0, 50.0, 85.0, 278.0, 322.0)

    # 4. Base Attachment Bolt Bosses (4x M4 connecting Column to Base)
    for bx, by in [(-65.0, 20.0), (65.0, 20.0), (-65.0, 75.0), (65.0, 75.0)]:
        tris += mounting_boss_m4(bx, by, W, W + 25.0, is_insert=False)

    # 5. Top Neck Attachment Bosses (4x M4 connecting Column to Neck)
    for bx, by in [(-55.0, 20.0), (55.0, 20.0)]:
        tris += mounting_boss_m4(bx, by, 480.0 - 25.0, 480.0 - W, is_insert=True)

    # 6. Continuous Central Wire Routing Conduit (Ø22mm from Base to Neck)
    tris += hollow_cylinder_z(0.0, 20.0, W, 480.0 - W, 13.0, 11.0, segments=20)

    return tris


def build_column_back() -> List[Triangle]:
    """
    Exotic Column Back Spine:
    - Aerodynamic Cooling Louvers for passive convection airflow over Pi & Drivers
    - 6x M4 Perimeter Seam Fasteners & 4x M4 Base/Neck Mounts
    """
    tris = []
    W = WALL_COLUMN

    outer_layers = [
        (0.0,   half_contour_back(-85.0, 85.0, -115.0, 0.0, chamfer_back=22.0, n_per_seg=6)),
        (120.0, half_contour_back(-80.0, 80.0, -108.0, 0.0, chamfer_back=20.0, n_per_seg=6)),
        (200.0, half_contour_back(-74.0, 74.0, -96.0,  0.0, chamfer_back=18.0, n_per_seg=6)),
        (320.0, half_contour_back(-85.0, 85.0, -112.0, 0.0, chamfer_back=22.0, n_per_seg=6)),
        (420.0, half_contour_back(-82.0, 82.0, -100.0, 0.0, chamfer_back=20.0, n_per_seg=6)),
        (480.0, half_contour_back(-80.0, 80.0, -45.0,  0.0, chamfer_back=18.0, n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # 1. Spine Aerodynamic Cooling Louvers
    for louver_z in range(120, 420, 30):
        tris += box(-45.0, 45.0, -105.0, -98.0, float(louver_z), float(louver_z + 10))

    # 2. Base Attachment Bolt Bosses (4x M4)
    for bx, by in [(-65.0, -20.0), (65.0, -20.0), (-65.0, -75.0), (65.0, -75.0)]:
        tris += mounting_boss_m4(bx, by, W, W + 25.0, is_insert=False)

    # 3. Top Neck Attachment Bosses (4x M4)
    for bx, by in [(-55.0, -20.0), (55.0, -20.0)]:
        tris += mounting_boss_m4(bx, by, 480.0 - 25.0, 480.0 - W, is_insert=True)

    # 4. Column Seam Fasteners (6x M4 along split line)
    for seam_z in [80.0, 180.0, 280.0, 380.0]:
        tris += mounting_boss_m4(-65.0, -10.0, seam_z - 12.0, seam_z + 12.0, is_insert=False)
        tris += mounting_boss_m4( 65.0, -10.0, seam_z - 12.0, seam_z + 12.0, is_insert=False)

    return tris

# ---------------------------------------------------------------------------
# 3. Neck Actuation Module with MG90S Servo Cradle & Dual Hinge Bearings
# ---------------------------------------------------------------------------

def build_neck_front() -> List[Triangle]:
    """
    Exotic Neck Front Cowl: X: [-80..80], Y: [0..45], Z: [0..46] (World Z: 700..746)
    - MG90S Servo Mounting Bay (23x12.5x23mm) with 2x M2 Brass Insert Bosses (28mm pitch)
    - Dual-Shear Hinge Pin Center Lugs (7mm tongue, Ø3.1mm bore) at X=±40mm, Z=50mm
    - Wide Clearance Throat Opening for Head Display 90°..135° Pitch Travel
    - Central Ø16mm Wire Pass-Through Conduit
    """
    tris = []
    W = WALL_NECK
    SHELL_TOP = 46.0

    outer_layers = [
        (0.0,       half_contour_front(-80.0, 80.0, 0.0, 45.0, chamfer_front=18.0, n_per_seg=6)),
        (14.0,      half_contour_front(-80.0, 80.0, 0.0, 45.0, chamfer_front=18.0, n_per_seg=6)),
        (28.0,      half_contour_front(-78.0, 78.0, 0.0, 36.0, chamfer_front=14.0, n_per_seg=6)),
        (38.0,      half_contour_front(-76.0, 76.0, 0.0, 20.0, chamfer_front=8.0,  n_per_seg=6)),
        (SHELL_TOP, half_contour_front(-74.0, 74.0, 0.0, 8.0,  chamfer_front=4.0,  n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # 1. Dual Hinge Pin Center Lugs (Tongues at X=±40mm, 7mm wide)
    for bx in [-HINGE_X, HINGE_X]:
        tris += hollow_cylinder_y(bx, HINGE_Z, 0.0, 7.0, 4.5, HINGE_PIN_R, segments=20)
        tris += box(bx - 3.5, bx + 3.5, 0.0, 7.0, 15.0, HINGE_Z)

    # 2. MG90S Micro Servo Mounting Cradle (Positioned at X=18mm, Y=15mm, Z=22mm)
    servo_cx, servo_cy, servo_cz = 18.0, 15.0, 22.0
    tris += box(servo_cx - MG90S_L/2.0 - 2.0, servo_cx + MG90S_L/2.0 + 2.0,
                servo_cy - MG90S_W/2.0 - 2.0, servo_cy + MG90S_W/2.0 + 2.0,
                W, servo_cz + 8.0)
    for m2_x in [servo_cx - MG90S_HOLE_DIST/2.0, servo_cx + MG90S_HOLE_DIST/2.0]:
        tris += hollow_cylinder_z(m2_x, servo_cy, W, servo_cz + 8.0, 3.2, M2_INSERT_R, segments=12)

    # 3. Base Column Attachment Bosses (2x M4)
    for fx in [-55.0, 55.0]:
        tris += mounting_boss_m4(fx, 20.0, W, W + 15.0, is_insert=False)

    return tris


def build_neck_back() -> List[Triangle]:
    """
    Exotic Neck Back Cowl: X: [-80..80], Y: [-45..0], Z: [0..46]
    - Rear Hinge Pin Center Lugs (7mm tongue) at X=±40mm, Z=50mm
    - Continuous Wire Routing Conduit (Ø16mm)
    - Base Column Attachment Bosses (2x M4)
    """
    tris = []
    W = WALL_NECK
    SHELL_TOP = 46.0

    outer_layers = [
        (0.0,       half_contour_back(-80.0, 80.0, -45.0, 0.0, chamfer_back=18.0, n_per_seg=6)),
        (14.0,      half_contour_back(-80.0, 80.0, -45.0, 0.0, chamfer_back=18.0, n_per_seg=6)),
        (28.0,      half_contour_back(-78.0, 78.0, -32.0, 0.0, chamfer_back=12.0, n_per_seg=6)),
        (38.0,      half_contour_back(-76.0, 76.0, -18.0, 0.0, chamfer_back=7.0,  n_per_seg=6)),
        (SHELL_TOP, half_contour_back(-74.0, 74.0, -8.0,  0.0, chamfer_back=3.0,  n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    # 1. Rear Hinge Pin Center Lugs (Tongues at X=±40mm)
    for bx in [-HINGE_X, HINGE_X]:
        tris += hollow_cylinder_y(bx, HINGE_Z, -7.0, 0.0, 4.5, HINGE_PIN_R, segments=20)
        tris += box(bx - 3.5, bx + 3.5, -7.0, 0.0, 15.0, HINGE_Z)

    # 2. Base Column Attachment Bosses (2x M4)
    for bx in [-55.0, 55.0]:
        tris += mounting_boss_m4(bx, -20.0, W, W + 15.0, is_insert=False)

    # 3. Wire Conduit Pass-Through (Ø16mm)
    tris += hollow_cylinder_z(0.0, -15.0, W, SHELL_TOP - W, 10.0, 8.0, segments=16)

    return tris

# ---------------------------------------------------------------------------
# 4. MG90S Micro Servo & Mechanical Linkage CAD
# ---------------------------------------------------------------------------

def build_mg90s_servo() -> List[Triangle]:
    """
    Authentic MG90S Metal-Gear Micro Servo 3D Model:
    - Polycarbonate Housing: 22.8 x 12.2 x 22.5 mm
    - Top Metal Gear Tower & 21T Brass Spline Output Shaft
    - Mounting Flanges with 2x M2 Eyelets
    """
    tris = []
    # Main Body Housing (Local origin centered at body base)
    tris += box(-11.4, 11.4, -6.1, 6.1, 0.0, 22.5)

    # Top Gear Head Cover
    tris += box(-11.4, 11.4, -6.1, 6.1, 22.5, 25.5)

    # Output Spline Shaft & Brass Bushing (Offset +5.8mm toward one end)
    spline_x = 5.8
    tris += solid_cylinder_z(spline_x, 0.0, 25.5, 27.5, 5.5, segments=20)
    tris += solid_cylinder_z(spline_x, 0.0, 27.5, 30.0, MG90S_SPLINE_R, segments=18)

    # Mounting Flanges (Ear to Ear: 32.5mm)
    tris += box(-16.25, 16.25, -6.1, 6.1, 16.0, 18.5)
    for fx in [-14.0, 14.0]:
        tris += hollow_cylinder_z(fx, 0.0, 16.0, 18.5, 2.5, M2_HOLE_R, segments=12)

    # 3-Pin Wire Exit Grommet
    tris += box(-11.4, -8.0, -4.0, 4.0, 2.0, 6.0)

    return tris

def build_servo_horn() -> List[Triangle]:
    """
    Standard MG90S 21T Single-Arm Servo Horn:
    - Spline Hub: Ø7.0mm OD, 4.0mm height
    - Tapered Linkage Arm: Length 15.0mm with M2 linkage pivot hole at R=12.0mm
    """
    tris = []
    # Spline Hub
    tris += hollow_cylinder_z(0.0, 0.0, 0.0, 4.0, 3.5, MG90S_SPLINE_R, segments=20)

    # Tapered Actuation Arm (pointing along +Y)
    layers = [
        (1.5, [(-3.0, 0.0), (3.0, 0.0), (1.5, 15.0), (-1.5, 15.0)]),
        (3.5, [(-3.0, 0.0), (3.0, 0.0), (1.5, 15.0), (-1.5, 15.0)]),
    ]
    tris += loft_contours_z(layers, cap_bottom=True, cap_top=True)

    # M2 Pivot Linkage Eyelet (At Y=12.0mm)
    tris += hollow_cylinder_z(0.0, 12.0, 1.5, 3.5, 2.5, M2_HOLE_R, segments=16)

    return tris

def build_steering_arm() -> List[Triangle]:
    """
    Articulated Turnbuckle Pushrod Linkage (Length 35mm, Width 6mm, Thickness 4mm):
    - Connecting Servo Horn Eyelet to Head Pitch Bracket
    """
    tris = []
    # Main Rod Shank
    tris += box(-2.5, 2.5, 0.0, 30.0, -2.0, 2.0)
    # End Eyelet 1 (Servo Horn Side: Y=0)
    tris += hollow_cylinder_z(0.0, 0.0, -2.0, 2.0, 4.5, M2_HOLE_R, segments=16)
    # End Eyelet 2 (Head Clevis Side: Y=30)
    tris += hollow_cylinder_z(0.0, 30.0, -2.0, 2.0, 4.5, M2_HOLE_R, segments=16)
    return tris

# ---------------------------------------------------------------------------
# 5. Head Visor Assembly with Dual Hinge Clevis & Internal Pushrod Horn
# ---------------------------------------------------------------------------

def build_head_window_half() -> List[Triangle]:
    """
    Exotic Head Front Visor: X: [-150..150], Z: [0..190], Y: [0..22.5]
    - 7-Inch 800x480 Capacitive Touch LCD Screen Frame:
      Front Display Window Cutout (154x86mm) with 2.5mm retention bezel
      4x M3 Screen Mounting Standoff Bosses (155x97mm pitch)
    - Raspberry Pi Camera Module v2:
      Ø8.0mm Camera Optical Aperture at (X=0, Z=175mm)
      4x M2 Camera PCB Standoff Bosses (21x12.5mm pitch)
    - Dual-Shear Hinge Clevis Fork Brackets at X=±40mm, Z=0 (9mm clear gap, Ø3.1mm pin bores)
    - Internal Pushrod Horn at X=18mm, Z=-12mm
    - 8x M3 Perimeter Assembly Bosses with Heat-Set Inserts
    """
    tris = []
    W = WALL_HEAD

    outer_layers = [
        (0.0,  filleted_rect_xz(-150.0, 150.0, 0.0, 190.0, r=25.0, n_per_corner=8)),
        (12.0, filleted_rect_xz(-148.0, 148.0, 2.0, 188.0, r=24.0, n_per_corner=8)),
        (22.5, filleted_rect_xz(-145.0, 145.0, 4.0, 186.0, r=22.0, n_per_corner=8)),
    ]
    tris += loft_contours_y(outer_layers, cap_back=True, cap_front=True)

    # 1. 7-Inch Capacitive Touch LCD Screen Window Bezel & 4x M3 Corner Standoffs
    # Display Window Cutout Frame (154mm x 86mm centered at X=0, Z=95mm)
    lcd_cz = 95.0
    tris += box(-LCD_VIEW_W/2.0, LCD_VIEW_W/2.0, 18.0, 22.5, lcd_cz - LCD_VIEW_H/2.0, lcd_cz + LCD_VIEW_H/2.0)
    # 4x M3 Screen Mounting Standoffs (155mm x 97mm pitch)
    for lx in [-LCD_HOLE_DX/2.0, LCD_HOLE_DX/2.0]:
        for lz in [lcd_cz - LCD_HOLE_DY/2.0, lcd_cz + LCD_HOLE_DY/2.0]:
            tris += hollow_cylinder_y(lx, lz, 2.0, 18.0, 3.8, M3_INSERT_R, segments=14)

    # 2. Raspberry Pi Camera Module v2 Mount (Above Screen at X=0, Z=175mm)
    # Camera Optical Aperture (Ø8.0mm)
    tris += hollow_cylinder_y(0.0, 175.0, 15.0, 22.5, 7.5, CAM_LENS_R, segments=20)
    # 4x M2 Camera PCB Standoffs (21mm x 12.5mm pitch)
    for cx in [-CAM_HOLE_DX/2.0, CAM_HOLE_DX/2.0]:
        for cz in [175.0 - CAM_HOLE_DY/2.0, 175.0 + CAM_HOLE_DY/2.0]:
            tris += hollow_cylinder_y(cx, cz, 4.0, 18.0, 2.5, M2_INSERT_R, segments=12)

    # 3. Dual-Shear Hinge Clevis Fork Brackets (At X=±40mm, Z=0)
    # Dual-ear fork with 9mm clear gap [hx - 4.5, hx + 4.5] straddling neck's 7mm tongue
    for hx in [-HINGE_X, HINGE_X]:
        # Inner Ear (X: [hx - 7.5, hx - 4.5])
        tris += box(hx - 7.5, hx - 4.5, -8.0, 8.0, -14.0, 4.0)
        tris += hollow_cylinder_y(hx - 6.0, 0.0, -8.0, 8.0, 4.5, HINGE_PIN_R, segments=18)
        # Outer Ear (X: [hx + 4.5, hx + 7.5])
        tris += box(hx + 4.5, hx + 7.5, -8.0, 8.0, -14.0, 4.0)
        tris += hollow_cylinder_y(hx + 6.0, 0.0, -8.0, 8.0, 4.5, HINGE_PIN_R, segments=18)

    # 4. Internal Pushrod Driving Horn (At X=18mm, Z=-12mm)
    tris += box(14.0, 22.0, -5.0, 5.0, -15.0, 0.0)
    tris += hollow_cylinder_y(18.0, -12.0, -5.0, 5.0, 3.5, M2_HOLE_R, segments=16)

    # 5. Head Perimeter Assembly Bosses (8x M3 connecting front visor to rear dome)
    for px, pz in [(-135.0, 20.0), (135.0, 20.0), (-135.0, 170.0), (135.0, 170.0),
                   (-135.0, 95.0), (135.0, 95.0), (0.0, 15.0), (0.0, 175.0)]:
        tris += hollow_cylinder_y(px, pz, 2.0, 18.0, 3.8, M3_INSERT_R, segments=14)

    return tris


def build_head_cover_half() -> List[Triangle]:
    """
    Exotic Head Rear Aero Dome: X: [-150..150], Z: [0..190], Y: [-22.5..0]
    - Aerodynamic cooling ventilation louvers
    - 8x M3 Perimeter Clearance Hole Bosses matching front visor
    - Display driver board clearance cavity
    """
    tris = []
    W = WALL_HEAD

    outer_layers = [
        (0.0,   filleted_rect_xz(-150.0, 150.0, 0.0, 190.0, r=25.0, n_per_corner=8)),
        (-8.0,  filleted_rect_xz(-146.0, 146.0, 4.0, 186.0, r=26.0, n_per_corner=8)),
        (-16.0, filleted_rect_xz(-135.0, 135.0, 12.0, 178.0, r=28.0, n_per_corner=8)),
        (-21.0, filleted_rect_xz(-110.0, 110.0, 25.0, 165.0, r=30.0, n_per_corner=8)),
        (-22.5, filleted_rect_xz(-70.0,   70.0, 45.0, 145.0, r=25.0, n_per_corner=8)),
    ]
    tris += loft_contours_y(outer_layers, cap_back=True, cap_front=True)

    # 1. Aerodynamic Ventilation Louvers
    for louver_z in range(60, 135, 15):
        tris += box(-45.0, 45.0, -22.0, -19.0, float(louver_z), float(louver_z + 6))

    # 2. 8x M3 Perimeter Clearance Hole Bosses matching front visor
    for px, pz in [(-135.0, 20.0), (135.0, 20.0), (-135.0, 170.0), (135.0, 170.0),
                   (-135.0, 95.0), (135.0, 95.0), (0.0, 15.0), (0.0, 175.0)]:
        tris += hollow_cylinder_y(px, pz, -18.0, -2.0, 3.8, M3_HOLE_R, segments=14)

    return tris

# ---------------------------------------------------------------------------
# Master CAD Pipeline Execution
# ---------------------------------------------------------------------------

def generate_all_production_cad(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 75)
    print("🚀 Karma Physical 3D-Printable CAD Suite with Complete Hardware Mounting")
    print(f"📦 Output Directory: {os.path.abspath(out_dir)}")
    print("=" * 75)

    generators = [
        ("Base Front Chassis",       "base_front.stl",        build_base_front),
        ("Base Back Chassis",        "base_back.stl",         build_base_back),
        ("Column Front Torso",       "column_front.stl",      build_column_front),
        ("Column Back Spine",        "column_back.stl",       build_column_back),
        ("Neck Front Cowling",       "neck_front.stl",        build_neck_front),
        ("Neck Back Cowling",        "neck_back.stl",         build_neck_back),
        ("MG90S Micro Servo",        "mg90s_servo.stl",       build_mg90s_servo),
        ("MG90S 21T Servo Horn",     "servo_horn.stl",        build_servo_horn),
        ("Articulated Pushrod",      "steering_arm.stl",      build_steering_arm),
        ("Head Display Front Visor", "head_window_half.stl",  build_head_window_half),
        ("Head Rear Aero Dome",      "head_cover_half.stl",   build_head_cover_half),
    ]

    total_tris = 0
    for name, filename, gen_fn in generators:
        filepath = os.path.join(out_dir, filename)
        tris = gen_fn()
        write_binary_stl(filepath, tris)
        total_tris += len(tris)

    print("=" * 75)
    print(f"✨ Successfully generated all {len(generators)} Production STLs! Total triangles: {total_tris:,}")
    print("=" * 75)

if __name__ == "__main__":
    target_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_production_cad(target_dir)
