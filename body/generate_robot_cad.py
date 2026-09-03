import math
import struct
import os
from typing import List, Tuple, Sequence

WALL_BASE     = 3.5
WALL_COLUMN   = 3.5
WALL_NECK     = 3.0
WALL_HEAD     = 3.0

M2_HOLE_R     = 1.1
M2_INSERT_R   = 1.6
M2_5_HOLE_R   = 1.4
M2_5_INSERT_R = 1.9
M3_HOLE_R     = 1.7
M3_INSERT_R   = 2.1
M4_HOLE_R     = 2.25
M4_INSERT_R   = 2.9
M3_BOSS_OD    = 7.5
M4_BOSS_OD    = 10.0

MOTOR_5840_L  = 58.0
MOTOR_5840_W  = 40.0
MOTOR_5840_H  = 35.0
MOTOR_SHAFT_R = 4.0
MOTOR_HOLE_DX = 30.0
MOTOR_HOLE_DY = 40.0

RPI4_PCB_L    = 85.0
RPI4_PCB_W    = 56.0
RPI4_HOLE_DX  = 58.0
RPI4_HOLE_DY  = 49.0

LCD_7INCH_W   = 165.0
LCD_7INCH_H   = 107.0
LCD_VIEW_W    = 154.0
LCD_VIEW_H    = 86.0
LCD_HOLE_DX   = 155.0
LCD_HOLE_DY   = 97.0

CAM_PCB_L     = 25.0
CAM_PCB_W     = 24.0
CAM_HOLE_DX   = 21.0
CAM_HOLE_DY   = 12.5
CAM_LENS_R    = 4.0

BATT_L        = 151.0
BATT_W        = 65.0
BATT_H        = 95.0

BTS_HOLE_DX   = 44.0
BTS_HOLE_DY   = 44.0

MG90S_L       = 23.0
MG90S_W       = 12.5
MG90S_H       = 23.0
MG90S_FLANGE_L= 32.5
MG90S_HOLE_DIST=28.0
MG90S_SPLINE_R= 2.4
MG90S_SPLINE_H= 4.5

HINGE_X       = 40.0
HINGE_Z       = 50.0
HINGE_PIN_R   = 1.55
HINGE_BOSS_R  = 5.5

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

def build_base_front() -> List[Triangle]:
    tris = []
    W = WALL_BASE

    outer_layers = [
        (0.0,   half_contour_front(-160.0, 160.0, 0.0, 210.0, chamfer_front=45.0, n_per_seg=6)),
        (40.0,  half_contour_front(-155.0, 155.0, 0.0, 195.0, chamfer_front=42.0, n_per_seg=6)),
        (100.0, half_contour_front(-130.0, 130.0, 0.0, 150.0, chamfer_front=32.0, n_per_seg=6)),
        (160.0, half_contour_front(-95.0,   95.0, 0.0,  95.0, chamfer_front=24.0, n_per_seg=6)),
        (220.0, half_contour_front(-65.0,   65.0, 0.0,  52.0, chamfer_front=16.0, n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    for mx in [-105.0, 105.0]:
        for hx in [mx - MOTOR_HOLE_DX/2.0, mx + MOTOR_HOLE_DX/2.0]:
            for hy in [15.0, 15.0 + MOTOR_HOLE_DY]:
                tris += mounting_boss_m3(hx, hy, W, W + 12.0, is_insert=True)
        tunnel_x0 = 135.0 if mx > 0 else -160.0
        tunnel_x1 = 160.0 if mx > 0 else -135.0
        tris += hollow_cylinder_x(35.0, 45.0, tunnel_x0, tunnel_x1, 12.0, 9.0, segments=20)

    for cx in [-12.0, 12.0]:
        for cy in [120.0, 150.0]:
            tris += mounting_boss_m3(cx, cy, 0.0, W + 8.0, is_insert=True)

    tris += box(-BATT_L/2.0 - 2.0, BATT_L/2.0 + 2.0, 0.0, BATT_W/2.0 + 2.0, W, W + 25.0)

    for bx, by in [(-45.0, 22.0), (45.0, 22.0)]:
        tris += mounting_boss_m4(bx, by, 190.0, 220.0 - W, is_insert=True)

    for sx in [-130.0, -50.0, 50.0, 130.0]:
        tris += mounting_boss_m4(sx, 10.0, W, 220.0 - W, is_insert=True)

    tris += hollow_cylinder_z(0.0, 12.0, 190.0, 220.0, 13.0, 11.0, segments=20)

    return tris

def build_base_back() -> List[Triangle]:
    tris = []
    W = WALL_BASE

    outer_layers = [
        (0.0,   half_contour_back(-160.0, 160.0, -190.0, 0.0, chamfer_back=45.0, n_per_seg=6)),
        (40.0,  half_contour_back(-155.0, 155.0, -175.0, 0.0, chamfer_back=42.0, n_per_seg=6)),
        (100.0, half_contour_back(-130.0, 130.0, -140.0, 0.0, chamfer_back=32.0, n_per_seg=6)),
        (160.0, half_contour_back(-95.0,   95.0,  -95.0, 0.0, chamfer_back=24.0, n_per_seg=6)),
        (220.0, half_contour_back(-65.0,   65.0,  -58.0, 0.0, chamfer_back=16.0, n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    for dx in [-60.0, 60.0]:
        for hx in [dx - BTS_HOLE_DX/2.0, dx + BTS_HOLE_DX/2.0]:
            for hy in [-85.0, -85.0 + BTS_HOLE_DY]:
                tris += mounting_boss_m3(hx, hy, W, W + 10.0, is_insert=True)

    for bx in [-16.0, 16.0]:
        tris += mounting_boss_m3(bx, -130.0, W, W + 8.0, is_insert=True)

    for cx in [-12.0, 12.0]:
        for cy in [-150.0, -120.0]:
            tris += mounting_boss_m3(cx, cy, 0.0, W + 8.0, is_insert=True)

    tris += box(-BATT_L/2.0 - 2.0, BATT_L/2.0 + 2.0, -BATT_W/2.0 - 2.0, 0.0, W, W + 25.0)

    tris += hollow_cylinder_y(-45.0, 50.0, -190.0, -180.0, 8.0, 5.8, segments=20)

    tris += box(35.0, 55.0, -190.0, -180.0, 43.5, 56.5)

    for bx, by in [(-45.0, -22.0), (45.0, -22.0)]:
        tris += mounting_boss_m4(bx, by, 190.0, 220.0 - W, is_insert=True)

    for sx in [-130.0, -50.0, 50.0, 130.0]:
        tris += mounting_boss_m4(sx, -10.0, W, 220.0 - W, is_insert=False)

    return tris

def build_column_front() -> List[Triangle]:
    tris = []
    W = WALL_COLUMN

    outer_layers = [
        (0.0,   half_contour_front(-65.0, 65.0, 0.0, 52.0, chamfer_front=16.0, n_per_seg=6)),
        (100.0, half_contour_front(-56.0, 56.0, 0.0, 44.0, chamfer_front=13.0, n_per_seg=6)),
        (220.0, half_contour_front(-48.0, 48.0, 0.0, 36.0, chamfer_front=10.0, n_per_seg=6)),
        (340.0, half_contour_front(-54.0, 54.0, 0.0, 40.0, chamfer_front=12.0, n_per_seg=6)),
        (420.0, half_contour_front(-60.0, 60.0, 0.0, 44.0, chamfer_front=14.0, n_per_seg=6)),
        (480.0, half_contour_front(-65.0, 65.0, 0.0, 42.0, chamfer_front=14.0, n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    rpi_cz = 200.0
    for rx in [-RPI4_HOLE_DY/2.0, RPI4_HOLE_DY/2.0]:
        for rz in [rpi_cz - RPI4_HOLE_DX/2.0, rpi_cz + RPI4_HOLE_DX/2.0]:
            tris += hollow_cylinder_y(rx, rz, 8.0, 22.0, 3.5, M2_5_INSERT_R, segments=12)

    tris += box(-50.0, -40.0, 10.0, 38.0, rpi_cz - 16.0, rpi_cz + 16.0)
    tris += box(-50.0, -40.0, 12.0, 28.0, rpi_cz + 18.0, rpi_cz + 32.0)
    tris += box(-50.0, -40.0, 8.0, 18.0,  rpi_cz - 32.0, rpi_cz - 24.0)

    for spk_x in [-36.0, 36.0]:
        tris += hollow_cylinder_y(spk_x, 320.0, 34.0, 42.0, 18.0, 15.0, segments=20)

    for bx in [-45.0, 45.0]:
        tris += mounting_boss_m4(bx, 22.0, W, W + 22.0, is_insert=False)

    for bx in [-45.0, 45.0]:
        tris += mounting_boss_m4(bx, 18.0, 480.0 - 22.0, 480.0 - W, is_insert=True)

    tris += hollow_cylinder_z(0.0, 8.0, W, 480.0 - W, 11.0, 9.0, segments=20)

    return tris

def build_column_back() -> List[Triangle]:
    tris = []
    W = WALL_COLUMN

    outer_layers = [
        (0.0,   half_contour_back(-65.0, 65.0, -58.0, 0.0, chamfer_back=16.0, n_per_seg=6)),
        (100.0, half_contour_back(-56.0, 56.0, -48.0, 0.0, chamfer_back=13.0, n_per_seg=6)),
        (220.0, half_contour_back(-48.0, 48.0, -40.0, 0.0, chamfer_back=10.0, n_per_seg=6)),
        (340.0, half_contour_back(-54.0, 54.0, -44.0, 0.0, chamfer_back=12.0, n_per_seg=6)),
        (420.0, half_contour_back(-60.0, 60.0, -44.0, 0.0, chamfer_back=14.0, n_per_seg=6)),
        (480.0, half_contour_back(-65.0, 65.0, -42.0, 0.0, chamfer_back=14.0, n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    for louver_z in range(120, 420, 30):
        tris += box(-30.0, 30.0, -44.0, -38.0, float(louver_z), float(louver_z + 10))

    for bx in [-45.0, 45.0]:
        tris += mounting_boss_m4(bx, -22.0, W, W + 22.0, is_insert=False)

    for bx in [-45.0, 45.0]:
        tris += mounting_boss_m4(bx, -18.0, 480.0 - 22.0, 480.0 - W, is_insert=True)

    for seam_z in [80.0, 180.0, 280.0, 380.0]:
        tris += mounting_boss_m4(-45.0, -8.0, seam_z - 10.0, seam_z + 10.0, is_insert=False)
        tris += mounting_boss_m4( 45.0, -8.0, seam_z - 10.0, seam_z + 10.0, is_insert=False)

    return tris

def build_neck_front() -> List[Triangle]:
    tris = []
    W = WALL_NECK
    SHELL_TOP = 46.0

    outer_layers = [
        (0.0,       half_contour_front(-65.0, 65.0, 0.0, 42.0, chamfer_front=14.0, n_per_seg=6)),
        (14.0,      half_contour_front(-65.0, 65.0, 0.0, 42.0, chamfer_front=14.0, n_per_seg=6)),
        (28.0,      half_contour_front(-65.0, 65.0, 0.0, 32.0, chamfer_front=12.0, n_per_seg=6)),
        (38.0,      half_contour_front(-65.0, 65.0, 0.0, 18.0, chamfer_front=7.0,  n_per_seg=6)),
        (SHELL_TOP, half_contour_front(-65.0, 65.0, 0.0, 8.0,  chamfer_front=4.0,  n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    for bx in [-HINGE_X, HINGE_X]:
        tris += hollow_cylinder_y(bx, HINGE_Z, 0.0, 7.0, 4.5, HINGE_PIN_R, segments=20)
        tris += box(bx - 3.5, bx + 3.5, 0.0, 7.0, 15.0, HINGE_Z)

    servo_cx, servo_cy, servo_cz = 18.0, 15.0, 22.0
    tris += box(servo_cx - MG90S_L/2.0 - 2.0, servo_cx + MG90S_L/2.0 + 2.0,
                servo_cy - MG90S_W/2.0 - 2.0, servo_cy + MG90S_W/2.0 + 2.0,
                W, servo_cz + 8.0)
    for m2_x in [servo_cx - MG90S_HOLE_DIST/2.0, servo_cx + MG90S_HOLE_DIST/2.0]:
        tris += hollow_cylinder_z(m2_x, servo_cy, W, servo_cz + 8.0, 3.2, M2_INSERT_R, segments=12)

    for fx in [-45.0, 45.0]:
        tris += mounting_boss_m4(fx, 18.0, W, W + 15.0, is_insert=False)

    return tris

def build_neck_back() -> List[Triangle]:
    tris = []
    W = WALL_NECK
    SHELL_TOP = 46.0

    outer_layers = [
        (0.0,       half_contour_back(-65.0, 65.0, -42.0, 0.0, chamfer_back=14.0, n_per_seg=6)),
        (14.0,      half_contour_back(-65.0, 65.0, -42.0, 0.0, chamfer_back=14.0, n_per_seg=6)),
        (28.0,      half_contour_back(-65.0, 65.0, -30.0, 0.0, chamfer_back=11.0, n_per_seg=6)),
        (38.0,      half_contour_back(-65.0, 65.0, -16.0, 0.0, chamfer_back=6.0,  n_per_seg=6)),
        (SHELL_TOP, half_contour_back(-65.0, 65.0, -8.0,  0.0, chamfer_back=3.0,  n_per_seg=6)),
    ]
    tris += loft_contours_z(outer_layers, cap_bottom=True, cap_top=True)

    for bx in [-HINGE_X, HINGE_X]:
        tris += hollow_cylinder_y(bx, HINGE_Z, -7.0, 0.0, 4.5, HINGE_PIN_R, segments=20)
        tris += box(bx - 3.5, bx + 3.5, -7.0, 0.0, 15.0, HINGE_Z)

    for bx in [-45.0, 45.0]:
        tris += mounting_boss_m4(bx, -18.0, W, W + 15.0, is_insert=False)

    tris += hollow_cylinder_z(0.0, -12.0, W, SHELL_TOP - W, 10.0, 8.0, segments=16)

    return tris

def build_mg90s_servo() -> List[Triangle]:
    tris = []
    tris += box(-11.4, 11.4, -6.1, 6.1, 0.0, 22.5)

    tris += box(-11.4, 11.4, -6.1, 6.1, 22.5, 25.5)

    spline_x = 5.8
    tris += solid_cylinder_z(spline_x, 0.0, 25.5, 27.5, 5.5, segments=20)
    tris += solid_cylinder_z(spline_x, 0.0, 27.5, 30.0, MG90S_SPLINE_R, segments=18)

    tris += box(-16.25, 16.25, -6.1, 6.1, 16.0, 18.5)
    for fx in [-14.0, 14.0]:
        tris += hollow_cylinder_z(fx, 0.0, 16.0, 18.5, 2.5, M2_HOLE_R, segments=12)

    tris += box(-11.4, -8.0, -4.0, 4.0, 2.0, 6.0)

    return tris

def build_servo_horn() -> List[Triangle]:
    tris = []
    tris += hollow_cylinder_z(0.0, 0.0, 0.0, 4.0, 3.5, MG90S_SPLINE_R, segments=20)

    layers = [
        (1.5, [(-3.0, 0.0), (3.0, 0.0), (1.5, 15.0), (-1.5, 15.0)]),
        (3.5, [(-3.0, 0.0), (3.0, 0.0), (1.5, 15.0), (-1.5, 15.0)]),
    ]
    tris += loft_contours_z(layers, cap_bottom=True, cap_top=True)

    tris += hollow_cylinder_z(0.0, 12.0, 1.5, 3.5, 2.5, M2_HOLE_R, segments=16)

    return tris

def build_steering_arm() -> List[Triangle]:
    tris = []
    tris += box(-2.5, 2.5, 0.0, 30.0, -2.0, 2.0)
    tris += hollow_cylinder_z(0.0, 0.0, -2.0, 2.0, 4.5, M2_HOLE_R, segments=16)
    tris += hollow_cylinder_z(0.0, 30.0, -2.0, 2.0, 4.5, M2_HOLE_R, segments=16)
    return tris

def build_head_window_half() -> List[Triangle]:
    tris = []
    W = WALL_HEAD

    outer_layers = [
        (0.0,  filleted_rect_xz(-150.0, 150.0, 0.0, 190.0, r=25.0, n_per_corner=8)),
        (12.0, filleted_rect_xz(-148.0, 148.0, 2.0, 188.0, r=24.0, n_per_corner=8)),
        (22.5, filleted_rect_xz(-145.0, 145.0, 4.0, 186.0, r=22.0, n_per_corner=8)),
    ]
    tris += loft_contours_y(outer_layers, cap_back=True, cap_front=True)

    lcd_cz = 95.0
    tris += box(-LCD_VIEW_W/2.0, LCD_VIEW_W/2.0, 18.0, 22.5, lcd_cz - LCD_VIEW_H/2.0, lcd_cz + LCD_VIEW_H/2.0)
    for lx in [-LCD_HOLE_DX/2.0, LCD_HOLE_DX/2.0]:
        for lz in [lcd_cz - LCD_HOLE_DY/2.0, lcd_cz + LCD_HOLE_DY/2.0]:
            tris += hollow_cylinder_y(lx, lz, 2.0, 18.0, 3.8, M3_INSERT_R, segments=14)

    tris += hollow_cylinder_y(0.0, 175.0, 15.0, 22.5, 7.5, CAM_LENS_R, segments=20)
    for cx in [-CAM_HOLE_DX/2.0, CAM_HOLE_DX/2.0]:
        for cz in [175.0 - CAM_HOLE_DY/2.0, 175.0 + CAM_HOLE_DY/2.0]:
            tris += hollow_cylinder_y(cx, cz, 4.0, 18.0, 2.5, M2_INSERT_R, segments=12)

    for hx in [-HINGE_X, HINGE_X]:
        tris += box(hx - 7.5, hx - 4.5, -8.0, 8.0, -14.0, 4.0)
        tris += hollow_cylinder_y(hx - 6.0, 0.0, -8.0, 8.0, 4.5, HINGE_PIN_R, segments=18)
        tris += box(hx + 4.5, hx + 7.5, -8.0, 8.0, -14.0, 4.0)
        tris += hollow_cylinder_y(hx + 6.0, 0.0, -8.0, 8.0, 4.5, HINGE_PIN_R, segments=18)

    tris += box(14.0, 22.0, -5.0, 5.0, -15.0, 0.0)
    tris += hollow_cylinder_y(18.0, -12.0, -5.0, 5.0, 3.5, M2_HOLE_R, segments=16)

    for px, pz in [(-135.0, 20.0), (135.0, 20.0), (-135.0, 170.0), (135.0, 170.0),
                   (-135.0, 95.0), (135.0, 95.0), (0.0, 15.0), (0.0, 175.0)]:
        tris += hollow_cylinder_y(px, pz, 2.0, 18.0, 3.8, M3_INSERT_R, segments=14)

    return tris

def build_head_cover_half() -> List[Triangle]:
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

    for louver_z in range(60, 135, 15):
        tris += box(-45.0, 45.0, -22.0, -19.0, float(louver_z), float(louver_z + 6))

    for px, pz in [(-135.0, 20.0), (135.0, 20.0), (-135.0, 170.0), (135.0, 170.0),
                   (-135.0, 95.0), (135.0, 95.0), (0.0, 15.0), (0.0, 175.0)]:
        tris += hollow_cylinder_y(px, pz, -18.0, -2.0, 3.8, M3_HOLE_R, segments=14)

    return tris

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
