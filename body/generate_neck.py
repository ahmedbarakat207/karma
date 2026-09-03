import math
import struct
import os

NECK_W      = 160.0
NECK_H      = 60.0
NECK_DEPTH  = 45.0
WALL        = 3.0

HINGE_Z     = 50.0
HINGE_R     = 3.5
HINGE_BOSS  = 7.0
HINGE_X     = 40.0
HINGE_SEGS  = 24

ROM_START_DEG = 90.0
ROM_END_DEG   = 135.0
ARC_R         = 20.0

def _normal(v1, v2, v3):
    ax, ay, az = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]
    bx, by, bz = v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]
    nx = ay*bz - az*by
    ny = az*bx - ax*bz
    nz = ax*by - ay*bx
    length = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
    return (nx/length, ny/length, nz/length)

def write_stl(path, triangles):
    with open(path, 'wb') as f:
        f.write(b'\x00' * 80)
        f.write(struct.pack('<I', len(triangles)))
        for v1, v2, v3 in triangles:
            n = _normal(v1, v2, v3)
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<3f', *v3))
            f.write(struct.pack('<H', 0))
    print(f"  Wrote {len(triangles):6d} triangles -> {os.path.basename(path)}")

def quad(v1, v2, v3, v4):
    return [(v1, v2, v3), (v1, v3, v4)]

def boss_with_hole(cx, cz, y0, y1, boss_r, hole_r, n):
    tris = []
    pts_outer_bot, pts_outer_top = [], []
    pts_inner_bot, pts_inner_top = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        cos_a, sin_a = math.cos(a), math.sin(a)
        pts_outer_bot.append((cx + boss_r*cos_a, y0, cz + boss_r*sin_a))
        pts_outer_top.append((cx + boss_r*cos_a, y1, cz + boss_r*sin_a))
        pts_inner_bot.append((cx + hole_r*cos_a, y0, cz + hole_r*sin_a))
        pts_inner_top.append((cx + hole_r*cos_a, y1, cz + hole_r*sin_a))

    for i in range(n):
        j = (i+1)%n
        tris += quad(pts_outer_bot[i], pts_outer_bot[j],
                     pts_outer_top[j], pts_outer_top[i])
    for i in range(n):
        j = (i+1)%n
        tris += quad(pts_inner_bot[j], pts_inner_bot[i],
                     pts_inner_top[i], pts_inner_top[j])
    for i in range(n):
        j = (i+1)%n
        tris.append((pts_outer_bot[j], pts_outer_bot[i], pts_inner_bot[i]))
        tris.append((pts_outer_bot[j], pts_inner_bot[i], pts_inner_bot[j]))
    for i in range(n):
        j = (i+1)%n
        tris.append((pts_outer_top[i], pts_outer_top[j], pts_inner_top[j]))
        tris.append((pts_outer_top[i], pts_inner_top[j], pts_inner_top[i]))
    return tris

def build_neck_front():
    tris = []
    X0, X1 = -NECK_W/2, NECK_W/2
    Y0, Y1 = 0.0, NECK_DEPTH
    Z0, Z1 = 0.0, NECK_H
    W = WALL

    bx_l = -HINGE_X
    bx_r =  HINGE_X
    bz   = HINGE_Z
    boss_gap = HINGE_BOSS + 1.5

    tris += quad((X0,Y1,Z0),(X1,Y1,Z0),(X1,Y1,Z1),(X0,Y1,Z1))
    tris += quad((X0,Y0,Z0),(X0,Y1,Z0),(X0,Y1,Z1),(X0,Y0,Z1))
    tris += quad((X1,Y1,Z0),(X1,Y0,Z0),(X1,Y0,Z1),(X1,Y1,Z1))

    tris += quad((X0,Y0,Z0),(X1,Y0,Z0),(X1,Y1,Z0),(X0,Y1,Z0))

    tris += quad((X0,Y0,Z0),(bx_l-boss_gap,Y0,Z0),
                 (bx_l-boss_gap,Y0,Z1),(X0,Y0,Z1))
    tris += quad((bx_l+boss_gap,Y0,Z0),(bx_r-boss_gap,Y0,Z0),
                 (bx_r-boss_gap,Y0,Z1),(bx_l+boss_gap,Y0,Z1))
    tris += quad((bx_r+boss_gap,Y0,Z0),(X1,Y0,Z0),
                 (X1,Y0,Z1),(bx_r+boss_gap,Y0,Z1))

    for bx in [bx_l, bx_r]:
        tris += boss_with_hole(bx, bz, Y0, Y1, HINGE_BOSS, HINGE_R, HINGE_SEGS)

    ARC_SEGS  = 24
    r_in  = HINGE_BOSS + 0.5
    r_out = ARC_R

    def ay(r, a): return r * math.sin(a)
    def az(r, a): return bz + r * math.cos(a)

    tris += quad((X0,Y0,Z1),(bx_l-boss_gap,Y0,Z1),
                 (bx_l-boss_gap,Y1,Z1),(X0,Y1,Z1))
    tris += quad((bx_l+boss_gap,Y0,Z1),(bx_r-boss_gap,Y0,Z1),
                 (bx_r-boss_gap,Y1,Z1),(bx_l+boss_gap,Y1,Z1))
    tris += quad((bx_r+boss_gap,Y0,Z1),(X1,Y0,Z1),
                 (X1,Y1,Z1),(bx_r+boss_gap,Y1,Z1))

    for bx in [bx_l, bx_r]:

        tris += quad((bx-boss_gap,Y0,Z1),(bx+boss_gap,Y0,Z1),
                     (bx+boss_gap,Y1,Z1),(bx-boss_gap,Y1,Z1))

        a_pts_in  = []
        a_pts_out = []
        for i in range(ARC_SEGS + 1):
            a = math.radians(ROM_START_DEG +
                             (ROM_END_DEG - ROM_START_DEG) * i / ARC_SEGS)
            a_pts_in .append((bx, ay(r_in,  a), az(r_in,  a)))
            a_pts_out.append((bx, ay(r_out, a), az(r_out, a)))

        a_top_in  = [(p[0], p[1], Z1) for p in a_pts_in]
        a_top_out = [(p[0], p[1], Z1) for p in a_pts_out]

        for i in range(ARC_SEGS):
            tris += quad(a_pts_out[i], a_pts_out[i+1],
                         a_top_out[i+1], a_top_out[i])
            tris += quad(a_pts_in[i+1], a_pts_in[i],
                         a_top_in[i], a_top_in[i+1])
            tris.append((a_pts_in[i],  a_pts_out[i],  a_pts_out[i+1]))
            tris.append((a_pts_in[i],  a_pts_out[i+1], a_pts_in[i+1]))

        tris += quad(a_pts_in[0],  a_pts_out[0],
                     a_top_out[0], a_top_in[0])
        tris += quad(a_pts_out[-1], a_pts_in[-1],
                     a_top_in[-1], a_top_out[-1])

    return tris

def build_neck_back():
    tris = []
    X0, X1 = -NECK_W/2, NECK_W/2
    Y0, Y1 = -NECK_DEPTH, 0.0
    Z0, Z1 = 0.0, NECK_H
    W = WALL

    bx_l = -HINGE_X
    bx_r =  HINGE_X
    bz   = HINGE_Z
    boss_gap = HINGE_BOSS + 1.5

    tris += quad((X1,Y0,Z0),(X0,Y0,Z0),(X0,Y0,Z1),(X1,Y0,Z1))
    tris += quad((X0,Y0,Z0),(X0,Y1,Z0),(X0,Y1,Z1),(X0,Y0,Z1))
    tris += quad((X1,Y1,Z0),(X1,Y0,Z0),(X1,Y0,Z1),(X1,Y1,Z1))

    tris += quad((X1,Y0,Z0),(X0,Y0,Z0),(X0,Y1,Z0),(X1,Y1,Z0))

    tris += quad((X1,Y0,Z1),(X0,Y0,Z1),(X0,Y0+W,Z1),(X1,Y0+W,Z1))
    tris += quad((X0,Y1,Z1),(X0,Y0,Z1),(X0+W,Y0,Z1),(X0+W,Y1,Z1))
    tris += quad((X1,Y1,Z1),(X1-W,Y1,Z1),(X1-W,Y0,Z1),(X1,Y0,Z1))

    tris += quad((X0,Y1,Z0),(bx_l-boss_gap,Y1,Z0),
                 (bx_l-boss_gap,Y1,Z1),(X0,Y1,Z1))
    tris += quad((bx_l+boss_gap,Y1,Z0),(bx_r-boss_gap,Y1,Z0),
                 (bx_r-boss_gap,Y1,Z1),(bx_l+boss_gap,Y1,Z1))
    tris += quad((bx_r+boss_gap,Y1,Z0),(X1,Y1,Z0),
                 (X1,Y1,Z1),(bx_r+boss_gap,Y1,Z1))

    for bx in [bx_l, bx_r]:
        tris += boss_with_hole(bx, bz, Y0, Y1, HINGE_BOSS, HINGE_R, HINGE_SEGS)

    return tris

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print("Generating neck STL files...")
    print(f"  Hinge pin axis : Y at X=+-{HINGE_X}, Z={HINGE_Z}")
    print(f"  ROM            : {ROM_START_DEG} deg (forward) -> {ROM_END_DEG} deg (down-fwd)")
    print(f"  Boss OD: {HINGE_BOSS*2} mm  hole: {HINGE_R*2} mm  arc-R: {ARC_R} mm")

    front_tris = build_neck_front()
    write_stl(os.path.join(out_dir, 'neck_front.stl'), front_tris)

    back_tris = build_neck_back()
    write_stl(os.path.join(out_dir, 'neck_back.stl'), back_tris)

    print("Done. Generated neck_front.stl and neck_back.stl.")
