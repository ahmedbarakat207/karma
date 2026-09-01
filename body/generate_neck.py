"""
generate_neck.py
================
Generates neck_front.stl and neck_back.stl for the robot body assembly.

Coordinate system (matches existing assembly):
  X : left (-) to right (+),  width ~160 mm total
  Y : front (positive) to back (negative)
  Z : bottom (0) to top (60 mm)

Neck geometry:
  - Overall box: X=-80..80, Z=0..60, Y front/back halves split at Y=0
  - Hinge pin axis : along Y at X=+/-HINGE_X, Z=HINGE_Z (near top of shell)
  - ROM: 90 deg = head looking FORWARD (horizontal, +Y direction)
         135 deg = head tilted 45 deg downward past horizontal
  - Arc slot is cut through the TOP face (Z=NECK_H) of each half so the
    head-mounting bracket can swing freely in the YZ plane.
  - Pin boss cylinders run the full Y depth of each half (front: Y=0..45,
    back: Y=-45..0) with an axial through-hole for the hinge pin.

Angle convention (YZ plane, measured from +Z toward +Y):
  90  deg  -> bracket points in +Y (head looks forward, horizontal)
  135 deg  -> bracket points 45 deg past +Y toward -Z (head tilts down-fwd)

Units: millimetres
"""

import math
import struct
import os

# ---------------------------------------------------------------------------
# Geometry constants  (match existing assembly measurements)
# ---------------------------------------------------------------------------
NECK_W      = 160.0        # total width (X: -80 .. +80)
NECK_H      = 60.0         # total height (Z: 0 .. 60)
NECK_DEPTH  = 45.0         # half-depth each side (Y: 0..45 front, -45..0 back)
WALL        = 3.0          # shell wall thickness

# Hinge moved to NEAR TOP of shell so the forward-looking head has room to swing
HINGE_Z     = 50.0         # Z centre of hinge pin axis (near top)
HINGE_R     = 3.5          # hinge pin radius (hole)
HINGE_BOSS  = 7.0          # outer radius of boss cylinder
HINGE_X     = 40.0         # X position of hinge pin (mirrored +/-)
HINGE_SEGS  = 24           # circle facets

# ROM in the YZ plane (angle measured from +Z axis toward +Y axis):
#   90 deg  = head pointing forward (+Y), i.e. looking straight ahead
#   135 deg = head tilted 45 deg past horizontal (looking down-forward)
ROM_START_DEG = 90.0       # start of arc cutout (head looking forward)
ROM_END_DEG   = 135.0      # end of arc cutout  (head tilted 45 deg down)
ARC_R         = 20.0       # arc clearance radius from hinge centre (mm)

# ---------------------------------------------------------------------------
# Low-level STL helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def quad(v1, v2, v3, v4):
    return [(v1, v2, v3), (v1, v3, v4)]


def boss_with_hole(cx, cz, y0, y1, boss_r, hole_r, n):
    """Hollow cylinder (boss with pin hole) along Y-axis at (cx, ?, cz)."""
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

    # Outer side
    for i in range(n):
        j = (i+1)%n
        tris += quad(pts_outer_bot[i], pts_outer_bot[j],
                     pts_outer_top[j], pts_outer_top[i])
    # Inner side (reversed)
    for i in range(n):
        j = (i+1)%n
        tris += quad(pts_inner_bot[j], pts_inner_bot[i],
                     pts_inner_top[i], pts_inner_top[j])
    # Annular bottom cap
    for i in range(n):
        j = (i+1)%n
        tris.append((pts_outer_bot[j], pts_outer_bot[i], pts_inner_bot[i]))
        tris.append((pts_outer_bot[j], pts_inner_bot[i], pts_inner_bot[j]))
    # Annular top cap (reversed)
    for i in range(n):
        j = (i+1)%n
        tris.append((pts_outer_top[i], pts_outer_top[j], pts_inner_top[j]))
        tris.append((pts_outer_top[i], pts_inner_top[j], pts_inner_top[i]))
    return tris


# ---------------------------------------------------------------------------
# Front half
# ---------------------------------------------------------------------------

def build_neck_front():
    """
    Front neck half  Y: 0 → +45 mm

    Features
    --------
    • Outer shell walls (front +Y, left -X, right +X)
    • Solid bottom plate (Z=0)
    • Top plate WITH arc slot cut at each boss location, allowing the
      head bracket to swing from 90 deg (forward) to 135 deg (down-fwd)
    • Mating back wall (Y=0) in three panel sections around the bosses
    • Hinge boss cylinders at X=+/-HINGE_X, Z=HINGE_Z, spanning Y=0..45
    """
    tris = []
    X0, X1 = -NECK_W/2, NECK_W/2
    Y0, Y1 = 0.0, NECK_DEPTH
    Z0, Z1 = 0.0, NECK_H
    W = WALL

    bx_l = -HINGE_X
    bx_r =  HINGE_X
    bz   = HINGE_Z
    boss_gap = HINGE_BOSS + 1.5   # half-width of gap in back wall around each boss

    # ---- Outer shell walls ----
    # Front face (+Y)
    tris += quad((X0,Y1,Z0),(X1,Y1,Z0),(X1,Y1,Z1),(X0,Y1,Z1))
    # Left face (-X)
    tris += quad((X0,Y0,Z0),(X0,Y1,Z0),(X0,Y1,Z1),(X0,Y0,Z1))
    # Right face (+X)
    tris += quad((X1,Y1,Z0),(X1,Y0,Z0),(X1,Y0,Z1),(X1,Y1,Z1))

    # ---- Bottom plate ----
    tris += quad((X0,Y0,Z0),(X1,Y0,Z0),(X1,Y1,Z0),(X0,Y1,Z0))

    # ---- Mating back wall (Y=0), panels left of left boss, centre, right of right boss ----
    tris += quad((X0,Y0,Z0),(bx_l-boss_gap,Y0,Z0),
                 (bx_l-boss_gap,Y0,Z1),(X0,Y0,Z1))
    tris += quad((bx_l+boss_gap,Y0,Z0),(bx_r-boss_gap,Y0,Z0),
                 (bx_r-boss_gap,Y0,Z1),(bx_l+boss_gap,Y0,Z1))
    tris += quad((bx_r+boss_gap,Y0,Z0),(X1,Y0,Z0),
                 (X1,Y0,Z1),(bx_r+boss_gap,Y0,Z1))

    # ---- Hinge boss cylinders ----
    for bx in [bx_l, bx_r]:
        tris += boss_with_hole(bx, bz, Y0, Y1, HINGE_BOSS, HINGE_R, HINGE_SEGS)

    # ---- Top plate with arc clearance slots ----
    # The arc is in the YZ plane at each boss X location.
    # Angle convention: a=90 deg -> +Y (forward), a=135 deg -> down-fwd.
    # Arc bounds at HINGE_Z=50, ARC_R=20:
    #   90  deg: Y=20, Z=50  (inside shell)
    #   135 deg: Y=14.1, Z=35.9 (inside shell)
    # So the bracket sweeps through the TOP face (Z=NECK_H=60).
    # We build the top plate as two filled strips (around the slot) at each boss.

    ARC_SEGS  = 24
    r_in  = HINGE_BOSS + 0.5
    r_out = ARC_R

    # Helper: YZ arc position
    def ay(r, a): return r * math.sin(a)           # Y component
    def az(r, a): return bz + r * math.cos(a)      # Z component (from hinge centre)

    # Compute the Y extents of the arc slot footprint in the top face (Z=Z1)
    # The bracket passes through the top face; the slot is a radial opening
    # bounded by r_in and r_out at each angle step.
    # We build the top plate as solid except for the wedge-shaped slots.

    # Solid top plate sections:
    # Left solid strip: X0 to (bx_l - boss_gap), full Y0..Y1
    tris += quad((X0,Y0,Z1),(bx_l-boss_gap,Y0,Z1),
                 (bx_l-boss_gap,Y1,Z1),(X0,Y1,Z1))
    # Middle solid strip: (bx_l+boss_gap) to (bx_r-boss_gap)
    tris += quad((bx_l+boss_gap,Y0,Z1),(bx_r-boss_gap,Y0,Z1),
                 (bx_r-boss_gap,Y1,Z1),(bx_l+boss_gap,Y1,Z1))
    # Right solid strip: (bx_r+boss_gap) to X1
    tris += quad((bx_r+boss_gap,Y0,Z1),(X1,Y0,Z1),
                 (X1,Y1,Z1),(bx_r+boss_gap,Y1,Z1))

    # Around each boss: partial top plate with the arc slot cut out
    # Build left and right solid tabs, then the arc slot walls.
    for bx in [bx_l, bx_r]:
        # Top plate patch from (bx-boss_gap)..(bx+boss_gap), Y=Y0..Y1
        # minus the arc sector r_in..r_out, angle 90..135 deg (in YZ)
        # We approximate by building the solid background and then the slot walls.

        # Solid top patch around boss (full rectangle, slot walls cap it below)
        tris += quad((bx-boss_gap,Y0,Z1),(bx+boss_gap,Y0,Z1),
                     (bx+boss_gap,Y1,Z1),(bx-boss_gap,Y1,Z1))

        # Arc slot WALLS (the slot is open through Z=Z1 top face)
        # Build: outer arc wall, inner arc wall, two radial end-caps,
        # and bottom faces of the slot (at Z_bottom = az(r, a)).
        a_pts_in  = []
        a_pts_out = []
        for i in range(ARC_SEGS + 1):
            a = math.radians(ROM_START_DEG +
                             (ROM_END_DEG - ROM_START_DEG) * i / ARC_SEGS)
            a_pts_in .append((bx, ay(r_in,  a), az(r_in,  a)))
            a_pts_out.append((bx, ay(r_out, a), az(r_out, a)))

        # Project points onto top face (Z=Z1)
        a_top_in  = [(p[0], p[1], Z1) for p in a_pts_in]
        a_top_out = [(p[0], p[1], Z1) for p in a_pts_out]

        for i in range(ARC_SEGS):
            # Outer vertical arc wall (from arc surface up to top face)
            tris += quad(a_pts_out[i], a_pts_out[i+1],
                         a_top_out[i+1], a_top_out[i])
            # Inner vertical arc wall (reversed winding — faces inward)
            tris += quad(a_pts_in[i+1], a_pts_in[i],
                         a_top_in[i], a_top_in[i+1])
            # Bottom arc surface (connects inner to outer at arc radius)
            tris.append((a_pts_in[i],  a_pts_out[i],  a_pts_out[i+1]))
            tris.append((a_pts_in[i],  a_pts_out[i+1], a_pts_in[i+1]))

        # Start radial cap (at ROM_START_DEG)
        tris += quad(a_pts_in[0],  a_pts_out[0],
                     a_top_out[0], a_top_in[0])
        # End radial cap (at ROM_END_DEG, reversed)
        tris += quad(a_pts_out[-1], a_pts_in[-1],
                     a_top_in[-1], a_top_out[-1])

    return tris


# ---------------------------------------------------------------------------
# Back half
# ---------------------------------------------------------------------------

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

    # Outer shell: back face (-Y), left face (-X), right face (+X)
    tris += quad((X1,Y0,Z0),(X0,Y0,Z0),(X0,Y0,Z1),(X1,Y0,Z1))
    tris += quad((X0,Y0,Z0),(X0,Y1,Z0),(X0,Y1,Z1),(X0,Y0,Z1))
    tris += quad((X1,Y1,Z0),(X1,Y0,Z0),(X1,Y0,Z1),(X1,Y1,Z1))

    # Bottom plate
    tris += quad((X1,Y0,Z0),(X0,Y0,Z0),(X0,Y1,Z0),(X1,Y1,Z0))

    # Top rim
    tris += quad((X1,Y0,Z1),(X0,Y0,Z1),(X0,Y0+W,Z1),(X1,Y0+W,Z1))
    tris += quad((X0,Y1,Z1),(X0,Y0,Z1),(X0+W,Y0,Z1),(X0+W,Y1,Z1))
    tris += quad((X1,Y1,Z1),(X1-W,Y1,Z1),(X1-W,Y0,Z1),(X1,Y0,Z1))

    # Front wall (Y=0 mating plane), panels around bosses
    tris += quad((X0,Y1,Z0),(bx_l-boss_gap,Y1,Z0),
                 (bx_l-boss_gap,Y1,Z1),(X0,Y1,Z1))
    tris += quad((bx_l+boss_gap,Y1,Z0),(bx_r-boss_gap,Y1,Z0),
                 (bx_r-boss_gap,Y1,Z1),(bx_l+boss_gap,Y1,Z1))
    tris += quad((bx_r+boss_gap,Y1,Z0),(X1,Y1,Z0),
                 (X1,Y1,Z1),(bx_r+boss_gap,Y1,Z1))

    # Hinge bosses (back half socket receives front half pin through the boss)
    for bx in [bx_l, bx_r]:
        tris += boss_with_hole(bx, bz, Y0, Y1, HINGE_BOSS, HINGE_R, HINGE_SEGS)

    return tris


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

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
