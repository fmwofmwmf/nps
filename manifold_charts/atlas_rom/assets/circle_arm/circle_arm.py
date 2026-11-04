"""circle_arm: a rotating cage-arm whose tip carries a pen turned by a fixed motor.

Two freedoms.  The arm (the cage and its bar out to the tip T) turns about the
motor axis C (hand crank).  The pen arm turns about T, driven from C: pink
(about C) drives blue (about the post P, which rides on the arm) through two
coupling rods, and blue drives yellow (about T) through two output rods.  Every
crank pair is 90 degrees out of phase, so no parallelogram can fold through a
dead centre, and pink, blue and yellow always share one absolute angle.

Only pink and the arm are hinged to ground, coaxially at C.  Blue sits in the
arm's eye at P and yellow in its eye at T, each held axially by collars; the
four rods are held by pins in rings.  Layers (z, mm), bottom to top:

    O_B -36..-30   output rod B         C_B -27..-21   coupling rod B
    K_B -18..-12   crank arms B         ARM  -9..-3    arm bar, eyes, mid frame
    PEN   0..6     pen beam             K_A   9..15    crank arms A
    C_A  18..24    coupling rod A       O_A  27..33    output rod A

Rods sit outboard of every hub column, so a rod passing over C, P or T finds
nothing in its layer.  Seen from the arm, the transmission sweeps a band of
half-width R + RING_R about the arm line; the cage posts stand outside it and
its plates above and below the stack, so the cage turns with the arm without
touching anything.

Ground (column, motors, upright, table) is display only.  The pen stays in its
own layer: a pen dropped to the table would cross an output-rod layer.

Units are mm.  `pose(phi, theta)` places everything: phi is the arm angle about
C, theta the common crank angle.
"""
import math

import build123d as bd
from cad import model

# axes and lengths
D = 120.0             # motor axis C to post axis P: more than R + RING_R + COL_R, so
                      # pink's cranks clear blue's column and blue's clear pink's
L = 300.0             # P to the tip T
R = 80.0              # every crank radius: long, so a pin travels far per degree
                      # against its ring clearance
PEN_R = 70.0          # pen tip from T
PEN_DIR = 45.0        # pen direction relative to crank A, deg: between the cranks,
                      # which keeps yellow's envelope (and its SDF grid) small
PHASE_B = 90.0        # crank B relative to crank A, deg

# sections
T_LAYER = 6.0
COL_R = 8.0           # hub column
PIN_R = 4.0
SPACER_R = 7.0
CAP_T = 1.5
RING_R = 10.0         # rod eye outer radius
ROD_W = 14.0
HUB_RING_R = 18.0     # arm eyes at C, P and T
CLEAR = 1.5           # radial clearance of every ring on its pin or column: two SDF
                      # voxels at the simulation's 0.75 mm (see export_sim_meshes.py)

LAYERS = {"O_B": -36.0, "C_B": -27.0, "K_B": -18.0, "ARM": -9.0,
          "PEN": 0.0, "K_A": 9.0, "C_A": 18.0, "O_A": 27.0}

C = (0.0, 0.0)
P0 = (D, 0.0)
T0 = (D + L, 0.0)

# cage, part of the arm
CAGE_X = (-70.0, 140.0)
CAGE_Y = 105.0        # posts clear the R + RING_R = 90 band by 8 mm
POST = 14.0
CAGE_Z = (-48.0, 48.0)
PLATE_T = 8.0
KNOB_AT = 220.0       # hand-crank knob, from C along the arm

# ground, display only
UPRIGHT_X = -190.0    # behind the cage's swept radius (about 177)
TABLE_Z = -130.0

COLORS = {"pink": "#e79ac0", "blue": "#8ab4e6", "rod": "#a6d9a0", "arm": "#6dbb74",
          "cage": "#eee3b0", "yellow": "#e9cf3c"}


def _e(deg):
    return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))


def _add(p, q, s=1.0):
    return (p[0] + s * q[0], p[1] + s * q[1])


def _cyl(xy, r, z0, z1):
    return bd.Pos(xy[0], xy[1], z0) * bd.Cylinder(r, z1 - z0, align=(bd.Align.CENTER,
                                                                      bd.Align.CENTER,
                                                                      bd.Align.MIN))


def _box(x0, x1, y0, y1, z0, z1):
    return bd.Pos((x0 + x1) / 2, (y0 + y1) / 2, z0) * bd.Box(
        x1 - x0, y1 - y0, z1 - z0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))


def _bar(p0, p1, width, z0, z1):
    """A flat bar with round ends from p0 to p1."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    slot = bd.extrude(bd.SlotCenterToCenter(length, width), z1 - z0)
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    return bd.Pos(mid[0], mid[1], z0) * bd.Rot(0, 0, math.degrees(math.atan2(dy, dx))) * slot


def _layer(name):
    z0 = LAYERS[name]
    return z0, z0 + T_LAYER


def _pin(xy, segments):
    """A stepped pin: (kind, z0, z1) segments, kind 'spacer', 'pin' or 'cap'."""
    radius = {"spacer": SPACER_R, "pin": PIN_R, "cap": SPACER_R}
    parts = [_cyl(xy, radius[kind], z0, z1) for kind, z0, z1 in segments]
    return bd.Compound(parts).fuse() if len(parts) > 1 else parts[0]


def _rod(p0, p1, layer):
    z0, z1 = _layer(layer)
    body = _bar(p0, p1, ROD_W, z0, z1)
    for p in (p0, p1):
        body = body + _cyl(p, RING_R, z0, z1)
        body = body - _cyl(p, PIN_R + CLEAR, z0 - 1, z1 + 1)
    return body


def _crank_body(center, pins_a, pins_b):
    """A hub column with crank arm A (top) and crank arm B (bottom) and their pins."""
    ka0, ka1 = _layer("K_A")
    kb0, kb1 = _layer("K_B")
    tip_a = _add(center, _e(0.0), R)
    tip_b = _add(center, _e(PHASE_B), R)
    body = _cyl(center, COL_R, kb0, ka1)
    body = body + _bar(center, tip_a, ROD_W, ka0, ka1)
    body = body + _bar(center, tip_b, ROD_W, kb0, kb1)
    return body + _pin(tip_a, pins_a) + _pin(tip_b, pins_b)


def _collars(body, center):
    """Discs just above and below the arm's eye, holding a column in it axially."""
    arm0, arm1 = _layer("ARM")
    body = body + _cyl(center, HUB_RING_R, _layer("K_B")[1], arm0)
    return body + _cyl(center, HUB_RING_R, arm1, _layer("PEN")[0])


def _up(*rings):
    """Pin segments rising from K_A through the given ring layers, then a cap."""
    segments, z = [], _layer("K_A")[1]
    for name in rings:
        r0, r1 = _layer(name)
        if r0 > z:
            segments.append(("spacer", z, r0))
        segments.append(("pin", r0, r1))
        z = r1
    segments.append(("cap", z, z + CAP_T))
    return segments


def _down(*rings):
    segments, z = [], _layer("K_B")[0]
    for name in rings:
        r0, r1 = _layer(name)
        if r1 < z:
            segments.append(("spacer", r1, z))
        segments.append(("pin", r0, r1))
        z = r0
    segments.append(("cap", z - CAP_T, z))
    return segments


def pink_body():
    return _crank_body(C, _up("C_A"), _down("C_B"))


def blue_body():
    return _collars(_crank_body(P0, _up("C_A", "O_A"), _down("C_B", "O_B")), P0)


def yellow_body():
    body = _collars(_crank_body(T0, _up("O_A"), _down("O_B")), T0)
    # pen beam and tip, kept inside the PEN layer
    p0, p1 = _layer("PEN")
    tip = _add(T0, _e(PEN_DIR), PEN_R)
    body = body + _bar(T0, tip, 10.0, p0, p1)
    return body + bd.Pos(tip[0], tip[1], p0) * bd.Cone(
        5.0, 1.0, p1 - p0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))


def cage():
    """The cage frame, rigid with the arm: posts, top and bottom frames, mid frame."""
    (x0, x1), (z0, z1) = CAGE_X, CAGE_Z
    a0, a1 = _layer("ARM")
    parts = []
    for x in (x0, x1):
        for y in (-CAGE_Y, CAGE_Y):
            parts.append(_box(x - POST / 2, x + POST / 2, y - POST / 2, y + POST / 2,
                              z0 - PLATE_T, z1 + PLATE_T))
    frames = [(z0 - PLATE_T, z0), (z1, z1 + PLATE_T), (a0, a1)]
    for f0, f1 in frames:
        for y in (-CAGE_Y, CAGE_Y):
            parts.append(_box(x0 - POST / 2, x1 + POST / 2, y - POST / 2, y + POST / 2, f0, f1))
        for x in (x0, x1):
            parts.append(_box(x - POST / 2, x + POST / 2, -CAGE_Y, CAGE_Y, f0, f1))
    # top and bottom bridges carrying the bearing bosses over C and P
    for f0, f1 in frames[:2]:
        parts.append(_box(x0, x1, -10.0, 10.0, f0, f1))
        parts.append(_cyl(C, HUB_RING_R, f0, f1))
        parts.append(_cyl(P0, HUB_RING_R, f0, f1))
    # mid frame back to the eye at C, stopping inside the eye's ring, clear of its bore
    parts.append(_box(x0, C[0] - HUB_RING_R + 2.0, -POST / 2, POST / 2, a0, a1))
    body = bd.Compound(parts).fuse()
    # the shaft bores at C, and bolt heads on the top corners
    body = body - _cyl(C, 6.0, z0 - PLATE_T - 1, z1 + PLATE_T + 1)
    for x in (x0, x1):
        for y in (-CAGE_Y, CAGE_Y):
            body = body + bd.Pos(x, y, z1 + PLATE_T) * bd.extrude(bd.RegularPolygon(5.5, 6), 4.0)
    return body


def arm_bar():
    """The bar from C out to T, its eyes at C, P and T, and the hand-crank knob."""
    z0, z1 = _layer("ARM")
    body = _bar(C, T0, ROD_W, z0, z1)
    for p in (C, P0, T0):
        body = body + _cyl(p, HUB_RING_R, z0, z1)
        body = body - _cyl(p, COL_R + CLEAR, z0 - 1, z1 + 1)
    # knob kept below K_A, so no rod or crank can reach it
    knob = _add(C, _e(0.0), KNOB_AT)
    return body + _cyl(knob, 8.0, z1, _layer("PEN")[1])


def coupling_rod(phase):
    layer = "C_A" if phase == 0.0 else "C_B"
    return _rod(_add(C, _e(phase), R), _add(P0, _e(phase), R), layer)


def output_rod(phase):
    layer = "O_A" if phase == 0.0 else "O_B"
    return _rod(_add(P0, _e(phase), R), _add(T0, _e(phase), R), layer)


# ground, display only: ground has no collision geometry
def top_motor_visual():
    z = CAGE_Z[1] + PLATE_T
    return _cyl(C, 30.0, z + 30.0, z + 80.0) + _cyl(C, 5.0, z, z + 30.0)


def bottom_motor_visual():
    """The housing below C, as in the video.  Decorative: the arm is hand cranked."""
    z = CAGE_Z[0] - PLATE_T
    return _cyl(C, 30.0, z - 60.0, z - 15.0) + _cyl(C, 5.0, z - 15.0, z)


def upright_visual():
    top = CAGE_Z[1] + PLATE_T + 80.0
    bottom = CAGE_Z[0] - PLATE_T - 60.0
    stand = _box(UPRIGHT_X - 10.0, UPRIGHT_X + 10.0, -45.0, 45.0, TABLE_Z, top + 20.0)
    foot = _box(UPRIGHT_X - 45.0, UPRIGHT_X + 45.0, -60.0, 60.0, TABLE_Z, TABLE_Z + 12.0)
    # brackets carrying the two motors
    upper = _box(UPRIGHT_X + 10.0, C[0] + 30.0, -30.0, 30.0, top, top + 10.0)
    lower = _box(UPRIGHT_X + 10.0, C[0] + 30.0, -30.0, 30.0, bottom - 10.0, bottom)
    return stand + foot + upper + lower


def table_visual():
    return _box(-230.0, 520.0, -420.0, 420.0, TABLE_Z - 6.0, TABLE_Z)


GROUND_VISUALS = (
    ("ground_upright", upright_visual, "#c6cad1"),
    ("ground_top_motor", top_motor_visual, "#cbc5a6"),
    ("ground_bottom_motor", bottom_motor_visual, "#b5a2ea"),
    ("ground_table", table_visual, "#f3f3f1"),
)


def _about(center, deg):
    return bd.Pos(center[0], center[1], 0) * bd.Rot(0, 0, deg) * bd.Pos(-center[0], -center[1], 0)


def _carry(start0, start, phi):
    """Move a rest-pose part so start0 lands on start, turned by phi."""
    return bd.Pos(start[0], start[1], 0) * bd.Rot(0, 0, phi) * bd.Pos(-start0[0], -start0[1], 0)


def pose(phi=0.0, theta=0.0):
    """The assembly at arm angle phi (about C) and crank angle theta, both in degrees."""
    u = _e(phi)
    post = _add(C, u, D)
    tip = _add(C, u, D + L)
    placed = []

    def put(shape, location, label, color):
        shape = location * shape
        shape.label, shape.color = label, color
        placed.append(shape)

    arm = _about(C, phi)
    put(arm_bar(), arm, "arm", COLORS["arm"])
    put(cage(), arm, "arm_cage", COLORS["cage"])
    put(pink_body(), _about(C, theta), "pink", COLORS["pink"])
    put(blue_body(), bd.Pos(post[0] - P0[0], post[1] - P0[1], 0) * _about(P0, theta),
        "blue", COLORS["blue"])
    put(yellow_body(), bd.Pos(tip[0] - T0[0], tip[1] - T0[1], 0) * _about(T0, theta),
        "yellow", COLORS["yellow"])
    for phase, tag in ((0.0, "A"), (PHASE_B, "B")):
        e0, e = _e(phase), _e(phase + theta)
        put(coupling_rod(phase), _carry(_add(C, e0, R), _add(C, e, R), phi),
            f"coupling_{tag}", COLORS["rod"])
        put(output_rod(phase), _carry(_add(P0, e0, R), _add(post, e, R), phi),
            f"output_{tag}", COLORS["rod"])
    for label, build, color in GROUND_VISUALS:
        put(build(), bd.Location(), label, color)
    assembly = bd.Compound(children=placed)
    assembly.label = "circle_arm"
    return assembly


@model
def circle_arm():
    return pose(0.0, 0.0)


@model
def circle_arm_posed():
    return pose(35.0, 130.0)


if __name__ == "__main__":
    circle_arm()
    circle_arm_posed()
