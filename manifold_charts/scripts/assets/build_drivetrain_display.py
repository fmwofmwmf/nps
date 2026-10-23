"""Build assets/drivetrain/display.blend: detailed render-only parts for the
`drivetrain` case, from the same `_drivetrain_params` the case's joints use.

    blender -b --factory-startup --python scripts/assets/build_drivetrain_display.py [-- [--v8] [OUT.blend]]

`--v8` builds the `drivetrain_v8` engine instead (assets/drivetrain_v8/display.blend):
two open-topped translucent banks with open liners, so the piston crowns show.

Everything is modelled in the case's world frame at the REST pose (every body
at transform = identity), so `viewers/blender_import_rom.py --display` can hang
each part on its animated body with an identity parent-inverse.  Names follow
that importer's convention:

    <body>.<part>   detail riding on that body (tyres, rims, discs, boots)
    anything else   static scenery (engine block, calipers, stand)

None of this geometry reaches the solver or the learner; the case's own
closed functional meshes set mass and inertia.
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                      # manifold_charts
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from atlas_rom.cases import _drivetrain_params as P             # noqa: E402

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
V8 = "--v8" in ARGV
_POS = [a for a in ARGV if not a.startswith("--")]
OUT = _POS[0] if _POS else os.path.join(REPO, "assets", "drivetrain_v8" if V8 else "drivetrain", "display.blend")


# ── tiny mesh kit (pure numpy -> from_pydata; no operator context needed) ────
class Part:
    def __init__(self):
        self.V, self.F = [], []

    def add(self, V, F):
        off = sum(len(v) for v in self.V)
        self.V.append(np.asarray(V, float))
        self.F += [[i + off for i in f] for f in F]
        return self


def revolve_x(profile, cx, segments=96, mirror=1.0, closed=False):
    """Revolve (axial offset a, radius r) points about the world x axis at x=cx.
    `mirror` = -1 flips the axial offsets (the left side of the car)."""
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    V = [(cx + mirror * a, r * math.cos(tt), r * math.sin(tt)) for a, r in profile for tt in t]
    n, m = segments, len(profile)
    F = []
    for i in range(m if closed else m - 1):
        i2 = (i + 1) % m
        for k in range(n):
            k2 = (k + 1) % n
            F.append([i * n + k, i * n + k2, i2 * n + k2, i2 * n + k])
    return V, F


def box(center, half, R=np.eye(3)):
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    V = (c * np.asarray(half, float)) @ np.asarray(R).T + np.asarray(center, float)
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return V, F


def frustum(p0, p1, half0, half1, up):
    """A tapered bar from p0 to p1 with rectangular sections (half widths)."""
    p0, p1, up = map(lambda v: np.asarray(v, float), (p0, p1, up))
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    w = np.cross(d, up); w /= np.linalg.norm(w)
    u = np.cross(w, d)
    V = []
    for p, (hw, hu) in ((p0, half0), (p1, half1)):
        for sw in (-1, 1):
            for su in (-1, 1):
                V.append(p + sw * hw * w + su * hu * u)
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return V, F


def prism_x(center, radius, length, sides=6, phase=0.0):
    cx, cy, cz = center
    t = phase + np.arange(sides) * 2 * np.pi / sides
    V = [(cx + s * length / 2, cy + radius * math.cos(a), cz + radius * math.sin(a))
         for s in (-1, 1) for a in t]
    F = [list(range(sides))[::-1], list(range(sides, 2 * sides))]
    F += [[k, (k + 1) % sides, sides + (k + 1) % sides, sides + k] for k in range(sides)]
    return V, F


def onto_axis(V, axis):
    """Rotate geometry built about +z onto `axis` (a proper rotation; axis.z > -1)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    v, c = np.cross([0.0, 0.0, 1.0], a), float(a[2])
    if np.linalg.norm(v) < 1e-12:
        return np.asarray(V, float)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K / (1.0 + c)
    return np.asarray(V, float) @ R.T


def tube_axis(base, axis, r_in, r_out, s0, s1, segments=64):
    """A closed annular solid along `axis` from base + s0*axis to base + s1*axis:
    an open-ended liner."""
    V, F = revolve_x([(s0, r_in), (s0, r_out), (s1, r_out), (s1, r_in)], 0.0, segments, closed=True)
    V = np.asarray(V)[:, [1, 2, 0]]                                   # revolved about z
    return onto_axis(V, axis) + np.asarray(base, float), F


def ring_x(cx, r_in, r_out, width, segments=96):
    """A closed annular solid about the x axis (washer / disc)."""
    h = width / 2
    return revolve_x([(-h, r_in), (-h, r_out), (h, r_out), (h, r_in)], cx, segments, closed=True)


def tube_z(center, r_in, r_out, height, segments=64):
    """A closed annular solid about a vertical axis (cylinder liner)."""
    V, F = revolve_x([(-height / 2, r_in), (-height / 2, r_out), (height / 2, r_out),
                      (height / 2, r_in)], 0.0, segments, closed=True)
    V = np.asarray(V)[:, [1, 2, 0]] + np.asarray(center, float)     # x-axis -> z-axis
    return V, F


# ── materials ────────────────────────────────────────────────────────────────
def material(name, rgb, metallic=0.0, roughness=0.5, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = roughness
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "BLENDED"
    return m


MAT = {}


def mats():
    MAT.update(
        rubber=material("rubber", (0.025, 0.025, 0.027), roughness=0.88),
        # metallic kept well under 1: on the renderer's plain white world a fully metallic surface reflects
        # nothing but white and reads as flat silver
        alloy=material("alloy", (0.78, 0.79, 0.81), metallic=0.6, roughness=0.35),
        chrome=material("chrome", (0.92, 0.92, 0.93), metallic=0.7, roughness=0.2),
        disc=material("brake_disc", (0.45, 0.45, 0.46), metallic=0.6, roughness=0.42),
        # white at rest; the importer turns a caliper with `active_when` red while its brake is engaged
        caliper=material("caliper", (0.90, 0.90, 0.90), roughness=0.35),
        glass=material("engine_glass", (0.70, 0.82, 0.95), roughness=0.05, alpha=0.18),
        liner=material("liner", (0.55, 0.56, 0.58), metallic=0.2, roughness=0.3, alpha=0.45),
        head=material("cast_aluminium", (0.66, 0.67, 0.69), metallic=0.3, roughness=0.45),
        cover=material("valve_cover", (0.62, 0.04, 0.04), metallic=0.3, roughness=0.3),
        paint=material("stand_paint", (0.10, 0.11, 0.13), metallic=0.2, roughness=0.6),
        bearing=material("bearing", (0.30, 0.31, 0.33), metallic=0.4, roughness=0.35),
    )


def emit(name, part, mat, smooth=True):
    me = bpy.data.meshes.new(name)
    V = np.vstack(part.V) if part.V else np.zeros((0, 3))
    me.from_pydata(V.tolist(), [], part.F)
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = smooth
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(MAT[mat])
    bpy.context.scene.collection.objects.link(ob)
    if smooth:                                    # keep hard edges crisp under smooth shading
        mod = ob.modifiers.new("smooth_by_angle", "EDGE_SPLIT")
        mod.split_angle = math.radians(35.0)
    return ob


# ── wheels and axles (ride on axle_left / axle_right) ────────────────────────
TYRE_W, RIM_R = 0.66, 0.86


CROWN, CROWN_HALF = 0.10, 0.44                                  # crowned (wide) tread base


def tread_base(a):
    """Radius of the crowned tread base at axial offset `a` from the wheel's plane."""
    return P.WHEEL_RADIUS - 0.035 - CROWN * (a / CROWN_HALF) ** 2


def tread_slope(a):
    return -2.0 * CROWN * a / CROWN_HALF ** 2


def wheel(side, wide=False, brake=True):
    """`wide`: a ~1.44x wider tyre and rim with three staggered tread rows and
    more blocks round the circumference (the V8).  `brake`: a vented disc."""
    s = 1.0 if side == "right" else -1.0
    cx = s * P.WHEEL_X
    body = f"axle_{side}"
    Ro = P.WHEEL_RADIUS
    k = 1.44 if wide else 1.0                                         # axial scale
    if wide:
        # Crowned tread: the tread base bulges, r(a) = Ro - 0.035 - CROWN*(a/HALF)^2,
        # rolling into the shoulders and the sidewalls.
        sidewall = [(0.43, RIM_R + 0.02), (0.475, RIM_R + 0.12), (0.497, RIM_R + 0.20),
                    (0.48, tread_base(CROWN_HALF) - 0.035)]                   # shoulder meets the crown
        crown = [(a, tread_base(a)) for a in np.linspace(0.44, -0.44, 23)]
        prof = ([(-a, r) for a, r in sidewall] + [(a, r) for a, r in reversed(crown)]
                + [(a, r) for a, r in reversed(sidewall)])
    else:
        # tyre carcass: bead -> bulged sidewall -> shoulder -> tread base -> back
        prof = [(-0.30, RIM_R + 0.02), (-0.33, RIM_R + 0.12), (-0.345, RIM_R + 0.24),
                (-0.33, Ro - 0.12), (-0.29, Ro - 0.055), (-0.22, Ro - 0.035),
                (0.22, Ro - 0.035), (0.29, Ro - 0.055), (0.33, Ro - 0.12),
                (0.345, RIM_R + 0.24), (0.33, RIM_R + 0.12), (0.30, RIM_R + 0.02)]
    emit(f"{body}.tyre", Part().add(*revolve_x(prof, cx, 128, closed=True)), "rubber")
    # tread: staggered rows of chevron-skewed blocks separated by grooves; on the
    # crowned tyre each row sits on, and tilts with, the curved tread base
    rows = (((-0.30, 0.0), (-0.10, 0.5), (0.10, 0.0), (0.30, 0.5)) if wide
            else ((-0.12, 0.0), (0.12, 0.5)))
    half_a = 0.08 if wide else 0.095
    tread, n = Part(), (84 if wide else 60)
    for row, (a_off, stagger) in enumerate(rows):
        r_row = tread_base(a_off) + 0.017 if wide else Ro - 0.018
        tilt = math.atan(tread_slope(a_off)) if wide else 0.0        # follow the crown
        for j in range(n):
            t = 2 * np.pi * (j + stagger) / n
            er = np.array([0.0, math.cos(t), math.sin(t)])
            et = np.array([0.0, -math.sin(t), math.cos(t)])
            ea = np.array([math.cos(tilt), 0.0, 0.0]) + math.sin(tilt) * er
            R = np.column_stack([ea, et, -math.sin(tilt) * np.array([1.0, 0.0, 0.0]) + math.cos(tilt) * er])
            chevron = 0.35                                          # tangential shear per axial unit
            skew = np.array([[1, 0, 0], [chevron * (-1 if row % 2 else 1), 1, 0], [0, 0, 1]], float)
            tread.add(*box((cx + a_off, *(er[1:] * r_row)),
                           (half_a, 0.045 * 60 / n, 0.022), R @ skew))
    emit(f"{body}.tread", tread, "rubber", smooth=False)
    # rim barrel with flanges; the outboard face carries the spokes
    rim = [(-0.31 * k, RIM_R + 0.07), (-0.29 * k, RIM_R + 0.07), (-0.27 * k, RIM_R - 0.01),
           (0.27 * k, RIM_R - 0.01), (0.29 * k, RIM_R + 0.07), (0.31 * k, RIM_R + 0.07),
           (0.31 * k, RIM_R - 0.06), (-0.31 * k, RIM_R - 0.06)]
    emit(f"{body}.rim", Part().add(*revolve_x(rim, cx, 128, s, closed=True)), "alloy")
    spokes = Part()
    face = cx + s * 0.20 * k
    for j in range(10):
        t = 2 * np.pi * j / 10 + (0.12 if j % 2 else -0.12)          # five split-spoke pairs
        p0 = (face + s * 0.05, 0.22 * math.cos(t), 0.22 * math.sin(t))
        p1 = (face - s * 0.06, (RIM_R - 0.05) * math.cos(t), (RIM_R - 0.05) * math.sin(t))
        spokes.add(*frustum(p0, p1, (0.05, 0.04), (0.028, 0.06), (1.0, 0.0, 0.0)))
    spokes.add(*ring_x(face + s * 0.05, 0.0, 0.25, 0.08))           # centre hub
    if wide:                                                          # hub barrel back to the axle flange
        spokes.add(*ring_x((face + s * P.HUB_X) / 2, 0.0, 0.12, abs(face - s * P.HUB_X)))
    emit(f"{body}.spokes", spokes, "alloy")
    lugs = Part()
    for j in range(5):
        t = 2 * np.pi * j / 5
        lugs.add(*prism_x((face + s * 0.11, 0.16 * math.cos(t), 0.16 * math.sin(t)), 0.035, 0.07))
    lugs.add(*ring_x(face + s * 0.12, 0.0, 0.075, 0.03, 48))        # centre cap
    emit(f"{body}.lugs", lugs, "chrome", smooth=False)
    if brake:                                                         # vented disc inboard of the hub
        dx = s * P.BRAKE_X
        disc = Part()
        for off in (-0.04, 0.04):
            disc.add(*ring_x(dx + off, 0.36, 0.66, 0.035))
        for j in range(36):
            t = 2 * np.pi * j / 36
            disc.add(*frustum((dx, 0.38 * math.cos(t), 0.38 * math.sin(t)),
                              (dx, 0.64 * math.cos(t + 0.15), 0.64 * math.sin(t + 0.15)),
                              (0.008, 0.024), (0.012, 0.024), (1.0, 0.0, 0.0)))
        disc.add(*ring_x(dx + s * 0.07, 0.12, 0.37, 0.10))          # hat
        emit(f"{body}.disc", disc, "disc")
    # CV-joint boots: accordion bellows at both ends of the half-shaft
    for tag, x0 in (("boot_inner", s * P.BOOT_INNER_X), ("boot_outer", s * P.BOOT_OUTER_X)):
        prof = [(-0.14, 0.07)]
        for j in range(7):
            prof.append((-0.12 + j * 0.04, 0.17 - 0.012 * j if j % 2 == 0 else 0.12 - 0.008 * j))
        prof.append((0.16, 0.065))
        emit(f"{body}.{tag}", Part().add(*revolve_x(prof, x0, 64, s)), "rubber")


def brakes_and_uprights(side, bracket=False):
    """Static: caliper over the disc's rear-top quadrant, carried by an upright
    behind it -- or, with `bracket`, by an arm from the stand's axle pedestal
    that stays inside a wide rim."""
    s = 1.0 if side == "right" else -1.0
    dx = s * P.BRAKE_X
    # rear-top of the disc; on a bracket, rear-BOTTOM, so the arm from the
    # pedestal runs at almost one angle and never crosses the shaft or boots
    t = math.radians(240.0 if bracket else 120.0)
    er = np.array([0.0, math.cos(t), math.sin(t)])
    et = np.array([0.0, -math.sin(t), math.cos(t)])
    R = np.column_stack([np.array([1.0, 0, 0]), et, er])
    # A U straddling the disc (faces at +-0.0575): a pad housing either side and
    # a bridge over the rim (outer radius 0.66), touching it nowhere.
    cal = Part()
    for side_sign in (-1.0, 1.0):
        cal.add(*box((dx + side_sign * 0.085, *(er[1:] * 0.55)), (0.025, 0.18, 0.09), R))
    cal.add(*box((dx, *(er[1:] * 0.715)), (0.11, 0.16, 0.035), R))
    ob = emit(f"caliper_{side}", cal, "caliper", smooth=False)
    ob["active_when"] = f"{side} wheel brake"     # red while that brake actuator is engaged (inline-4 declares only the left one)
    ob["active_color"] = (0.72, 0.06, 0.05)
    if bracket:
        arm = Part().add(*frustum((s * P.PEDESTAL_X, -0.09, -0.30), (dx - s * 0.11, *(er[1:] * 0.50)),
                                  (0.05, 0.04), (0.04, 0.035), (1.0, 0.0, 0.0)))
        emit(f"caliper_bracket_{side}", arm, "paint", smooth=False)
        return
    up = Part().add(*frustum((s * (P.WHEEL_X - 0.42), 0.0, 0.30), (s * (P.WHEEL_X - 0.42), 0.0, -0.95),
                             (0.07, 0.10), (0.06, 0.08), (0.0, 1.0, 0.0)))
    up.add(*ring_x(s * (P.WHEEL_X - 0.42), 0.07, 0.17, 0.12, 64))
    emit(f"upright_{side}", up, "paint", smooth=False)


def engine():
    x0, x1 = P.CYL_X[0] - 0.30, P.CYL_X[-1] + 0.30
    y0, y1 = P.CRANK_Y - 0.30, P.CRANK_Y + 0.30
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    top = P.CRANK_Z + P.CRANK_RADIUS + P.ROD_LENGTH + 0.22               # above the TDC crown
    emit("engine_block", Part().add(*box((xm, ym, (top - 0.32) / 2), ((x1 - x0) / 2, (y1 - y0) / 2,
                                                                        (top + 0.32) / 2))),
         "glass", smooth=False)
    liners = Part()
    for x in P.CYL_X:
        liners.add(*tube_z((x, P.CRANK_Y, top - 0.30), P.BORE_RADIUS + 0.005, P.BORE_RADIUS + 0.03, 0.58))
    emit("cylinder_liners", liners, "liner")
    emit("cylinder_head", Part().add(*box((xm, ym, top + 0.10), ((x1 - x0) / 2 - 0.02, (y1 - y0) / 2 - 0.02, 0.10))),
         "head", smooth=False)
    cover = Part().add(*box((xm, ym, top + 0.26), ((x1 - x0) / 2 - 0.08, (y1 - y0) / 2 - 0.10, 0.06)))
    for k in range(6):                                                # cast fins on the cover
        cover.add(*box((x0 + 0.25 + k * (x1 - x0 - 0.5) / 5, ym, top + 0.33), (0.015, (y1 - y0) / 2 - 0.14, 0.02)))
    emit("valve_cover", cover, "cover", smooth=False)
    emit("oil_pan", Part().add(*box((xm, ym, -0.40), ((x1 - x0) / 2 - 0.05, (y1 - y0) / 2 - 0.04, 0.08))),
         "head", smooth=False)


def engine_v8():
    """Two banks of open cylinder liners along the fore-aft crank -- no block,
    no heads -- so the pistons are visible; and an oil pan."""
    fore = np.array([0.0, 1.0, 0.0])                                     # crank axis
    s_liner, s_deck = 0.36, 0.84
    for bank in ("A", "B"):
        a = np.asarray(P.V8_BANK_AXIS[bank])
        ys = [y + P.V8_ROD_OFFSET[bank] for y in P.V8_PIN_Y]
        liners = Part()
        for yy in ys:
            liners.add(*tube_axis(yy * fore, a, P.BORE_RADIUS + 0.005, P.BORE_RADIUS + 0.03,
                                  s_liner, s_deck))
        emit(f"bank_{bank}_liners", liners, "liner")
    y0, y1 = P.V8_PIN_Y[0] - 0.30, P.V8_PIN_Y[-1] + 0.30
    emit("oil_pan", Part().add(*box((0.0, (y0 + y1) / 2, -0.40), (0.26, (y1 - y0) / 2, 0.08))),
         "head", smooth=False)


def stand():
    base = Part().add(*(box((0.0, 1.55, P.STAND_Z), (2.7, 3.0, 0.07)) if V8 else
                        box((0.0, 0.35, P.STAND_Z), (2.7, 1.1, 0.07))))
    emit("stand_base", base, "paint", smooth=False)
    posts = Part()
    for sx in (-1.0, 1.0):
        x = sx * P.PEDESTAL_X
        posts.add(*box((x, 0.0, (P.STAND_Z - 0.18) / 2), (0.07, 0.09, (-P.STAND_Z - 0.18) / 2)))
        posts.add(*box((x, 0.0, -0.15), (0.10, 0.16, 0.06)))
    if V8:                                                            # under the oil pan, and the rear shaft
        for y in (P.V8_PIN_Y[0] - 0.30, P.V8_PIN_Y[-1] + 0.30):
            posts.add(*box((0.0, y, (P.STAND_Z - 0.48) / 2), (0.08, 0.08, (-P.STAND_Z - 0.48) / 2)))
        posts.add(*box((0.0, 1.30, (P.STAND_Z - 0.13) / 2), (0.06, 0.06, (-P.STAND_Z - 0.13) / 2)))
    else:
        for x in (P.CYL_X[0] - 0.30, P.CYL_X[-1] + 0.30):
            posts.add(*box((x, P.CRANK_Y, (P.STAND_Z - 0.48) / 2), (0.08, 0.08, (-P.STAND_Z - 0.48) / 2)))
    emit("stand_posts", posts, "paint", smooth=False)
    bearings = Part()
    for sx in (-1.0, 1.0):
        bearings.add(*ring_x(sx * P.PEDESTAL_X, P.SHAFT_RADIUS + 0.005, 0.15, 0.16))
    if V8:                                                            # pillow block on the crank's rear shaft
        bearings.add(*tube_axis((0.0, 1.30, 0.0), (0.0, 1.0, 0.0), P.JOURNAL_RADIUS + 0.005, 0.13, -0.08, 0.08))
    else:
        bearings.add(*ring_x(P.JOURNAL_X[1] - 0.03, P.JOURNAL_RADIUS + 0.005, 0.13, 0.06))
    emit("pillow_blocks", bearings, "bearing")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats()
    for side in ("left", "right"):
        wheel(side, wide=V8)
        brakes_and_uprights(side, bracket=V8)
    engine_v8() if V8 else engine()
    stand()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUT)
    n_parts = sum(1 for ob in bpy.data.objects if "." in ob.name)
    n_tris = sum(len(ob.data.polygons) for ob in bpy.data.objects if ob.type == "MESH")
    print(f"saved {OUT}: {len(bpy.data.objects)} objects ({n_parts} body-mounted), {n_tris} faces")


main()
