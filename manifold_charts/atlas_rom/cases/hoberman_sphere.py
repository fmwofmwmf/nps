"""cases/hoberman_sphere.py — a Hoberman sphere: six great-circle rings of
angulated scissor pairs, joined at the 30 vertices of an icosidodecahedron.

Angulated scissor ring
-----------------------
A planar ring of `n` scissor units, each two identical kinked rods pinned at
their kinks.  In the unit's own frame (bisector along +y, centre of the ring
at the origin) rod A runs from its kink C = (0, r) out to

    A1 = C + a (cos g, sin g)          A2 = C + a (cos(g + k), sin(g + k))

and rod B is its mirror image across the bisector.  With the kink set to
k = pi - theta, theta = 2 pi / n, both end pivots on each side stay on a fixed
radial line (at +-theta/2 from the bisector) for EVERY opening g, with
r = -a cos(g - theta/2) / sin(theta/2).  So n units close a ring that only
grows and shrinks radially — the Hoberman condition.  The rod angle g is how
the geometry is built here; nothing about it reaches the pipeline.

Sphere
------
The icosidodecahedron's 60 edges lie on six great decagons.  Each decagon
carries one ring of `n = 10 * UNITS_PER_EDGE` units, its pivots on axes normal
to the decagon's plane.  At every vertex two rings cross; each vertex gets an
outer and an inner hub, and each ring's two rods meeting there pin to the hub
about that ring's own normal.  Between vertices the rods of one ring pin
directly to each other.  The hubs only translate radially.

The mechanism is heavily overconstrained (the naive joint count says it is
rigid); it moves only because the geometry above is exact.  Per
`finite-vs-instantaneous-mobility`, that has to be measured, not assumed.

Grounding: the outer hub at the bottom vertex is the fixed body (the sphere
sits on a post).  That removes the six rigid-body freedoms without any
redundant ground constraint.  Each central pivot carries a stop on its own
travel (`limit_lo` / `limit_hi`): fully opened, inner and outer hubs meet;
fully closed, the inner hubs meet at the centre.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, JointDef, MechanismSpec, MeshSpec, SamplingOptions

# ─── geometry ────────────────────────────────────────────────────────────────

UNITS_PER_EDGE = 2
N_UNITS = 10 * UNITS_PER_EDGE              # units per great-circle ring
THETA = 2.0 * math.pi / N_UNITS            # angle one unit subtends at the centre
KINK = math.pi - THETA                     # rod kink angle (the Hoberman condition)
ARM = 0.15                                 # length of each rod half (kink to end)
ROD_THICK = 0.018
HUB_RADIUS = 0.022
GAMMA_LO = math.radians(118.0)             # nearly closed: inner hubs approach the centre
GAMMA_HI = math.radians(178.0)             # nearly open: inner and outer hubs approach each other
GAMMA_REST = 0.5 * (GAMMA_LO + GAMMA_HI)
# Each pivot's travel is twice the rod's in-plane turn (rod B mirrors rod A).
PIVOT_LIMIT = 2.0 * (GAMMA_HI - GAMMA_REST)

_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _icosidodecahedron():
    """The 30 vertices (unit vectors) and the six great-decagon normals."""
    verts = []
    for s in (-1.0, 1.0):
        for axis in range(3):
            v = np.zeros(3)
            v[axis] = s * _PHI
            verts.append(v)
    base = np.array([0.5, _PHI / 2.0, _PHI * _PHI / 2.0])
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                v = base * (sx, sy, sz)
                for shift in range(3):
                    verts.append(np.roll(v, shift))
    verts = np.array(verts) / _PHI
    # The decagon normals are the icosahedron's five-fold axes; of the two
    # icosahedron orientations, keep the one whose axes each have exactly ten
    # vertices on their equator (one normal per +- pair).
    normals = []
    for x, y in ((1.0, _PHI), (_PHI, 1.0)):
        for s in (-1.0, 1.0):
            for shift in range(3):
                n = np.roll(np.array([0.0, x, s * y]), shift)
                n /= np.linalg.norm(n)
                if (np.sum(np.abs(verts @ n) < 1e-9) == 10
                        and not any(abs(n @ m) > 1 - 1e-9 for m in normals)):
                    normals.append(n)
    assert len(normals) == 6, len(normals)
    return verts, np.array(normals)


VERTICES, NORMALS = _icosidodecahedron()
BASE_VERTEX = int(np.argmin(VERTICES[:, 2]))          # the vertex pointing straight down


def _ring_frame(normal):
    """In-plane basis (e1 on a vertex, e2 = n x e1) and the decagon's vertices
    in angular order from e1."""
    on = [i for i, v in enumerate(VERTICES) if abs(v @ normal) < 1e-9]
    assert len(on) == 10, len(on)
    e1 = VERTICES[on[0]]
    e2 = np.cross(normal, e1)
    angle = {i: math.atan2(VERTICES[i] @ e2, VERTICES[i] @ e1) % (2 * math.pi) for i in on}
    return e1, e2, angle


def _unit_points(gamma):
    """Rod geometry in a unit's frame: kink C and the four end points."""
    r = -ARM * math.cos(gamma - THETA / 2.0) / math.sin(THETA / 2.0)
    c = np.array([0.0, r])
    a1 = c + ARM * np.array([math.cos(gamma), math.sin(gamma)])
    a2 = c + ARM * np.array([math.cos(gamma + KINK), math.sin(gamma + KINK)])
    mirror = np.array([-1.0, 1.0])
    return c, a1, a2, a2 * mirror, a1 * mirror        # C, A1, A2, B1, B2


def _radii(gamma):
    """Outer / inner hub distance from the sphere centre."""
    _c, a1, a2, b1, _b2 = _unit_points(gamma)
    side = np.array([-math.sin(THETA / 2.0), math.cos(THETA / 2.0)])
    return float(a1 @ side), float(b1 @ side)


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _assembly(gamma):
    """World transforms of every body at opening `gamma`, plus the world
    pivot points the joints are built from.  The sphere centre is shifted so
    the base hub stays where it is at the rest opening."""
    out_r, in_r = _radii(gamma)
    centre = np.array([0.0, 0.0, out_r - _radii(GAMMA_REST)[0]])
    poses, points = {}, {}
    _c, a1, a2, b1, b2 = _unit_points(gamma)
    for k, n in enumerate(NORMALS):
        e1, e2, _angle = _ring_frame(n)
        for j in range(N_UNITS):
            phi = (j + 0.5) * THETA
            ey = math.cos(phi) * e1 + math.sin(phi) * e2
            ex = math.sin(phi) * e1 - math.cos(phi) * e2
            frame = np.column_stack((ex, ey, n))
            world = lambda p: centre + frame @ np.array([p[0], p[1], 0.0])
            kink = world(_c)
            # Rod frames are world-aligned at rest (identity rotation, origin
            # at the kink).  Rod A turns by (gamma - GAMMA_REST) about n; rod
            # B, its mirror, by the opposite amount.
            turn = gamma - GAMMA_REST
            for rod, sign in (("a", 1.0), ("b", -1.0)):
                T = np.eye(4)
                T[:3, :3] = frame @ _rot_z(sign * turn) @ frame.T
                T[:3, 3] = kink
                poses[f"c{k}_u{j}_{rod}"] = T
            points[(k, j)] = {"kink": kink, "a1": world(a1), "a2": world(a2),
                              "b1": world(b1), "b2": world(b2), "frame": frame}
    for v, d in enumerate(VERTICES):
        for level, rad in (("out", out_r), ("in", in_r)):
            T = np.eye(4)
            T[:3, 3] = centre + rad * d
            poses[_hub(v, level)] = T
    return poses, points


def _hub(v, level):
    return "hub_base" if (v == BASE_VERTEX and level == "out") else f"hub{v}_{level}"


# ─── bodies and joints ───────────────────────────────────────────────────────

_POSE, _POINTS = _assembly(GAMMA_REST)
_ROD_COLORS = [(0.86, 0.33, 0.25), (0.95, 0.68, 0.20), (0.40, 0.72, 0.38),
               (0.25, 0.58, 0.85), (0.55, 0.40, 0.80), (0.90, 0.45, 0.65)]


def _rod(k, j, rod):
    """A kinked rod in its own (world-aligned) frame, kink at the origin.
    Rod B sits one thickness above rod A along the pivot axis so the pair
    does not overlap; the joints are defined by pivot and axis alone, so the
    offset is visual only."""
    p = _POINTS[(k, j)]
    ends = (p["a1"], p["a2"]) if rod == "a" else (p["b2"], p["b1"])
    lift = (-0.5 if rod == "a" else 0.5) * ROD_THICK * p["frame"][:, 2]
    # Cylinders, not ``link``: a link bar lies in the xy-plane only, and every
    # ring here is tilted out of it.  Each half runs diagonally from the rod's
    # layer at the crossing pivot (+-half a thickness along the pivot axis, so
    # the pair clears each other there) to the ring's mid-plane at its end, so
    # neighbouring rods meet flush at every end pin and hub.
    parts = []
    for e in ends:
        seg = (e - p["kink"]) - lift
        parts.append(MeshSpec("cylinder", {"radius": 0.5 * ROD_THICK, "height": float(np.linalg.norm(seg)),
                                           "center": (lift + 0.5 * seg).tolist(), "axis": seg.tolist(),
                                           "sections": 8}))
    return BodyDef(f"c{k}_u{j}_{rod}", MeshSpec("compound", {"parts": parts}),
                   initial_transform=_POSE[f"c{k}_u{j}_{rod}"], color=_ROD_COLORS[k])


def _hub_body(v, level):
    name = _hub(v, level)
    parts = [MeshSpec("sphere", {"radius": HUB_RADIUS, "center": [0, 0, 0], "subdiv": 1})]
    if name == "hub_base":
        # The post the sphere stands on.
        parts.append(MeshSpec("cylinder", {"radius": 0.012, "height": 0.6,
                                           "center": [0, 0, -0.3], "axis": [0, 0, 1],
                                           "sections": 12}))
    return BodyDef(name, MeshSpec("compound", {"parts": parts}), fixed=(name == "hub_base"),
                   initial_transform=_POSE[name], color=(0.82, 0.82, 0.84))


def _joints():
    joints = []
    for k, n in enumerate(NORMALS):
        e1, e2, vertex_angle = _ring_frame(n)
        vertex_at = {round(a / THETA) % N_UNITS: v for v, a in vertex_angle.items()}
        for j in range(N_UNITS):
            p = _POINTS[(k, j)]
            joints.append(JointDef(f"c{k}_u{j}_pivot", "revolute", f"c{k}_u{j}_a", f"c{k}_u{j}_b",
                                   pivot=p["kink"], axis=n,
                                   limit_lo=-PIVOT_LIMIT, limit_hi=PIVOT_LIMIT))
        for j in range(N_UNITS):
            nxt = (j + 1) % N_UNITS
            here, there = _POINTS[(k, j)], _POINTS[(k, nxt)]
            # Unit j's left end pair meets unit j+1's right end pair at
            # connection m = j+1: A1 on B2 (outer), B1 on A2 (inner).
            pairs = (("out", f"c{k}_u{j}_a", here["a1"], f"c{k}_u{nxt}_b", there["b2"]),
                     ("in", f"c{k}_u{j}_b", here["b1"], f"c{k}_u{nxt}_a", there["a2"]))
            for level, body_l, p_l, body_r, p_r in pairs:
                assert np.linalg.norm(p_l - p_r) < 1e-9, (k, j, level)
                m = nxt
                if m in vertex_at:
                    hub = _hub(vertex_at[m], level)
                    joints += [JointDef(f"{hub}_c{k}_u{j}", "revolute", hub, body_l, pivot=p_l, axis=n),
                               JointDef(f"{hub}_c{k}_u{nxt}", "revolute", hub, body_r, pivot=p_l, axis=n)]
                else:
                    joints.append(JointDef(f"c{k}_link{m}_{level}", "revolute", body_l, body_r,
                                           pivot=p_l, axis=n))
    return joints


_BODIES = ([_hub_body(BASE_VERTEX, "out")]
           + [_rod(k, j, rod) for k in range(len(NORMALS)) for j in range(N_UNITS) for rod in ("a", "b")]
           + [_hub_body(v, level) for v in range(len(VERTICES)) for level in ("out", "in")
              if _hub(v, level) != "hub_base"])

CASE = AtlasCaseSpec(
    name="hoberman_sphere",
    mechanism=MechanismSpec(bodies=_BODIES, joints=_joints(), up_axis="z", gravity=0.0),
    coordinates=[],
    contact_model="none",
    actuators=[ActuatorDef("scissor torque", kind="joint_torque", target="c0_u0_pivot", hi=2.0)],
    build=BuildOptions(
        # The motion is one curve of length ~27.4 in the packed 3588-D pose
        # metric, so step 0.2 closes the front at ~140 samples.  One probe
        # projection costs ~0.6 s (sparse LSMR path), ~48 per frontier node,
        # so ~28 s/node and ~65 min in all; the budget leaves 2x headroom.
        # No sample cap: the frontier must close on its own.
        sampling=SamplingOptions(n_samples=None, step=0.2, max_seconds=7200,
                                 checkpoint_seconds=300,
                                 tangent_eps=0.04, tangent_rel_tol=0.20),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
