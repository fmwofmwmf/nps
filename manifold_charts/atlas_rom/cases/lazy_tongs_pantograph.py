"""cases/lazy_tongs_pantograph.py — a three-cell lazy-tongs pantograph with two
skew (Sylvester) gusset pens: six crossed bars, eight pins, five pens, and
still exactly two degrees of freedom.

THE LATTICE.  Every body turns with one of two directions,

    z1 = exp(i * theta1)                 ("rising" bars, family 1)
    z2 = exp(i * (theta1 + theta_rel))   ("falling" bars, family 2)

and every point of the linkage is  X = a * z1 + b * z2  for fixed (a, b).  A
family-1 bar keeps b fixed and spans a; a family-2 bar keeps a fixed and spans
b.  Two bars of opposite family can pin together only at the one lattice point
they share, so the whole assembly is affine in (z1, z2) and there is no circle
intersection anywhere — no elbow branch, same as `pantograph_2d`.

With cell size L, the tongs are (k = 1, 2, 3):

    r_k  (family 1)  b = (k-1) L,  a in [(k-1) L, (k+1) L]
    f_k  (family 2)  a = k L,      b in [(k-2) L, k L]

pinned at

    O     ground - r1   (0, 0)                 the fixed pivot
    X_k   r_k    - f_k  (k L, (k-1) L)         each bar's midpoint: the scissor cross
    U_k   f_k    - r_k+1 (k L, k L)            upper hinges, k = 1, 2
    W_k   r_k    - f_k+1 ((k+1) L, (k-1) L)    lower hinges, k = 1, 2

THE PENS.  U_1, U_2 and the free top of f_3 sit at (L,L), (2L,2L), (3L,3L):
always on the ray through O along z1 + z2, at magnifications 1 : 2 : 3, for
every configuration — a straight pantograph with three nested scales.

Two more pens ride on rigid triangular gussets (Sylvester's skew pantograph).
A gusset on r_2 carries Q1 = (L w, L) and one on f_1 carries Q2 = (L, L / w)
for a fixed complex w, so `Q2 = Q1 / w` identically: Q2 traces a copy of Q1
shrunk by |w| and rotated by -arg(w) about O.  Both identities are asserted
at random configurations below, not just at rest.

THE DRIVE.  A grounded 1:1 spur pinion turns a wheel on r_1 about O, so the
whole tongs spin freely about O.  The scissor is NOT driven at X_1 directly:
the tongs fold FLAT at theta_rel = 0 or pi, where every pin is collinear and
the pin Jacobian loses rank (measured: tangent dim 4), so a scissor that can
make a full turn passes a singular circle twice per turn.  Instead an input
crank on r_1 drives f_1 through a connecting rod -- in r_1's frame a Grashof
crank-rocker four-bar (ground X_1 -> input axle, crank, rod, rocker = f_1 about
X_1).  The input crank turns fully while the scissor only rocks inside
ROCKER_RANGE, far from flat, so the configuration space is a smooth 2-torus:
(crank about O) x (input crank about its axle on r_1).  A crank-rocker has two
assembly modes (mirror images, disconnected); the rest pose fixes one.

DOF (planar, 9 driven bodies, all pins revolute about +z):

    27  =  3 * 9      (6 tong bars, drive pinion, input crank, rod)
   -24  =  12 pins * 2
    -1  =  drive gear law
   ----
     2  =  crank about O + input crank

The count is exact.  Every lattice point is a fixed combination of z1 and z2,
and the four-bar keeps theta_rel inside ROCKER_RANGE, so no closure fails and
no rank drops anywhere on the torus.
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, JointDef, MechanismSpec, MeshSpec, SamplingOptions

Z = [0.0, 0.0, 1.0]
THICK = 0.045
L = 0.32                                  # lattice cell size: bars are 2 L long
N_CELLS = 3
W = 1.45 * cmath.exp(0.75j)               # skew-pen ratio: Q1 / Q2, fixed

THETA1_REST = 0.15                        # crank (r_1) absolute angle at rest
THETA_REL_REST = 0.5 * math.pi            # scissor opening at rest: square cells
STEP = 0.87 * 2.79 / 2.41                 # pantograph_2d step, rescaled (see build)

# Each body gets its own z layer, purely cosmetic: pins are lines along +z, so
# the layer never enters a constraint, but stacked bars read as real scissors.
LAYER_DZ = 0.05


def _z(theta1: float, theta_rel: float) -> tuple[complex, complex]:
    return cmath.exp(1j * theta1), cmath.exp(1j * (theta1 + theta_rel))


def _pt(a: complex, b: complex, theta1=THETA1_REST, theta_rel=THETA_REL_REST,
        z: float = 0.0) -> list[float]:
    z1, z2 = _z(theta1, theta_rel)
    x = a * z1 + b * z2
    return [x.real, x.imag, z]


# ---------------------------------------------------------------- the lattice
# body name -> (family, fixed coordinate, span of the free coordinate, layer)
RISING = {f"r{k}": (1, (k - 1) * L, ((k - 1) * L, (k + 1) * L), 2 * k - 1)
          for k in range(1, N_CELLS + 1)}
FALLING = {f"f{k}": (2, k * L, ((k - 2) * L, k * L), 2 * k)
           for k in range(1, N_CELLS + 1)}
LATTICE = {**RISING, **FALLING}

# pin name -> (body_a, body_b, lattice (a, b))
PINS: dict[str, tuple[str, str, tuple[complex, complex]]] = {
    "O": ("__ground", "r1", (0.0, 0.0))}
for k in range(1, N_CELLS + 1):
    PINS[f"X{k}"] = (f"r{k}", f"f{k}", (k * L, (k - 1) * L))
for k in range(1, N_CELLS):
    PINS[f"U{k}"] = (f"f{k}", f"r{k + 1}", (k * L, k * L))
    PINS[f"W{k}"] = (f"r{k}", f"f{k + 1}", ((k + 1) * L, (k - 1) * L))

# pen name -> (carrying body, lattice (a, b))
PENS = {
    "pen_1": ("r2", (L, L)),
    "pen_2": ("r3", (2 * L, 2 * L)),
    "pen_3": ("f3", (3 * L, 3 * L)),
    "skew_outer": ("r2", (L * W, L)),        # Q1
    "skew_inner": ("f1", (L, L / W)),        # Q2 = Q1 / W
}


def _on_body(body: str, ab: tuple[complex, complex]) -> bool:
    fam, fixed, (lo, hi), _ = LATTICE[body]
    a, b = ab
    own, other = (a, b) if fam == 1 else (b, a)
    return abs(other - fixed) < 1e-12 and lo - 1e-12 <= own.real <= hi + 1e-12


def _forward(theta1: float, theta_rel: float) -> dict[str, np.ndarray]:
    """Every pin and pen position from the two independent joint angles."""
    out = {}
    for name, (_a, _b, ab) in PINS.items():
        out[name] = np.array(_pt(*ab, theta1, theta_rel))
    for name, (_body, ab) in PENS.items():
        out[name] = np.array(_pt(*ab, theta1, theta_rel))
    return out


def _body_transform(body: str, theta1: float, theta_rel: float) -> np.ndarray:
    """(3, 4) map from a body's rest-baked world points to its pose at
    (theta1, theta_rel) — used only by the self-check below."""
    fam, fixed, _span, _layer = LATTICE[body]
    z1r, z2r = _z(THETA1_REST, THETA_REL_REST)
    z1, z2 = _z(theta1, theta_rel)
    if fam == 1:
        rot, t = z1 / z1r, fixed * z2 - (z1 / z1r) * fixed * z2r
    else:
        rot, t = z2 / z2r, fixed * z1 - (z2 / z2r) * fixed * z1r
    c, s = rot.real, rot.imag
    return np.array([[c, -s, 0.0, t.real], [s, c, 0.0, t.imag], [0.0, 0.0, 1.0, 0.0]])


# ------------------------------------------------------------ self-consistency
for _name, (_ba, _bb, _ab) in PINS.items():
    for _body in (_ba, _bb):
        assert _body == "__ground" or _on_body(_body, _ab), (_name, _body)
for _name, (_body, _ab) in PENS.items():
    assert _body in LATTICE, _name
assert 3 * len(LATTICE) - 2 * len(PINS) == 2, "Gruebler count must be exactly 2"

_rng = np.random.default_rng(0)
for _t1, _tr in [(THETA1_REST, THETA_REL_REST), *_rng.uniform(-math.pi, math.pi, (8, 2))]:
    # identities are algebraic, so check them past the stop too
    _f = _forward(_t1, _tr)
    _ray = _f["pen_1"][:2]
    for _k, _pen in ((2, "pen_2"), (3, "pen_3")):
        assert np.allclose(_f[_pen][:2], _k * _ray, atol=1e-12), (_pen, _t1, _tr)
    _q1 = complex(*_f["skew_outer"][:2])
    _q2 = complex(*_f["skew_inner"][:2])
    assert abs(_q2 - _q1 / W) < 1e-12, "skew pens must stay similar about O"
    # every pin is carried consistently by both of its bodies' rigid motions
    for _name, (_ba, _bb, _ab) in PINS.items():
        _rest = np.array(_pt(*_ab) + [1.0])
        for _body in (_ba, _bb):
            if _body != "__ground":
                assert np.allclose(_body_transform(_body, _t1, _tr) @ _rest,
                                   _f[_name], atol=1e-12), (_name, _body)


# --------------------------------------------------------------------- meshes
def _bar(a_b0, a_b1, z, thick=THICK):
    return MeshSpec("link", {"p0": _pt(*a_b0, z=z), "p1": _pt(*a_b1, z=z),
                             "thick": thick})


def _boss(ab, z, radius=0.028, height=0.07):
    return MeshSpec("cylinder", {"radius": radius, "height": height,
                                 "center": _pt(*ab, z=z), "axis": Z, "sections": 16})


def _scissor_bar(body):
    fam, fixed, (lo, hi), layer = LATTICE[body]
    z = layer * LAYER_DZ
    ends = ((lo, fixed), (hi, fixed)) if fam == 1 else ((fixed, lo), (fixed, hi))
    parts = [_bar(*ends, z)]
    for pin_ab in (p[2] for p in PINS.values() if body in p[:2]):
        parts.append(_boss(pin_ab, z, radius=0.022, height=0.06))
    return parts


def _gusset(body, corner_a, corner_b, apex, z):
    """A welded triangle of bars from two points on the bar out to a pen."""
    return [_bar(corner_a, apex, z, 0.035), _bar(corner_b, apex, z, 0.035),
            _boss(apex, z, radius=0.034, height=0.11)]


_parts = {b: _scissor_bar(b) for b in LATTICE}
_parts["r2"] += _gusset("r2", PINS["U1"][2], PINS["X2"][2], PENS["skew_outer"][1],
                        LATTICE["r2"][3] * LAYER_DZ)
_parts["f1"] += _gusset("f1", PINS["X1"][2], PINS["U1"][2], PENS["skew_inner"][1],
                        LATTICE["f1"][3] * LAYER_DZ)
for _pen in ("pen_1", "pen_2", "pen_3"):
    _body, _ab = PENS[_pen]
    _parts[_body].append(_boss(_ab, LATTICE[_body][3] * LAYER_DZ, radius=0.032,
                               height=0.16))

# ---------------------------------------------------------------------- drive
# One external 1:1 spur mesh: a grounded pinion turns a wheel on r1 about O
# (carrier: ground).  A non-integer ratio q/p would make the pose-space
# configuration set p-sheeted (the mesh repeats only every p wheel turns);
# 1:1 keeps the pinion pose a function of the wheel pose.
CRANK_TEETH, DRIVE_TEETH, CRANK_MODULE = 24, 24, 0.012
R_CRANK = 0.5 * CRANK_MODULE * CRANK_TEETH
R_DRIVE = 0.5 * CRANK_MODULE * DRIVE_TEETH
DRIVE_AZIMUTH = -2.2                       # where the drive pinion sits around O
GEAR_W = 0.04
Z_DRIVE = -0.02                            # drive mesh below the tongs

_drive_dir = np.array([math.cos(DRIVE_AZIMUTH), math.sin(DRIVE_AZIMUTH), 0.0])
DRIVE_CENTER = (R_CRANK + R_DRIVE) * _drive_dir

# ----------------------------------------------------- scissor crank-rocker
# In r1's frame: ground link X1 -> input axle (FB_GROUND, along r1), input
# crank FB_CRANK, rod FB_ROD, rocker = f1 about X1 with the rod pinned
# FB_ROCKER along f1.  The rod is sized so the rest opening (pi/2) is reached.
FB_GROUND, FB_CRANK, FB_ROCKER = 0.18, 0.05, 0.16
INPUT_AB = (L + FB_GROUND, 0.0)                  # input axle, on r1
ROCKER_AB = (L, FB_ROCKER)                       # rod pin, on f1
assert _on_body("r1", INPUT_AB) and _on_body("f1", ROCKER_AB)
INPUT_CENTER = np.array(_pt(*INPUT_AB))
ROCKER_PIN = np.array(_pt(*ROCKER_AB))
FB_ROD = float(np.linalg.norm(ROCKER_PIN - INPUT_CENTER))
_links = sorted((FB_GROUND, FB_CRANK, FB_ROD, FB_ROCKER))
assert FB_CRANK == _links[0] and _links[0] + _links[3] < _links[1] + _links[2], \
    "four-bar must be a Grashof crank-rocker (crank shortest)"


def _rocker_extremes():
    """theta_rel at the rocker's two dead centres (crank and rod collinear)."""
    out = []
    for reach in (FB_ROD - FB_CRANK, FB_ROD + FB_CRANK):
        c = (FB_ROCKER ** 2 + FB_GROUND ** 2 - reach ** 2) / (2 * FB_ROCKER * FB_GROUND)
        out.append(math.acos(c))
    return tuple(sorted(out))


ROCKER_RANGE = _rocker_extremes()            # the scissor's whole travel
assert 0.25 < ROCKER_RANGE[0] < THETA_REL_REST < ROCKER_RANGE[1] < math.pi - 0.25


def _crank_pin_rest():
    """Circle(input axle, crank) meets circle(rocker pin, rod); take the
    intersection nearer the input axle side (fixes one assembly mode)."""
    c0, c1 = INPUT_CENTER[:2], ROCKER_PIN[:2]
    d = float(np.linalg.norm(c1 - c0))
    a = (FB_CRANK ** 2 - FB_ROD ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(max(FB_CRANK ** 2 - a ** 2, 0.0))
    e = (c1 - c0) / d
    n = np.array([-e[1], e[0]])
    pin = c0 + a * e + h * n
    return np.array([pin[0], pin[1], 0.0])


CRANK_PIN = _crank_pin_rest()
assert abs(np.linalg.norm(CRANK_PIN - INPUT_CENTER) - FB_CRANK) < 1e-9
assert abs(np.linalg.norm(CRANK_PIN - ROCKER_PIN) - FB_ROD) < 1e-9
Z_INPUT = LATTICE["f1"][3] * LAYER_DZ + 0.06     # input crank above f1
Z_ROD = Z_INPUT + 0.05


def _gear(radius, teeth, center, z, phase):
    return MeshSpec("spur_gear", {"pitch_radius": radius, "teeth": teeth,
                                  "width": GEAR_W, "tooth_phase": phase,
                                  "center": [center[0], center[1], z], "axis": Z})


# tooth of the wheel points at its pinion; the pinion's gap faces it
_parts["r1"].append(_gear(R_CRANK, CRANK_TEETH, (0.0, 0.0), Z_DRIVE, DRIVE_AZIMUTH))
_z_r1, _z_f1 = LATTICE["r1"][3] * LAYER_DZ, LATTICE["f1"][3] * LAYER_DZ
_parts["r1"].append(_boss(INPUT_AB, 0.5 * (_z_r1 + Z_INPUT), radius=0.02,
                          height=Z_INPUT - _z_r1))             # input axle post
_parts["f1"].append(_boss(ROCKER_AB, 0.5 * (_z_f1 + Z_ROD), radius=0.02,
                          height=Z_ROD - _z_f1))               # rod pin post
_drive_parts = [
    _gear(R_DRIVE, DRIVE_TEETH, DRIVE_CENTER, Z_DRIVE,
          DRIVE_AZIMUTH + math.pi + math.pi / DRIVE_TEETH),
    MeshSpec("cylinder", {"radius": 0.025, "height": 0.12,
                          "center": [DRIVE_CENTER[0], DRIVE_CENTER[1], Z_DRIVE + 0.04],
                          "axis": Z, "sections": 16}),          # input hub
]
_input_parts = [
    _gear(0.075, 18, INPUT_CENTER, Z_INPUT, 0.0),             # crank disc
    MeshSpec("cylinder", {"radius": 0.016, "height": 0.08,
                          "center": [CRANK_PIN[0], CRANK_PIN[1], Z_INPUT + 0.03],
                          "axis": Z, "sections": 16}),          # crank pin
]
_rod_parts = [
    MeshSpec("link", {"p0": [CRANK_PIN[0], CRANK_PIN[1], Z_ROD],
                      "p1": [ROCKER_PIN[0], ROCKER_PIN[1], Z_ROD], "thick": 0.03}),
]

_ground_parts = [
    MeshSpec("box", {"center": [0.0, 0.0, -0.09], "half": [0.16, 0.16, 0.02]}),
    MeshSpec("box", {"center": [DRIVE_CENTER[0], DRIVE_CENTER[1], -0.09],
                     "half": [0.07, 0.07, 0.02]}),
    MeshSpec("cylinder", {"radius": 0.04, "height": 0.05, "center": [0.0, 0.0, -0.065],
                          "axis": Z, "sections": 20}),
    MeshSpec("cylinder", {"radius": 0.018, "height": 0.07,
                          "center": [DRIVE_CENTER[0], DRIVE_CENTER[1], -0.06],
                          "axis": Z, "sections": 16}),
]

STEEL = (0.62, 0.64, 0.68)
BRASS = (0.86, 0.68, 0.30)
COLORS = {"r1": (0.93, 0.53, 0.18), "r2": BRASS, "r3": (0.93, 0.53, 0.18),
          "f1": (0.36, 0.62, 0.86), "f2": STEEL, "f3": (0.36, 0.62, 0.86)}

# 9 bodies, 12 pins, 1 gear law: 27 - 24 - 1 = 2
assert 3 * (len(LATTICE) + 3) - 2 * (len(PINS) + 4) - 1 == 2


CASE = AtlasCaseSpec(
    name="lazy_tongs_pantograph",
    mechanism=MechanismSpec(
        bodies=[BodyDef("__ground", MeshSpec("compound", {"parts": _ground_parts}),
                        fixed=True, color=(0.40, 0.42, 0.45))]
        + [BodyDef(b, MeshSpec("compound", {"parts": _parts[b]}), color=COLORS[b])
           for b in ("r1", "f1", "r2", "f2", "r3", "f3")]
        + [BodyDef("drive_pinion", MeshSpec("compound", {"parts": _drive_parts}),
                   color=STEEL),
           BodyDef("input_crank", MeshSpec("compound", {"parts": _input_parts}),
                   color=BRASS),
           BodyDef("rod", MeshSpec("compound", {"parts": _rod_parts}), color=STEEL)],
        joints=[JointDef(name, "revolute", ba, bb, pivot=_pt(*ab), axis=Z)
                for name, (ba, bb, ab) in PINS.items()]
        + [JointDef("drive_axle", "revolute", "__ground", "drive_pinion",
                    pivot=DRIVE_CENTER, axis=Z),
           JointDef("input_axle", "revolute", "r1", "input_crank",
                    pivot=INPUT_CENTER, axis=Z),
           JointDef("crank_pin", "revolute", "input_crank", "rod",
                    pivot=CRANK_PIN, axis=Z),
           JointDef("rocker_pin", "revolute", "rod", "f1",
                    pivot=ROCKER_PIN, axis=Z),
           # phi_pinion = -ratio * phi_wheel, both relative to the carrier
           JointDef("drive_mesh", "gear_ratio", "drive_pinion", "r1",
                    body_c="__ground", pivot=DRIVE_CENTER, axis=Z,
                    ratio=CRANK_TEETH / DRIVE_TEETH)],
        up_axis="y",                     # planar linkage in xy: gravity is in-plane
        render_up_axis="z",
        gravity=9.8,
    ),
    contact_model="none",                # ideal pins and gear laws; no tooth contact
    coordinates=[
        CoordinateDef("crank_angle", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "O"}),
        CoordinateDef("input_angle", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "input_axle"}),
    ],
    actuators=[
        ActuatorDef("drive torque", kind="joint_torque", target="drive_axle",
                    hi=60.0 * DRIVE_TEETH / CRANK_TEETH),
        ActuatorDef("input torque", kind="joint_torque", target="input_axle", hi=10.0),
    ],
    build=BuildOptions(
        # Same treatment as pantograph_2d; the fold pockets that needed
        # max_probes_per_node there never occur here (the four-bar keeps the
        # scissor inside ROCKER_RANGE); it is kept as a guard.  `step`
        # is pantograph_2d's 0.87 rescaled by the ratio of geometric-mean
        # pose-Jacobian singular values (2.79 here vs 2.41 there, measured
        # over random in-range poses by FD through `_body_transform`).
        sampling=SamplingOptions(n_samples=1600, step=STEP,
                                 max_seconds=900, max_probes_per_node=16,
                                 tangent_eps=0.04, tangent_rel_tol=0.20),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
