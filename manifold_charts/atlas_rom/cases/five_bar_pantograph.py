"""cases/five_bar_pantograph.py — a five-bar whose two distal links ARE the bottom
cell of a scissor (lazy-tongs) lift.

THE FIVE-BAR.  Two grounded pivots A, B (positions and link length from
`five_bar`) drive a1, a2.  The distal links b1 = C-E and b2 = D-E close the loop
at E.  Here each is stretched through E to twice its length, so E is the MIDPOINT
of both: b1 = r1 and b2 = f1 are the first crossed pair of a lazy-tongs chain,
and E is its cross pin X_1.

THE SCISSOR.  `lazy_tongs_pantograph`'s lattice (read its docstring), grown from
that first pair.  With cell size L = |CE| = |DE| every pin is

    X = C + a z1 + b z2,    z1 = (E - C) / L,    z2 = (E - D) / L,

so C (r1's end) is the lattice origin and the rest of the chain, cells 2..N,
hangs off r1 and f1 at U_k = C + k L (z1 + z2) and W_k.  The lattice is affine in
(z1, z2), so the extra cells close with no circle intersection and add no DOF.

DOF (planar, 2 + 2N driven bodies, all pins revolute about +z):

    3 (2 + 2N)                  a1, a2, r_k, f_k
   -2 (2 + 2 + N + 2 (N-1))     A, B, C, D, X_k, U_k, W_k
   --------
     2                          asserted below for any N

The tongs lift is the five-bar's own distal pair, so the only freedoms are the
two drivers; the chain is slaved to (C, D).

THE PENS.  U_1 and the free top of f_N lie on the ray from C along z1 + z2, at
magnifications 1 : N, for every configuration.  The centre of that homothety is
C, which rides on a1; it is not a fixed pivot.

NO STOPS.  No joint limit or workspace bound is declared.  The scissor still
folds flat (z1 parallel to z2) and the five-bar still has its own singular
poses; the pipeline meets them as it finds them.  The rest pose is the symmetric
one with square cells (opening pi/2), not `five_bar`'s nearly flat rest.

No coordinates or chart structure are declared: like `five_bar`, the pipeline
discovers the (branched) configuration set from simulator frames.
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from atlas_rom.cases import five_bar as _fb
from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, ChartOptions, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

Z = [0.0, 0.0, 1.0]
A, B = _fb.A, _fb.B
THICK = 0.10                              # five-bar driver width
SC_THICK = 0.06                           # scissor bar width

N_CELLS = 3
LINK = 1.5                                # five_bar's la = lb
LA = LINK                                 # driver length
L = LINK                                  # cell size: scissor bars are 2 L long
MAGNIFICATION = float(N_CELLS)            # |tip - C| / |U_1 - C|, any configuration
assert abs(np.linalg.norm(np.array(_fb.C) - np.array(A)) - LA) < 1e-4
assert abs(np.linalg.norm(np.array(_fb.E) - np.array(_fb.C)) - L) < 1e-4

# ----------------------------------------------------------------- rest pose
# Symmetric about x = 0 with square cells: z1 = (sin phi, cos phi), z2 = (-sin phi,
# cos phi), opening pi - 2 phi = pi/2.  C + L z1 = E puts E on the axis.
PHI = 0.25 * math.pi
_cx = -L * math.sin(PHI)
_cy = math.sqrt(LA ** 2 - (_cx - A[0]) ** 2)
C_REST = complex(_cx, _cy)
D_REST = complex(-_cx, _cy)
E_REST = C_REST + L * complex(math.sin(PHI), math.cos(PHI))
Z1R = (E_REST - C_REST) / L
Z2R = (E_REST - D_REST) / L
assert abs(abs(Z1R) - 1.0) < 1e-12 and abs(abs(Z2R) - 1.0) < 1e-12
assert abs(abs(D_REST - complex(B[0], B[1])) - LA) < 1e-12

Z0 = 0.0
LAYER_DZ = 0.05                           # cosmetic z layer per bar; never in a constraint


def _pt(a: complex, b: complex, z: float = 0.0) -> list[float]:
    x = C_REST + a * Z1R + b * Z2R
    return [x.real, x.imag, z]


# ---------------------------------------------------------------- the lattice
# body name -> (family, fixed coordinate, span of the free coordinate, layer)
RISING = {f"r{k}": (1, (k - 1) * L, ((k - 1) * L, (k + 1) * L), 2 * k - 1)
          for k in range(1, N_CELLS + 1)}
FALLING = {f"f{k}": (2, k * L, ((k - 2) * L, k * L), 2 * k)
           for k in range(1, N_CELLS + 1)}
LATTICE = {**RISING, **FALLING}
BODY_ORDER = [f"{fam}{k}" for k in range(1, N_CELLS + 1) for fam in ("r", "f")]

# pin name -> (body_a, body_b, lattice (a, b)); C and D join the drivers to the
# first pair, and X_1 is the five-bar's E.
PINS: dict[str, tuple[str, str, tuple[complex, complex]]] = {
    "C": ("a1", "r1", (0.0, 0.0)),
    "D": ("a2", "f1", (L, -L))}
for k in range(1, N_CELLS + 1):
    PINS[f"X{k}"] = (f"r{k}", f"f{k}", (k * L, (k - 1) * L))
for k in range(1, N_CELLS):
    PINS[f"U{k}"] = (f"f{k}", f"r{k + 1}", (k * L, k * L))
    PINS[f"W{k}"] = (f"r{k}", f"f{k + 1}", ((k + 1) * L, (k - 1) * L))

# the pens: close = U_1 on f1, far = the free top of f_N
CLOSE = ("f1", (L, L))
TIP = (f"f{N_CELLS}", (N_CELLS * L, N_CELLS * L))


def _on_body(body: str, ab: tuple[complex, complex]) -> bool:
    fam, fixed, (lo, hi), _ = LATTICE[body]
    a, b = ab
    own, other = (a, b) if fam == 1 else (b, a)
    return abs(other - fixed) < 1e-12 and lo - 1e-12 <= own.real <= hi + 1e-12


# ------------------------------------------------------------ self-consistency
for _name, (_ba, _bb, _ab) in PINS.items():
    assert _on_body(_bb, _ab), (_name, _bb)
    assert _ba in LATTICE and _on_body(_ba, _ab) or _ba in ("a1", "a2"), (_name, _ba)
assert _on_body(*CLOSE) and _on_body(*TIP)

assert (3 * (2 + len(LATTICE)) - 2 * (2 + len(PINS))) == 2, "Gruebler count must be exactly 2"

assert np.allclose(_pt(*PINS["X1"][2])[:2], [E_REST.real, E_REST.imag], atol=1e-12)
assert np.allclose(_pt(*PINS["C"][2])[:2], [C_REST.real, C_REST.imag], atol=1e-12)
assert np.allclose(_pt(*PINS["D"][2])[:2], [D_REST.real, D_REST.imag], atol=1e-12)
assert np.allclose(np.array(_pt(*TIP[1]))[:2] - [C_REST.real, C_REST.imag],
                   MAGNIFICATION * (np.array(_pt(*CLOSE[1]))[:2] - [C_REST.real, C_REST.imag]),
                   atol=1e-12)


# --------------------------------------------------------------------- meshes
def _bar(a_b0, a_b1, z, thick=SC_THICK):
    return MeshSpec("link", {"p0": _pt(*a_b0, z=z), "p1": _pt(*a_b1, z=z),
                             "thick": thick})


def _boss(ab, z, radius=0.035, height=0.08):
    return MeshSpec("cylinder", {"radius": radius, "height": height,
                                 "center": _pt(*ab, z=z), "axis": Z, "sections": 16})


def _layer_z(body):
    return Z0 + LATTICE[body][3] * LAYER_DZ


def _scissor_bar(body):
    fam, fixed, (lo, hi), _layer = LATTICE[body]
    z = _layer_z(body)
    ends = ((lo, fixed), (hi, fixed)) if fam == 1 else ((fixed, lo), (fixed, hi))
    parts = [_bar(*ends, z)]
    for pin_ab in (p[2] for p in PINS.values() if body in p[:2]):
        parts.append(_boss(pin_ab, z))
    return parts


_parts = {b: _scissor_bar(b) for b in LATTICE}
_tip_body, _tip_ab = TIP
_parts[_tip_body].append(_boss(_tip_ab, _layer_z(_tip_body), radius=0.07, height=0.2))
_close_body, _close_ab = CLOSE
_parts[_close_body].append(_boss(_close_ab, _layer_z(_close_body), radius=0.055, height=0.16))

_ground_parts = [
    MeshSpec("link", {"p0": [A[0] - 0.25, A[1], 0.0], "p1": [B[0] + 0.25, B[1], 0.0],
                      "thick": 0.13}),
]

PALETTE = [(0.93, 0.53, 0.18), (0.36, 0.62, 0.86), (0.86, 0.68, 0.30), (0.62, 0.64, 0.68)]
# rising bars warm, falling bars cool; odd cells saturated, even cells muted
COLORS = {b: PALETTE[(0 if b[0] == "r" else 1) + (0 if int(b[1:]) % 2 else 2)]
          for b in BODY_ORDER}


def _driver(name, pivot, tip, color):
    return BodyDef(name=name, mesh=MeshSpec("link", {
        "p0": [pivot[0], pivot[1], 0.0], "p1": [tip.real, tip.imag, 0.0],
        "thick": THICK}), color=color)


CASE = AtlasCaseSpec(
    name="five_bar_pantograph",
    mechanism=MechanismSpec(
        bodies=[
            _driver("a1", A, C_REST, (0.93, 0.53, 0.18)),      # driver, left
            _driver("a2", B, D_REST, (0.93, 0.53, 0.18)),      # driver, right
            BodyDef("__ground", MeshSpec("compound", {"parts": _ground_parts}),
                    fixed=True, color=(0.45, 0.47, 0.50)),
        ] + [BodyDef(b, MeshSpec("compound", {"parts": _parts[b]}), color=COLORS[b])
             for b in BODY_ORDER],
        joints=[
            JointDef("A", "revolute", "__ground", "a1", pivot=A, axis=Z),
            JointDef("B", "revolute", "__ground", "a2", pivot=B, axis=Z),
        ] + [JointDef(name, "revolute", ba, bb, pivot=_pt(*ab), axis=Z)
             for name, (ba, bb, ab) in PINS.items()],
        up_axis="y",                     # planar linkage in xy: gravity is in-plane
        render_up_axis="z",
        gravity=9.8,
    ),
    seeds=[],
    contact_model="none",                # ideal pins; the manifold is the loops
    coordinates=[],                      # discovered, as in five_bar: none declared
    actuators=[
        ActuatorDef("driver A torque", kind="joint_torque", target="A", hi=40.0),
        ActuatorDef("driver B torque", kind="joint_torque", target="B", hi=40.0),
    ],
    build=BuildOptions(
        # `step` is a pose-space distance (12 numbers per driven body).  five_bar's
        # 1.0 is rescaled so the expected node count (pose-space area / step^2)
        # matches five_bar's: area = integral of sqrt(det J^T J) over the driver
        # angles (alpha_A, alpha_B), elbow-up branch only, 40000 uniform draws:
        # 313 for five_bar, 1707 here, so step = sqrt(1707 / 313) = 2.33.
        sampling=SamplingOptions(
            n_samples=None, step=2.33, max_seconds=1800,
            representation="planar_xytheta", graph_k=32, max_probes_per_node=16),
        charts=ChartOptions(mode="mesh_atlas", overlap=0.25),
        canonical=CanonicalOptions(enabled=True, planarity=0.2,
                                   residual_discovery=True,
                                   autoencoder_charts=True),
        decoder=DecoderOptions(epochs=6000, smooth_lambda=1e-3,
                               closure_weight=1.0),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=1500),
        expect=Expectations(n_bodies=2 + len(LATTICE)),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2),
    ),
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "exact_canonical"}},
)


# Display / task-space hook for ``viewers.rom_target``: pen name -> (body, rest-world local point).  The FIRST pen
# is the draggable one: the close pen U_1.  There is no fixed pivot, so no reach circles.
PEN_TARGET = {
    "pens": {"close": (_close_body, _pt(*_close_ab)),
             "tip": (_tip_body, _pt(*_tip_ab))},
    "circles": [],
    "torque_limit": 80.0,
}
