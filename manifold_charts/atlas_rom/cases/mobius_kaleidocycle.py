"""cases/mobius_kaleidocycle.py — a closed ring of M tetrahedra, hinged edge to
edge, whose rim vertices sit on the boundary curve of a Mobius strip instead of
the two parallel circles an ordinary (orientable) kaleidocycle uses.

Rim construction
-----------------
The classic kaleidocycle takes 2N rim points alternating between a top and a
bottom circle (an antiprism) and builds tetrahedron k from four CONSECUTIVE rim
points, so T_k and T_{k+1} share exactly the edge between rim points k+1 and
k+2 — a true hinge (2 shared vertices), not a shared face.  That antiprism rim
closes after one trip around the axis and is topologically an annulus.

Here the rim instead samples the well-known single boundary curve of a Mobius
strip (half-width w, centre radius R):

    x(t) = (R + w*cos(t/2)) * cos(t)
    y(t) = (R + w*cos(t/2)) * sin(t)
    z(t) = w * sin(t/2)

which has period 4*pi, not 2*pi — the curve must be traced TWICE around the
centre circle to return to its start, the discrete signature of the strip's
one half-twist.  Sampling M rim points at t_j = j * 4*pi/M for j = 0..M-1 and
building tetrahedra the same "4 consecutive rim points" way as the classic
construction gives a ring of M tetrahedra with the same edge-hinge adjacency,
but carried around a non-orientable rim instead of an orientable one.

What this case does NOT claim
------------------------------
The classic ring-of-6-isogonal-tetrahedra kaleidocycle is a proven 1-DOF
mechanism; that proof depends on a special (isogonal) choice of tetrahedron
shape, not merely "hinge-adjacent". This rim's tetrahedra are whatever shape
falls out of the Mobius curve sample above — there is no proof here that the
closed loop of M revolute joints has any finite mobility at all.  Per
`finite-vs-instantaneous-mobility`, passing a rank test is not evidence of
finite travel.  Whether this specific twisted ring flexes, and through how
many degrees of freedom, is exactly what the sampling/canonical-discovery
stage below is for — nothing is declared, per `mesh-contact-6dof-default` and
`no-per-case-cheating`: the only constraints assumed are the M revolute hinges
this file states, and one tetrahedron (T0) held fixed in the hand, the same
way a person actually flexes the physical toy.
"""
from __future__ import annotations

import colorsys

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, ChartOptions, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

# ─── rim geometry ─────────────────────────────────────────────────────────────

M = 12                          # tetrahedra in the ring == rim points
R = 1.0                         # centre-circle radius
W = 0.35                        # Mobius strip half-width (rim wobble amplitude)
EDGE_RADIUS = 0.05

_t = np.arange(M) * (4.0 * np.pi / M)
_RIM = np.stack([(R + W * np.cos(_t / 2.0)) * np.cos(_t),
                 (R + W * np.cos(_t / 2.0)) * np.sin(_t),
                 W * np.sin(_t / 2.0)], axis=1)


def _bar_mesh(p0, p1, radius=EDGE_RADIUS):
    """A cylindrical edge p0->p1; unlike `MeshSpec("link", ...)` this survives
    leaving the xy-plane, which every tetrahedron edge here does."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    return MeshSpec("cylinder", {"radius": radius,
                                 "height": float(np.linalg.norm(d)),
                                 "center": (0.5 * (p0 + p1)).tolist(),
                                 "axis": d.tolist(), "sections": 12})


def _tetra_mesh(pts):
    """Wireframe tetrahedron: a compound of its 6 edges."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return MeshSpec("compound", {"parts": [_bar_mesh(pts[i], pts[j]) for i, j in pairs]})


def _color(k):
    r, g, b = colorsys.hsv_to_rgb((k / M) % 1.0, 0.55, 0.85)
    return (r, g, b)


_bodies = []
_joints = []
for k in range(M):
    pts = [_RIM[(k + off) % M] for off in range(4)]
    _bodies.append(BodyDef(name=f"T{k}", mesh=_tetra_mesh(pts), fixed=(k == 0),
                           color=_color(k)))
for k in range(M):
    p_hi, p_lo = _RIM[(k + 1) % M], _RIM[(k + 2) % M]
    axis = p_lo - p_hi
    _joints.append(JointDef(f"hinge_{k}", "revolute", f"T{k}", f"T{(k + 1) % M}",
                            pivot=0.5 * (p_hi + p_lo), axis=axis))

CASE = AtlasCaseSpec(
    name="mobius_kaleidocycle",
    mechanism=MechanismSpec(
        bodies=_bodies,
        joints=_joints,
        up_axis="z",
        gravity=9.8,
    ),
    seeds=[],
    contact_model="none",            # ideal hinges; adjacent tetrahedra never touch
    # No coordinate can be declared: the manifold's dimension (if it has finite
    # mobility at all) is discovered, not assumed. See module docstring.
    coordinates=[],
    actuators=[
        ActuatorDef("hinge_0 torque", kind="joint_torque", target="hinge_0", hi=20.0),
    ],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=8000, step=0.05, max_seconds=3600,
                                 tangent_rel_tol=0.08, graph_k=24),
        charts=ChartOptions(mode="mesh_atlas", overlap=0.25),
        canonical=CanonicalOptions(enabled=True, dimension=2, planarity=0.2,
                                   residual_discovery=True,
                                   autoencoder_charts=True),
        decoder=DecoderOptions(epochs=6000, smooth_lambda=1e-3, closure_weight=1.0),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=1500),
        expect=Expectations(n_bodies=M - 1),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2),
    ),
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "exact_canonical"},
               "exact": {"dt": 1e-3, "dynamics_backend": "exact_canonical"},
               "distilled": {"dt": 1e-3, "dynamics_backend": "distilled_canonical"},
               "direct": {"dt": 1e-3, "dynamics_backend": "direct_constrained"}},
)
