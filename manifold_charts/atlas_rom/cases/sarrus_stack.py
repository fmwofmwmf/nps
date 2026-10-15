"""cases/sarrus_stack.py — two stacked Sarrus linkages, all-revolute, no contact.

A Sarrus linkage is the classical proof that a straight prismatic slide can be
built entirely from revolute joints, and it is a genuinely SPATIAL 6R closed
loop, not two independent legs: base plate -> link1a -> link1b -> top plate
(three hinges, all parallel to +x) closes back base plate -> link2a -> link2b
-> top plate (three hinges, all parallel to +y).  Each 3-hinge chain alone
would let the top plate translate AND rotate within its own plane; putting
the two perpendicular chains in the same closed loop cancels every rotation
and leaves exactly one freedom, pure +z translation, verified below by the
Chebychev-Grubler count landing on 0 (an overconstrained mechanism) while the
measured constraint-Jacobian nullity is 1 — the general lesson of
`three_rrs.py`, that only a rank/finite-travel measurement (not the naive
count) tells the truth, applies here too, just in the direction the naive
count is too PESSIMISTIC rather than too optimistic.

Each chain is a symmetric elbow: two equal-length links, pinned to the plate
above and below at points that share x AND y (only z, the stage height,
differs) so the top plate never drifts sideways; the elbow point swings out
and back as the shared angle theta closes.  `H(theta) = 2*Lh*cos(theta)` for
both chains by construction, which is what forces their hinge angles to move
together as the single shared coordinate.

This case stacks TWO such units — ground -> mid plate -> top plate — for a
2-D coordinate space, one elbow angle per stage.  All joints are ideal and
nothing is meant to touch, so the build runs on the fast analytic
`ideal_oracle` path (verified 75x over driving libuipc on `three_rrs`).
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, CanonicalOptions, ChartOptions, CoordinateDef, DecoderOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

# ─── geometry ───────────────────────────────────────────────────────────────

X0 = 0.35                      # chain-1 anchor offset on every plate (+x)
Y0 = 0.35                      # chain-2 anchor offset on every plate (+y)
LH = 0.60                      # link length, both links, both chains, both stages
THETA0 = math.radians(45.0)    # home elbow angle from vertical
STAGE_HEIGHT = 2.0 * LH * math.cos(THETA0)

# Physical travel limits, measured from the built/rest pose.  The outer
# hinges follow the stage coordinate; the elbow hinges sweep twice as far.
MIN_DELTA = math.radians(12.0) - THETA0
MAX_DELTA = math.radians(78.0) - THETA0

LINK_RADIUS = 0.035
HUB_RADIUS = 0.035
_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])


def _bar_mesh(p0, p1, radius):
    """A cylindrical link p0->p1; unlike `MeshSpec("link", ...)` this does not
    flatten bars that leave the xy-plane, which every link here does."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    return MeshSpec("cylinder", {"radius": radius,
                                 "height": float(np.linalg.norm(d)),
                                 "center": (0.5 * (p0 + p1)).tolist(),
                                 "axis": d.tolist(), "sections": 16})


def _plate_mesh(anchors):
    """A plate: a compound bar joining its two chain anchor points (plus a
    stub through the centre so the body reads as more than a single edge)."""
    p0, p1 = anchors
    mid = 0.5 * (p0 + p1)
    return MeshSpec("compound", {"parts": [
        _bar_mesh(p0, p1, HUB_RADIUS),
        _bar_mesh(mid - np.array([0.0, 0.0, 0.08]),
                  mid + np.array([0.0, 0.0, 0.08]), HUB_RADIUS)]})


def _elbow(anchor, out_axis, theta):
    """Elbow point of a symmetric 2-link chain rooted at `anchor`, bulging
    along `out_axis` (the in-chain-plane direction perpendicular to the
    chain's hinge axis and to +z)."""
    return anchor + LH * math.sin(theta) * out_axis + np.array([0.0, 0.0, LH * math.cos(theta)])


def _sarrus_stage(prefix, base_body, base_p0, base_q0, top_body, theta):
    """One Sarrus unit's 4 links + 6 hinges between two named plates.

    `base_p0`/`base_q0` are the chain-1/chain-2 anchors on the base plate;
    the corresponding top-plate anchors are directly above them by
    `STAGE_HEIGHT`, by construction (see module docstring).
    """
    top_p0 = base_p0 + np.array([0.0, 0.0, STAGE_HEIGHT])
    top_q0 = base_q0 + np.array([0.0, 0.0, STAGE_HEIGHT])
    elbow1 = _elbow(base_p0, -_Y, theta)    # chain 1: hinge axis +x, bulges outward -y
    elbow2 = _elbow(base_q0, -_X, theta)    # chain 2: hinge axis +y, bulges outward -x

    bodies = [
        BodyDef(f"{prefix}_link1a", _bar_mesh(base_p0, elbow1, LINK_RADIUS),
                color=(0.93, 0.53, 0.18)),
        BodyDef(f"{prefix}_link1b", _bar_mesh(elbow1, top_p0, LINK_RADIUS),
                color=(0.93, 0.53, 0.18)),
        BodyDef(f"{prefix}_link2a", _bar_mesh(base_q0, elbow2, LINK_RADIUS),
                color=(0.36, 0.62, 0.86)),
        BodyDef(f"{prefix}_link2b", _bar_mesh(elbow2, top_q0, LINK_RADIUS),
                color=(0.36, 0.62, 0.86)),
    ]
    joints = [
        JointDef(f"{prefix}_h1", "revolute", base_body, f"{prefix}_link1a",
                 pivot=base_p0, axis=_X, limit_lo=MIN_DELTA, limit_hi=MAX_DELTA),
        JointDef(f"{prefix}_h2", "revolute", f"{prefix}_link1a", f"{prefix}_link1b",
                 pivot=elbow1, axis=_X, limit_lo=2.0 * MIN_DELTA, limit_hi=2.0 * MAX_DELTA),
        JointDef(f"{prefix}_h3", "revolute", f"{prefix}_link1b", top_body,
                 pivot=top_p0, axis=_X, limit_lo=MIN_DELTA, limit_hi=MAX_DELTA),
        JointDef(f"{prefix}_h4", "revolute", base_body, f"{prefix}_link2a",
                 pivot=base_q0, axis=_Y, limit_lo=MIN_DELTA, limit_hi=MAX_DELTA),
        JointDef(f"{prefix}_h5", "revolute", f"{prefix}_link2a", f"{prefix}_link2b",
                 pivot=elbow2, axis=_Y, limit_lo=2.0 * MIN_DELTA, limit_hi=2.0 * MAX_DELTA),
        JointDef(f"{prefix}_h6", "revolute", f"{prefix}_link2b", top_body,
                 pivot=top_q0, axis=_Y, limit_lo=MIN_DELTA, limit_hi=MAX_DELTA),
    ]
    return bodies, joints, top_p0, top_q0


_GROUND_P0 = np.array([X0, 0.0, 0.0])
_GROUND_Q0 = np.array([0.0, Y0, 0.0])

_ground = BodyDef("ground", _plate_mesh((_GROUND_P0, _GROUND_Q0)),
                  fixed=True, color=(0.45, 0.47, 0.50))

_stage1_bodies, _stage1_joints, _mid_p0, _mid_q0 = _sarrus_stage(
    "stage1", "ground", _GROUND_P0, _GROUND_Q0, "mid_plate", THETA0)
_mid = BodyDef("mid_plate", _plate_mesh((_mid_p0, _mid_q0)),
              color=(0.35, 0.72, 0.45))

_stage2_bodies, _stage2_joints, _top_p0, _top_q0 = _sarrus_stage(
    "stage2", "mid_plate", _mid_p0, _mid_q0, "top_plate", THETA0)
_top = BodyDef("top_plate", _plate_mesh((_top_p0, _top_q0)),
              color=(0.80, 0.45, 0.75))

MECHANISM = MechanismSpec(
    bodies=[_ground] + _stage1_bodies + [_mid] + _stage2_bodies + [_top],
    joints=_stage1_joints + _stage2_joints,
    up_axis="z", gravity=9.8,
)

# `joint_angle` is measured from the joint's REST travel, which is exactly 0
# at THETA0 (every body's mesh is baked at its built pose).  So the bound is a
# DELTA around 0, not an absolute elbow angle: kept away from both
# singularities — theta -> 0 fully extends the elbow (links collinear with
# the stage axis, H saturates at 2*LH), theta -> pi/2 fully collapses it
# (H -> 0, links horizontal, plates meet).
CASE = AtlasCaseSpec(
    name="sarrus_stack",
    mechanism=MECHANISM,
    contact_model="none",
    coordinates=[
        CoordinateDef("stage1_angle", topology="euclidean", source="joint_angle",
                      args={"joint": "stage1_h1"}),
        CoordinateDef("stage2_angle", topology="euclidean", source="joint_angle",
                      args={"joint": "stage2_h1"}),
    ],
    actuators=[
        # Increasing either elbow angle lowers its platform; the negative gain
        # makes a positive control input mean "up" for the user-facing vector.
        ActuatorDef("middle_platform_up", kind="joint_torque",
                    target="stage1_h1", hi=50.0, gain=-1.0),
        ActuatorDef("top_platform_up", kind="joint_torque",
                    target="stage2_h1", hi=50.0, gain=-1.0),
    ],
    build=BuildOptions(
        # step=0.12 filled the 2-D bounded domain at 12991 samples (22 min,
        # well under max_seconds, so that run was coverage-limited not
        # time-limited); front density scales as 1/step^2 for an area fill,
        # so step*sqrt(12991/1000) ~= 0.43 targets ~1000 samples for the
        # whole space.
        sampling=SamplingOptions(n_samples=2000, step=0.43, max_seconds=3600,
                                 tangent_rel_tol=0.10),
        charts=ChartOptions(mode="single"),
        canonical=CanonicalOptions(autoencoder_charts=True),
        # Two closed 6R loops: a per-body pose loss alone tears them, same
        # reasoning as `scissor_lift_stack` and `five_bar`.
        decoder=DecoderOptions(epochs=5000, smooth_lambda=1e-3,
                               closure_weight=1.0),
        boundary=BoundaryOptions(kind="sdf", k=1e7, margin=0.0,
                                 epochs=3000, width=64, layers=3),
        expect=Expectations(dimension=2, n_bodies=10),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2)),
    scenarios={"default": {"dt": 1e-3, "initial_u": [0.0, 0.0],
                           "boundary_solver": "damped"}},
)


def VIEW_EMBED(q):
    """Display-only 2-D coordinate for ``view_sample_set``: (stage-1 plate travel
    relative to ground, stage-2 plate travel relative to stage 1).  Each Sarrus
    stage forces its plate to translate along z, so these are the mechanism's
    two DOF directly.  Never passed to the fitter; recomputed from ``q`` alone.
    """
    import numpy as np

    bodies = CASE.mechanism.driven_bodies
    order = [b.name for b in bodies]
    T = np.asarray(q, float).reshape(len(q), len(order), 3, 4)
    z0 = {b.name: np.asarray(b.initial_transform, float).reshape(4, 4)[2, 3] for b in bodies}
    mid, top = order.index("mid_plate"), order.index("top_plate")
    d1 = T[:, mid, 2, 3] - z0["mid_plate"]
    d2 = (T[:, top, 2, 3] - T[:, mid, 2, 3]) - (z0["top_plate"] - z0["mid_plate"])
    return np.column_stack((d1, d2))
