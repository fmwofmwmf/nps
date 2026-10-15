"""Diagnostic eight-cross stacked scissor mechanism (not an atlas case)."""
from __future__ import annotations
import math
from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, CoordinateDef, DecoderOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances, WorkspaceBound

N = 8
L = 1.0
HALF = 0.3
H = math.sqrt(L * L - (2 * HALF) ** 2)
Z = [0, 0, 1]
X = [1, 0, 0]
TH0 = math.atan2(H, 2 * HALF)
TH1 = math.pi - TH0
# A scissor stage is singular at both ends of its travel: at pi/2 every leg is
# vertical (the stack gains an instantaneous DOF and the stiff-joint solve blows
# up), and at 0 the legs are collinear and the stack self-intersects.  Chart the
# regular stratum between them, expressed as a minimum stage height the same way
# the single-stage case declares one.
MIN_STAGE_HEIGHT = 0.30
MIN_ANGLE = math.asin(MIN_STAGE_HEIGHT / L)
MAX_ANGLE = 1.43


def tr(x, y, z=0):
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


def pose(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]


def link(name, p0, p1, transform, color):
    return BodyDef(name, MeshSpec("link", {"p0": p0, "p1": p1,
                     "thick": 0.05}), configuration="planar_rigid",
                     initial_transform=transform, color=color)


def box(name, transform, color=(0.7, 0.7, 0.7), fixed=False):
    return BodyDef(name, MeshSpec("box", {"center": [0, 0, 0],
                     "half": [0.03, 0.03, 0.03]}),
                   configuration="fixed" if fixed else "planar_rigid",
                   initial_transform=transform, color=color, fixed=fixed)


bodies = [BodyDef("ground", MeshSpec("link", {"p0": [-HALF, 0, 0],
                    "p1": [HALF, 0, 0], "thick": 0.06}), fixed=True,
                    initial_transform=tr(0, 0), color=(0.45, 0.47, 0.5))]
# Only the final platform is a horizontal bar. Intermediate stage endpoints
# are direct leg-to-leg welds.
bodies.append(link("platform_top", [-HALF, 0, 0], [HALF, 0, 0],
                   tr(0, N * H), (0.35, 0.72, 0.45)))
for i in range(N):
    lo, hi = i * H, (i + 1) * H
    bodies += [
        link(f"leg_a_{i}", [-L / 2, 0, 0], [L / 2, 0, 0],
             pose(0, (lo + hi) / 2, TH0), (0.93, 0.53, 0.18)),
        link(f"leg_b_{i}", [-L / 2, 0, 0], [L / 2, 0, 0],
             pose(0, (lo + hi) / 2, TH1), (0.36, 0.62, 0.86)),
    ]
    if i == 0:
        bodies.append(box("roller_bottom", tr(HALF, lo)))
    if i == N - 1:
        bodies.append(box("roller_top", tr(-HALF, hi)))

joints = []
for i in range(N):
    lo, hi = i * H, (i + 1) * H
    a, b = f"leg_a_{i}", f"leg_b_{i}"
    joints.append(JointDef(f"cross_{i}", "revolute", a, b,
                           pivot=[0, (lo + hi) / 2, 0], axis=Z))
    if i == 0:
        joints += [
            JointDef("base_pin", "revolute", "ground", a,
                     pivot=[-HALF, lo, 0], axis=Z),
            JointDef("base_rail", "prismatic", "ground", "roller_bottom",
                     pivot=[HALF, lo, 0], axis=X),
            JointDef("bottom", "revolute", "roller_bottom", b,
                     pivot=[HALF, lo, 0], axis=Z),
        ]
    elif i > 0:
        # Continuous lazy-tongs arms: the upper endpoint of each lower arm is
        # cross-connected to the matching lower endpoint of the next stage.
        joints += [
            JointDef(f"right_hinge_{i-1}_{i}", "revolute",
                     f"leg_a_{i-1}", f"leg_b_{i}",
                     pivot=[HALF, lo, 0], axis=Z),
            JointDef(f"left_hinge_{i-1}_{i}", "revolute",
                     f"leg_b_{i-1}", f"leg_a_{i}",
                     pivot=[-HALF, lo, 0], axis=Z),
        ]
    if i == N - 1:
        joints += [
            JointDef("top_pin", "revolute", a, "platform_top",
                     pivot=[HALF, hi, 0], axis=Z),
            JointDef("top", "revolute", b, "roller_top",
                     pivot=[-HALF, hi, 0], axis=Z),
            JointDef("top_rail", "prismatic", "roller_top", "platform_top",
                     pivot=[-HALF, hi, 0], axis=X),
        ]

# The stack has no contact and no joint limits, so the ideal-joint manifold has
# no physical boundary -- the advancing front otherwise marches over the
# legs-vertical singularity into the inverted, self-intersecting branch (the
# platform height N*L*sin(theta) is not monotonic in the leg angle, so a plain
# floor lets the over-centre arc back in).  Bracket the intended pre-vertical
# stratum directly: platform_top travel along the stacking axis (y) equals
# N*L*sin(stage_angle_0), so stops at the two stage-angle endpoints are stops in
# platform height.  This is the same kind of workspace stop the single-stage
# case declares.
_workspace = [WorkspaceBound("platform_height", body="platform_top", axis="y",
                             lo=N * L * math.sin(MIN_ANGLE),
                             hi=N * L * math.sin(MAX_ANGLE))]

MECHANISM = MechanismSpec(bodies=bodies, joints=joints, up_axis="y",
                          render_up_axis="z", gravity=9.8,
                          workspace_bounds=_workspace)

CASE = AtlasCaseSpec(
    name="scissor_lift_stack",
    mechanism=MECHANISM,
    contact_model="none",
    # The coupled stack has one intrinsic coordinate.  The first leg angle is
    # used as the global chart coordinate; all remaining stage angles are
    # constrained responses, not independent coordinates.
    coordinates=[CoordinateDef(
        "stage_angle_0", topology="bounded", lo=MIN_ANGLE, hi=MAX_ANGLE,
        source="body_angle",
        args={"body": "leg_a_0", "axis": Z, "reference": "identity"})],
    build=BuildOptions(
        # `step` is a step in the ambient pose tangent (unit-normalised); spread
        # across 19 bodies a 0.16 pose step advances stage_angle_0 by ~1e-2 rad,
        # so the ~1.1 rad reachable arc between the workspace stops fills in
        # ~100 samples.  `n_samples` is a safety ceiling only -- the front
        # drains once both directions reach the platform_height bounds.
        # `tangent_eps` << `step` keeps the secant-PCA tangent rank at 1 (the
        # single-stage case sets the same 1e-3 probe).
        sampling=SamplingOptions(n_samples=200, step=0.16, max_seconds=1800,
                                 tangent_rel_tol=0.80, tangent_eps=1e-3),
        # 28 joints over 8 stages: a per-body pose loss alone tears them, so the
        # decoder trains against the declared loop closures like the single-stage
        # case does, and is held to the same closure tolerance.
        decoder=DecoderOptions(epochs=5000, smooth_lambda=1e-3,
                               closure_weight=1.0),
        # The chart is bounded at both ends by singularities, and with no
        # actuator the stack simply falls: without a barrier the run leaves the
        # charted range within a second and keeps going into pure decoder
        # extrapolation (negative stage angles, i.e. an inverted mechanism).
        # k has to be read against THIS mechanism's scale, not copied from the
        # single-stage case: B = k*e^2/2 must absorb the stack's kinetic energy
        # (~1.5e3 J after a full fall) inside a small e, so k=1e4 stops it only
        # 0.6 rad past the bound.  1e8 holds the overshoot to 0.004 rad and
        # bounces with ~2% energy drift measured outside contact.
        boundary=BoundaryOptions(kind="sdf", k=1e8, margin=0.0,
                                 epochs=3000, width=64, layers=3),
        expect=Expectations(dimension=1, n_bodies=19),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2)),
    # Without a declared start the runner falls back to u=0, which is 0.3 rad
    # below the charted range: the decoder extrapolates and the stack renders
    # with its loop closures torn open.  Start where the mechanism is built.
    scenarios={"default": {"dt": 1e-3, "initial_u": [TH0],
                            "boundary_solver": "damped"}},
)


def VIEW_EMBED(q):
    """Display-only 1-D coordinate for ``view_sample_set``, drawn as a line: the
    scissor-lift extension = ``platform_top`` travel along the stacking axis (y).
    The stack has a single DOF.  Never passed to the fitter.
    """
    import numpy as np

    bodies = CASE.mechanism.driven_bodies
    order = [b.name for b in bodies]
    T = np.asarray(q, float).reshape(len(q), len(order), 3, 4)
    top = order.index("platform_top")
    y0 = np.asarray(bodies[top].initial_transform, float).reshape(4, 4)[1, 3]
    lift = T[:, top, 1, 3] - y0
    return np.column_stack((lift, np.zeros_like(lift)))
