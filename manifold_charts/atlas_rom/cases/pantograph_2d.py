"""cases/pantograph_2d.py — the actual drafting pantograph: a fixed pivot O, a
tracing pen P, and a drawing pen C that always stays collinear with O and P at
a FIXED magnification, for every reachable configuration.

GEOMETRY.  Two straight rigid rods and two coupler bars:

    crank  (body)  O --a-- D --Lb-- A        (O grounded; D, A on the same line)
    arm    (body)  A --La-- B --p-- P        (pinned to crank at A; B, P collinear with A)
    link_b (body)  B --Lb-- C                (pinned to arm at B)
    link_d (body)  D --La-- C                (pinned to crank at D)

closing the loop O-D-C-B-A-O.  ABCD (with D on crank, not a separate vertex)
forms a parallelogram: BC = AD = Lb and DC = AB = La, so `C = B + (D - A)`
exactly (verified below) rather than solved by circle intersection — this loop
has NO discrete elbow branch the way `five_bar`'s does, because the closure is
an affine sum, not a geometric intersection.

THE PANTOGRAPH IDENTITY.  Choosing `p = La * Lb / a` makes O, C, P exactly
collinear with |OP| / |OC| = 1 + Lb / a for every (theta1, theta_rel), not just
at rest — this is what a real pantograph is FOR: drag the outer pen P and the
inner pen C traces an exact 1/k-scale copy, always on the ray through O.
Verified analytically (`_forward` below) to 1e-9 over arbitrary angles, not
just the rest pose.

DOF, counted the way the mechanism actually closes (planar, 4 driven bodies):

    12  =  3 * 4 planar freedoms
   -10  =  5 ideal revolute joints * 2 constraints
   ----
     2  =  crank angle (joint O) + elbow angle (joint A)

Unlike `five_bar`'s two ground cranks (which both feed into a circle-intersection
for the coupler point E), these two joints serially determine D, A, B, P with no
ambiguity at all, and the closure formula for C is a fixed affine expression,
not a branch choice — so declared `joint_angle` coordinates are injective here
(same treatment as `double_pendulum` / `twin_independent_pistons`), and no
chart discovery is needed.

FEASIBILITY.  |B - D|^2 = Lb^2 + La^2 + 2*Lb*La*cos(theta_rel) always lies in
[(Lb-La)^2, (Lb+La)^2], so the C-B-D triangle closes for EVERY theta_rel, and
theta1 is a free rigid rotation of the whole assembly about O that never
affects that distance. So, unlike `five_bar`, there is no reachable-region
boundary to discover: both coordinates sweep the full 2*pi torus.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, JointDef, MechanismSpec, MeshSpec, SamplingOptions

Z = [0.0, 0.0, 1.0]                       # every pin is a planar hinge about +z
THICK = 0.06

OD = 0.35                                 # crank: O to D
DA = 0.85                                 # crank: D to A            (= BC)
AB = 0.65                                 # arm: A to B              (= DC)
BP = AB * DA / OD                         # arm: B to P, forces the pantograph identity
MAGNIFICATION = 1.0 + DA / OD             # |OP| / |OC|, for any configuration

THETA1_REST = 1.3                         # crank absolute angle at rest
THETA_REL_REST = 0.9                      # joint-A relative angle at rest


def _forward(theta1: float, theta_rel: float) -> dict[str, np.ndarray]:
    """All five named points from the two independent joint angles.

    `C` is the parallelogram sum, not a circle intersection: this is the whole
    reason the loop has no elbow branch. Returns 3-vectors (z=0) so callers can
    feed pivots straight into `JointDef`/mesh helpers.
    """
    u1 = np.array([math.cos(theta1), math.sin(theta1), 0.0])
    u2 = np.array([math.cos(theta1 + theta_rel), math.sin(theta1 + theta_rel), 0.0])
    O = np.zeros(3)
    D = OD * u1
    A = (OD + DA) * u1
    B = A + AB * u2
    P = A + (AB + BP) * u2
    C = B + (D - A)
    return {"O": O, "D": D, "A": A, "B": B, "P": P, "C": C}


_rest = _forward(THETA1_REST, THETA_REL_REST)
O, D, A, B, P, C = (_rest[k] for k in ("O", "D", "A", "B", "P", "C"))

assert abs(np.linalg.norm(C - B) - DA) < 1e-9
assert abs(np.linalg.norm(C - D) - AB) < 1e-9
_cross = C[0] * P[1] - C[1] * P[0]
assert abs(_cross) < 1e-9, "O, C, P must be collinear"
assert abs(np.linalg.norm(P) / np.linalg.norm(C) - MAGNIFICATION) < 1e-9


def _bar(name, p0, p1, color):
    return BodyDef(name=name, mesh=MeshSpec("link", {"p0": list(p0), "p1": list(p1),
                                                      "thick": THICK}),
                   color=color)


CASE = AtlasCaseSpec(
    name="pantograph_2d",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef("__ground", MeshSpec("sphere", {"radius": 0.05, "subdiv": 1}),
                    fixed=True, color=(0.45, 0.47, 0.50)),
            _bar("crank", O, A, (0.93, 0.53, 0.18)),      # O..D..A, driven at O
            _bar("arm", A, P, (0.36, 0.62, 0.86)),        # A..B..P, driven at A
            _bar("link_b", B, C, (0.72, 0.36, 0.72)),
            _bar("link_d", D, C, (0.35, 0.72, 0.45)),
        ],
        joints=[
            JointDef("O", "revolute", "__ground", "crank", pivot=O, axis=Z),
            JointDef("A", "revolute", "crank", "arm", pivot=A, axis=Z),
            JointDef("B", "revolute", "arm", "link_b", pivot=B, axis=Z),
            JointDef("D", "revolute", "link_d", "crank", pivot=D, axis=Z),
            JointDef("C", "revolute", "link_b", "link_d", pivot=C, axis=Z),  # the closure
        ],
        up_axis="y",                     # planar linkage in xy: gravity is in-plane
        render_up_axis="z",
        gravity=9.8,
    ),
    contact_model="none",                # ideal joints; parallelogram closure, no contact
    coordinates=[
        CoordinateDef("crank_angle", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "O"}),
        CoordinateDef("elbow_angle", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "A"}),
    ],
    actuators=[
        ActuatorDef("crank torque", kind="joint_torque", target="O", hi=40.0),
        ActuatorDef("elbow torque", kind="joint_torque", target="A", hi=20.0),
    ],
    build=BuildOptions(
        # Both angles sweep a full 2*pi with no feasibility boundary (see
        # module docstring).  `step` is a POSE-space (ambient ~48-dim ->
        # measured, not literally 48 -- 4 driven bodies packed) Euclidean
        # distance, not an angle -- the naive `2*pi/sqrt(n_samples)` heuristic
        # used by double_pendulum / twin_independent_pistons implicitly
        # assumes 1 radian moves the pose ~1 unit, which undercounts here:
        # the arm's BP extension (see MAGNIFICATION) makes pen P sweep a much
        # bigger radius than a unit rotation, so 1 radian moves the packed
        # pose by ~2.4 units on average (measured: SVD of d(pose)/d[theta1,
        # theta_rel] over 400 random configs gives singular values
        # sigma_max~3.9/sigma_min~1.5, geometric-mean ~2.41). Using the naive
        # step left the advancing front needing far more than 1000 samples to
        # close every cone, so it hit the n_samples cap mid-frontier
        # (frontier_complete=False). Scaling step by that factor (0.1987 *
        # 2.41 ~= 0.48) was still short: cone-retirement (each node needs its
        # local cone covered by neighbours, not just areal packing) needs
        # more samples than pure area/step^2 predicts. Measured at step=0.48:
        # still not complete after 2065 samples (queued cones plateaued at
        # 700-1200 instead of emptying), consistent with the Area_pose/step^2
        # estimate of ~3254 samples needed to close. Rescaling by
        # sqrt(3254/1000) gives step ~= 0.87, targeting genuine closure near
        # 1000 samples. At step=0.87 the queued/open cone ratio was actually
        # trending down (0.38 -> 0.25 over the last 100s) when it ran out of
        # the 600s budget at 1300 samples -- density is right, it just needed
        # more wall-clock... WRONG: a 1200s retry raised `undersized` from
        # 38->3138 with almost no new samples. Traced it to theta_rel~=0/+-pi
        # (arm fully extended along, or folded back onto, the crank): the
        # Jacobian isn't singular there (ratio ~2-3x, same order as
        # everywhere else), but curvature is sharp enough that ANY probe
        # direction from those ~10-15 nodes gets snapped back near the
        # parent by the constraint projection, so they never satisfy the
        # maximin cone-closure test and get re-queued forever (585 probes on
        # one node, vs the ~16 a well-conditioned 2-D cone needs).
        # `max_probes_per_node` gives up on a node after 16 failed tries
        # instead of grinding -- a localized fix, no step change needed.
        sampling=SamplingOptions(n_samples=1600, step=0.87,
                                 max_seconds=600, max_probes_per_node=16,
                                 tangent_eps=0.04, tangent_rel_tol=0.20),
    ),
    scenarios={"default": {"dt": 1e-3}},
)


# Display / task-space hook for ``viewers.rom_target``: pen name -> (body, rest-world local point).  The FIRST pen
# is the draggable one.  Reach circles are drawn for reference only.
PEN_TARGET = {
    "pens": {"C": ("link_b", [float(x) for x in C]), "P": ("arm", [float(x) for x in P])},
    "circles": [("reach (inner)", abs(OD - AB)), ("reach (outer)", OD + AB)],
    "torque_limit": 20.0,
}
