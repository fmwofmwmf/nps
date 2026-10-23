"""Two side-by-side planar pendulums whose end balls can collide.

Each pendulum hangs from its own unlimited revolute pivot, so absent collision
the pair has two continuous rotary freedoms.  The pivots are close enough that
the end balls meet when the pendulums swing toward each other; that clearance
boundary is a physical contact between two moving bodies, not a declared
domain restriction.  The atlas receives only the simulated articulated poses
and contact-validity evidence.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, CanonicalOptions, ChartOptions, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances


LENGTH = 2.0
BALL_RADIUS = .5
PIVOT_1 = np.asarray((0.0, 0.0))
PIVOT_2 = np.asarray((4.0, 0.0))
THICK = .06
_AXIS = (0.0, 0.0, 1.0)


def ball_positions(theta_1, theta_2):
    """Evaluator-only ball centres; each angle is measured from hanging down."""
    def tip(pivot, theta):
        return np.asarray((pivot[0] + LENGTH * np.sin(theta),
                           pivot[1] - LENGTH * np.cos(theta)))
    return tip(PIVOT_1, theta_1), tip(PIVOT_2, theta_2)


def _periodic_components(mask: np.ndarray) -> list[int]:
    """4-connected component sizes on the evaluator's periodic angle grid."""
    mask = np.asarray(mask, bool)
    seen = np.zeros_like(mask)
    sizes = []
    n0, n1 = mask.shape
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        queue = deque((start,)); seen[start] = True; size = 0
        while queue:
            i, j = queue.popleft(); size += 1
            for neighbour in (((i - 1) % n0, j), ((i + 1) % n0, j),
                              (i, (j - 1) % n1), (i, (j + 1) % n1)):
                if mask[neighbour] and not seen[neighbour]:
                    seen[neighbour] = True; queue.append(neighbour)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def characterize_collision_complement(resolution=240):
    """Dense evaluator-only characterization of this physical design.

    Ball-ball overlap is the only collision the geometry admits (the rods
    cannot reach each other without the balls overlapping first), so the
    region count is derived from the chosen geometry, never declared to the
    case or learner.  The grid is periodic in both pivots.
    """
    angles = np.linspace(-np.pi, np.pi, int(resolution), endpoint=False)
    theta_1, theta_2 = np.meshgrid(angles, angles, indexing="ij")
    (x1, y1), (x2, y2) = ball_positions(theta_1, theta_2)
    collision = (x1 - x2) ** 2 + (y1 - y2) ** 2 <= (2.0 * BALL_RADIUS) ** 2
    return {
        "resolution": int(resolution),
        "collision_component_sizes": _periodic_components(collision),
        "admissible_component_sizes": _periodic_components(~collision),
    }


EVALUATOR_REFERENCE = {
    "unconstrained_joint_space": "T^2",
    "method": "periodic dense collision-complement characterization",
    "topology": "frozen only after characterize_collision_complement",
}


def _pendulum(name, pivot, color):
    p0 = (float(pivot[0]), float(pivot[1]), 0.)
    p1 = (float(pivot[0]), float(pivot[1]) - LENGTH, 0.)
    return BodyDef(name, MeshSpec("compound", {"parts": [
        MeshSpec("link", {"p0": p0, "p1": p1, "thick": THICK}),
        MeshSpec("sphere", {"center": p1, "radius": BALL_RADIUS, "subdiv": 2}),
    ]}), density=180.0, color=color)


CASE = AtlasCaseSpec(
    name="twin_pendulum_contact",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef("__ground", MeshSpec("sphere", {"radius": .05, "subdiv": 1}), fixed=True,
                    color=(.30, .32, .35)),
            _pendulum("pendulum_1", PIVOT_1, (.25, .55, .90)),
            _pendulum("pendulum_2", PIVOT_2, (.32, .76, .48)),
        ],
        joints=[
            JointDef("pivot_1", "revolute", "pendulum_1", "__ground",
                     pivot=(*PIVOT_1, 0.), axis=_AXIS),
            JointDef("pivot_2", "revolute", "pendulum_2", "__ground",
                     pivot=(*PIVOT_2, 0.), axis=_AXIS),
        ],
        # The two swinging bodies are the only contact pair; each pendulum is
        # excluded from the ground by its joint.
        contact_pairs=[("pendulum_1", "pendulum_2")],
        up_axis="y", render_up_axis="z", gravity=9.8,
    ),
    contact_model="exact_clearance",
    coordinates=[],
    actuators=[ActuatorDef("pivot 1 torque", kind="joint_torque", target="pivot_1", hi=40.0),
               ActuatorDef("pivot 2 torque", kind="joint_torque", target="pivot_2", hi=40.0)],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=None, step=.19749553, tangent_eps=.025, max_seconds=None),
        charts=ChartOptions(mode="mesh_atlas", overlap=.25),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
        decoder=DecoderOptions(epochs=3500, smooth_lambda=1e-4),
        dynamics=DynamicsOptions(distill=True, epochs=1500, n_teacher=1000),
        boundary=BoundaryOptions(kind="sdf", k=1e8, margin=0.0),
        expect=Expectations(n_bodies=2, contact_pairs=[("pendulum_1", "pendulum_2")]),
        tolerances=Tolerances(decoder_translation_rmse=.05, decoder_rotation_deg=3.0),
    ),
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "direct_constrained"}},
)
