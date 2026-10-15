"""A full-rotation planar 2R arm with a physically explicit collision obstacle.

The shoulder and elbow bearings have no stops, so absent collision the mechanism
has two continuous rotary freedoms.  A fixed spherical obstacle lies in the
tool's reachable annulus.  Its clearance boundary is therefore a physical
contact boundary, not a declared domain restriction.  The atlas receives only
the simulated articulated poses and contact-validity evidence.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, CanonicalOptions, ChartOptions, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances


LINK_1 = 1.0
LINK_2 = 1.0
TOOL_RADIUS = .10
OBSTACLE_CENTER = np.asarray((1.45, .38))
OBSTACLE_RADIUS = .17
_AXIS = (0.0, 0.0, 1.0)


def tool_position(theta_1, theta_2):
    """Evaluator-only forward kinematics for the physical tool centre."""
    return np.asarray((LINK_1 * np.cos(theta_1) + LINK_2 * np.cos(theta_1 + theta_2),
                       LINK_1 * np.sin(theta_1) + LINK_2 * np.sin(theta_1 + theta_2)))


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

    This deliberately derives the number of collision regions from the chosen
    geometry rather than declaring a puncture count to the case or learner.
    The grid is periodic in both bearings, matching their physical full turns.
    """
    angles = np.linspace(-np.pi, np.pi, int(resolution), endpoint=False)
    theta_1, theta_2 = np.meshgrid(angles, angles, indexing="ij")
    x = LINK_1 * np.cos(theta_1) + LINK_2 * np.cos(theta_1 + theta_2)
    y = LINK_1 * np.sin(theta_1) + LINK_2 * np.sin(theta_1 + theta_2)
    collision = ((x - OBSTACLE_CENTER[0]) ** 2 + (y - OBSTACLE_CENTER[1]) ** 2
                 <= (TOOL_RADIUS + OBSTACLE_RADIUS) ** 2)
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


CASE = AtlasCaseSpec(
    name="obstacle_constrained_2r",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef("__ground", MeshSpec("sphere", {"radius": .05, "subdiv": 1}), fixed=True,
                    color=(.30, .32, .35)),
            BodyDef("link_1", MeshSpec("link", {"p0": (0., 0., 0.), "p1": (LINK_1, 0., 0.),
                                                   "thick": .045}), density=180.0, color=(.25, .55, .90)),
            BodyDef("link_2", MeshSpec("compound", {"parts": [
                MeshSpec("link", {"p0": (LINK_1, 0., 0.), "p1": (LINK_1 + LINK_2, 0., 0.), "thick": .045}),
                MeshSpec("sphere", {"center": (LINK_1 + LINK_2, 0., 0.), "radius": TOOL_RADIUS, "subdiv": 2}),
            ]}), density=180.0, color=(.32, .76, .48)),
            BodyDef("obstacle", MeshSpec("sphere", {"center": (*OBSTACLE_CENTER, 0.),
                                                       "radius": OBSTACLE_RADIUS, "subdiv": 2}),
                    fixed=True, color=(.83, .32, .28)),
        ],
        joints=[
            JointDef("shoulder", "revolute", "link_1", "__ground", pivot=(0., 0., 0.), axis=_AXIS),
            JointDef("elbow", "revolute", "link_2", "link_1", pivot=(LINK_1, 0., 0.), axis=_AXIS),
        ],
        # Only the tool-bearing second link is allowed to encounter the obstacle.
        # Jointed link pairs are excluded by their joints, and no broad-phase
        # convenience contacts are admitted.
        contact_pairs=[("link_2", "obstacle")],
        up_axis="z", gravity=0.0,
    ),
    contact_model="exact_clearance",
    coordinates=[],
    actuators=[ActuatorDef("shoulder torque", kind="joint_torque", target="shoulder", hi=20.0),
               ActuatorDef("elbow torque", kind="joint_torque", target="elbow", hi=20.0)],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=None, step=.19749553, tangent_eps=.025, max_seconds=None),
        charts=ChartOptions(mode="mesh_atlas", overlap=.25),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
        decoder=DecoderOptions(epochs=3500, smooth_lambda=1e-4),
        dynamics=DynamicsOptions(distill=True, epochs=1500, n_teacher=1000),
        boundary=BoundaryOptions(kind="sdf", k=1e8, margin=0.0),
        expect=Expectations(n_bodies=2, contact_pairs=[("link_2", "obstacle")]),
        tolerances=Tolerances(decoder_translation_rmse=.05, decoder_rotation_deg=3.0),
    ),
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "direct_constrained"}},
)
