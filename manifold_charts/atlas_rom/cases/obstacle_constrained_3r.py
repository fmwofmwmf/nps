"""A full-rotation planar 3R arm with a ball tool and two fixed spherical obstacles.

The three bearings have no stops, so absent collision the mechanism has three
continuous rotary freedoms (the `planar_3r_arm` chain).  A ball welded to the
last link may not penetrate either obstacle; the links themselves are not
declared contact partners and pass through everything, as the links of
`planar_3r_arm` pass through each other.  The clearance boundary is therefore a
physical contact boundary, not a declared domain restriction.  The atlas
receives only the simulated articulated poses and contact-validity evidence.

Every declared contact pair is sphere-sphere, so the analytic oracle enforces
the clearance in closed form (`spec.primitive_clearance`) instead of the
libuipc contact oracle.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, JointDef, MechanismSpec, MeshSpec, SamplingOptions

Z = [0.0, 0.0, 1.0]                       # all hinges are planar about +z
LINK_1 = 0.6
LINK_2 = 0.5
LINK_3 = 0.4
THICK = 0.06
BALL_RADIUS = 0.08
OBSTACLES = {"obstacle_a": ((0.9, 0.5), 0.15),
             "obstacle_b": ((-0.4, -1.0), 0.12)}

SHOULDER = [0.0, 0.0, 0.0]
ELBOW = [0.0, -LINK_1, 0.0]
WRIST = [0.0, -LINK_1 - LINK_2, 0.0]
TIP = [0.0, -LINK_1 - LINK_2 - LINK_3, 0.0]


def _bar(name, p0, p1, color):
    return BodyDef(name=name, mesh=MeshSpec("link", {"p0": p0, "p1": p1,
                                                      "thick": THICK}),
                   color=color)


def tool_position(theta_1, theta_2, theta_3):
    """Evaluator-only forward kinematics for the ball centre (relative joint
    angles, measured from the hanging rest pose)."""
    a1 = np.asarray(theta_1)
    a2, a3 = a1 + theta_2, a1 + theta_2 + theta_3
    return np.stack((LINK_1 * np.sin(a1) + LINK_2 * np.sin(a2) + LINK_3 * np.sin(a3),
                     -(LINK_1 * np.cos(a1) + LINK_2 * np.cos(a2) + LINK_3 * np.cos(a3))), axis=-1)


def _periodic_component_sizes(mask: np.ndarray) -> list[int]:
    """Connected component sizes of a boolean grid that wraps on every axis."""
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    parent = list(range(count + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for axis in range(mask.ndim):
        first, last = np.take(labels, 0, axis=axis), np.take(labels, -1, axis=axis)
        for a, b in zip(first[(first > 0) & (last > 0)], last[(first > 0) & (last > 0)]):
            parent[find(a)] = find(b)
    sizes: dict[int, int] = {}
    for label, size in zip(*np.unique(labels[labels > 0], return_counts=True)):
        root = find(int(label))
        sizes[root] = sizes.get(root, 0) + int(size)
    return sorted(sizes.values(), reverse=True)


def characterize_collision_complement(resolution=96):
    """Dense evaluator-only characterization of this physical design.

    Derives the collision and admissible regions from the chosen geometry on a
    grid periodic in all three bearings, rather than declaring a topology to
    the case or learner.
    """
    angles = np.linspace(-np.pi, np.pi, int(resolution), endpoint=False)
    grid = np.meshgrid(angles, angles, angles, indexing="ij")
    tool = tool_position(*grid)
    collision = np.zeros(grid[0].shape, bool)
    for centre, radius in OBSTACLES.values():
        collision |= np.linalg.norm(tool - np.asarray(centre), axis=-1) < radius + BALL_RADIUS
    return {
        "resolution": int(resolution),
        "collision_fraction": float(collision.mean()),
        "collision_component_sizes": _periodic_component_sizes(collision),
        "admissible_component_sizes": _periodic_component_sizes(~collision),
    }


EVALUATOR_REFERENCE = {
    "unconstrained_joint_space": "T^3",
    "intrinsic_dimension": 3,
    "method": "periodic dense collision-complement characterization",
    "topology": "frozen only after characterize_collision_complement",
}


CASE = AtlasCaseSpec(
    name="obstacle_constrained_3r",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef("__ground", MeshSpec("sphere", {"radius": 0.05, "subdiv": 1}),
                    fixed=True, color=(0.30, 0.32, 0.35)),
            _bar("link_1", SHOULDER, ELBOW, (0.93, 0.53, 0.18)),
            _bar("link_2", ELBOW, WRIST, (0.36, 0.62, 0.86)),
            _bar("link_3", WRIST, TIP, (0.45, 0.78, 0.42)),
            BodyDef("ball", MeshSpec("sphere", {"center": TIP, "radius": BALL_RADIUS,
                                                "subdiv": 2}),
                    color=(0.95, 0.85, 0.30)),
            *[BodyDef(name, MeshSpec("sphere", {"center": (*centre, 0.0), "radius": radius,
                                                 "subdiv": 2}),
                      fixed=True, color=(0.83, 0.32, 0.28))
              for name, (centre, radius) in OBSTACLES.items()],
        ],
        joints=[
            JointDef("shoulder", "revolute", "__ground", "link_1",
                     pivot=SHOULDER, axis=Z),
            JointDef("elbow", "revolute", "link_1", "link_2",
                     pivot=ELBOW, axis=Z),
            JointDef("wrist", "revolute", "link_2", "link_3",
                     pivot=WRIST, axis=Z),
            JointDef("tool", "fixed", "link_3", "ball", pivot=TIP),
        ],
        # Only the ball may encounter the obstacles.  Jointed link pairs are
        # excluded by their joints, and no broad-phase convenience contacts are
        # admitted.
        contact_pairs=[("ball", name) for name in OBSTACLES],
        up_axis="y",
        render_up_axis="z",
        gravity=0.0,
    ),
    contact_model="exact_clearance",
    # Discovery receives no coordinate labels, UVs, or declared chart count.
    coordinates=[],
    actuators=[
        ActuatorDef("shoulder torque", kind="joint_torque",
                    target="shoulder", hi=40.0),
        ActuatorDef("elbow torque", kind="joint_torque",
                    target="elbow", hi=20.0),
        ActuatorDef("wrist torque", kind="joint_torque",
                    target="wrist", hi=10.0),
    ],
    build=BuildOptions(
        # step sizes the front for ~7800 samples: the admissible C-space volume
        # in the packed 48-D pose metric is ~1735 (quadrature of sqrt det JᵀJ
        # over the joint angles, collision cells removed; 3.1% of T^3), and the
        # closed planar_3r_arm front accepted ~0.349*step^3 of volume per sample
        # (7864 samples at step 0.75), so step = (1735 / (0.349 * 7800))^(1/3)
        # ~= 0.86.  No sample cap and no time budget: the frontier must close on
        # its own for the dataset to be trainable.
        sampling=SamplingOptions(sampler="analytic", n_samples=None, step=0.86,
                                 max_seconds=None, checkpoint_seconds=1200.,
                                 tangent_eps=0.086, tangent_rel_tol=0.20),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
