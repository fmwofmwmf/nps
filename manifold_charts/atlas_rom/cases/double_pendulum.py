"""cases/double_pendulum.py — a planar 2R open chain under gravity: the classic
double pendulum, built as two ideal revolute joints with no closure loop at all.

DOF, counted the way the mechanism actually closes (planar, 2 driven bodies):

    6   =  3 * 2 planar freedoms
   -4   =  2 ideal revolute joints * 2 constraints
   ----
    2   =  shoulder angle + elbow angle

Unlike `five_bar` or `pantograph_2d`, this chain never closes back on itself, so
there is no coupler triangle to solve and no discrete elbow-in/elbow-out branch:
the two joint angles ARE a valid global coordinate, injective everywhere on the
2-torus they sweep.  That is what makes this the simplest possible case in the
"contact-free ideal joint" family — declared `joint_angle` coordinates, no
chart discovery, no closure loss, `sampler="analytic"` (the default the spec
picks for `contact_model="none"`).

Geometry is rest-world-baked (bodies start at transform = identity, links given
in world coordinates directly) matching the convention used throughout
`atlas_rom/cases/`.  The chain hangs down from the shoulder at rest so gravity
drives the swing the way a physical double pendulum would.
"""
from __future__ import annotations

import math

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, JointDef, MechanismSpec, MeshSpec, SamplingOptions

Z = [0.0, 0.0, 1.0]                       # both hinges are planar about +z
LINK_1 = 0.9
LINK_2 = 0.7
THICK = 0.06

SHOULDER = [0.0, 0.0, 0.0]
ELBOW = [0.0, -LINK_1, 0.0]
TIP = [0.0, -LINK_1 - LINK_2, 0.0]


def _bar(name, p0, p1, color):
    return BodyDef(name=name, mesh=MeshSpec("link", {"p0": p0, "p1": p1,
                                                      "thick": THICK}),
                   color=color)


CASE = AtlasCaseSpec(
    name="double_pendulum",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef("__ground", MeshSpec("sphere", {"radius": 0.05, "subdiv": 1}),
                    fixed=True, color=(0.30, 0.32, 0.35)),
            _bar("link_1", SHOULDER, ELBOW, (0.93, 0.53, 0.18)),
            _bar("link_2", ELBOW, TIP, (0.36, 0.62, 0.86)),
        ],
        joints=[
            JointDef("shoulder", "revolute", "__ground", "link_1",
                     pivot=SHOULDER, axis=Z),
            JointDef("elbow", "revolute", "link_1", "link_2",
                     pivot=ELBOW, axis=Z),
        ],
        up_axis="y",                     # planar chain in xy: gravity is in-plane
        render_up_axis="z",
        gravity=9.8,
    ),
    contact_model="none",                # ideal joints, open chain, no closure
    coordinates=[
        CoordinateDef("shoulder_angle", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "shoulder"}),
        CoordinateDef("elbow_angle", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "elbow"}),
    ],
    actuators=[
        ActuatorDef("shoulder torque", kind="joint_torque",
                    target="shoulder", hi=40.0),
        ActuatorDef("elbow torque", kind="joint_torque",
                    target="elbow", hi=20.0),
    ],
    build=BuildOptions(
        # Both angles sweep a full 2*pi; observed coverage has area (2*pi)^2,
        # so 2*pi/sqrt(1000) targets the requested sample density (same
        # heuristic as twin_independent_pistons' independent 2*pi cranks).
        sampling=SamplingOptions(n_samples=1000, step=2.0 * math.pi / math.sqrt(1000),
                                 max_seconds=600,
                                 tangent_eps=0.04, tangent_rel_tol=0.20),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
