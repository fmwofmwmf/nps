"""cases/planar_3r_arm.py — a planar 3R open chain under gravity: the triple
pendulum, built as three ideal revolute joints with no closure loop at all.

DOF, counted the way the mechanism actually closes (planar, 3 driven bodies):

    9   =  3 * 3 planar freedoms
   -6   =  3 ideal revolute joints * 2 constraints
   ----
    3   =  shoulder + elbow + wrist angle

The three-dimensional sibling of `double_pendulum`: a flat product of circles,
so every difficulty lies in the periodic wrap rather than in curvature or
closure.  No coordinates are declared: the learner receives only
simulator-produced poses and discovers its own charts.  The expected answer
lives in `EVALUATOR_REFERENCE`, which only the evaluator reads.

Geometry is rest-world-baked (bodies start at transform = identity, links given
in world coordinates directly) matching the convention used throughout
`atlas_rom/cases/`.  The chain hangs down from the shoulder at rest.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, JointDef, MechanismSpec, MeshSpec, SamplingOptions

Z = [0.0, 0.0, 1.0]                       # all hinges are planar about +z
LINK_1 = 0.6
LINK_2 = 0.5
LINK_3 = 0.4
THICK = 0.06

SHOULDER = [0.0, 0.0, 0.0]
ELBOW = [0.0, -LINK_1, 0.0]
WRIST = [0.0, -LINK_1 - LINK_2, 0.0]
TIP = [0.0, -LINK_1 - LINK_2 - LINK_3, 0.0]


def _bar(name, p0, p1, color):
    return BodyDef(name=name, mesh=MeshSpec("link", {"p0": p0, "p1": p1,
                                                      "thick": THICK}),
                   color=color)


CASE = AtlasCaseSpec(
    name="planar_3r_arm",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef("__ground", MeshSpec("sphere", {"radius": 0.05, "subdiv": 1}),
                    fixed=True, color=(0.30, 0.32, 0.35)),
            _bar("link_1", SHOULDER, ELBOW, (0.93, 0.53, 0.18)),
            _bar("link_2", ELBOW, WRIST, (0.36, 0.62, 0.86)),
            _bar("link_3", WRIST, TIP, (0.45, 0.78, 0.42)),
        ],
        joints=[
            JointDef("shoulder", "revolute", "__ground", "link_1",
                     pivot=SHOULDER, axis=Z),
            JointDef("elbow", "revolute", "link_1", "link_2",
                     pivot=ELBOW, axis=Z),
            JointDef("wrist", "revolute", "link_2", "link_3",
                     pivot=WRIST, axis=Z),
        ],
        up_axis="y",                     # planar chain in xy: gravity is in-plane
        render_up_axis="z",
        gravity=9.8,
    ),
    contact_model="none",                # ideal joints, open chain, no closure
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
        # step sizes the front for ~6300 samples: the C-space volume in the
        # packed 36-D pose metric is ~1158 (quadrature of sqrt det JᵀJ over the
        # joint angles), and the front accepts ~0.434*step^3 of volume per
        # sample (calibrated on generic_so3's closed SO(3) front, 4857 samples
        # at step 0.473), so step = (1158 / (0.434 * 6300))^(1/3) ~= 0.75.
        # No sample cap and no time budget: the frontier must close on its own
        # for the dataset to be trainable.
        sampling=SamplingOptions(n_samples=None, step=0.75,
                                 max_seconds=None, checkpoint_seconds=1200.,
                                 tangent_eps=0.075, tangent_rel_tol=0.20),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
    ),
    scenarios={"default": {"dt": 1e-3}},
)

# The learner never receives this assertion; it is checked independently by the
# evaluator/test harness.
EVALUATOR_REFERENCE = {"configuration_space": "T^3", "intrinsic_dimension": 3,
                       "representation": "shoulder, elbow and wrist joint angles"}
