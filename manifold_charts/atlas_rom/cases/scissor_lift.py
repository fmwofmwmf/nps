"""A single-pair, roller-ended scissor lift."""
from __future__ import annotations

import math

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances, WorkspaceBound

LEG = 1.0
BASE_HALF = 0.3
HEIGHT = math.sqrt(LEG * LEG - (2.0 * BASE_HALF) ** 2)
MIN_HEIGHT = 0.50
MIN_ANGLE = math.asin(MIN_HEIGHT / LEG)
MAX_ANGLE = 1.43
THETA_A = math.atan2(HEIGHT, 2.0 * BASE_HALF)
THETA_B = math.pi - THETA_A
Z = [0.0, 0.0, 1.0]
X = [1.0, 0.0, 0.0]


def _translation(x, y, z=0.0):
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


def _pose(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]


def _anchor(name, x):
    return BodyDef(name=name, mesh=MeshSpec("box", {"center": [0, 0, 0],
                     "half": [0.03, 0.03, 0.03]}), fixed=True,
                   initial_transform=_translation(x, 0), color=(0.45, 0.47, 0.50))


CASE = AtlasCaseSpec(
    name="scissor_lift",
    mechanism=MechanismSpec(
        bodies=[
            _anchor("ground_pin", -BASE_HALF),
            _anchor("ground_rail", BASE_HALF),
            BodyDef("roller_bottom", MeshSpec("box", {"center": [0, 0, 0],
                     "half": [0.03, 0.03, 0.03]}),
                    initial_transform=_translation(BASE_HALF, 0),
                    color=(0.72, 0.74, 0.78)),
            BodyDef("leg_a", MeshSpec("link", {"p0": [-LEG / 2, 0, 0],
                     "p1": [LEG / 2, 0, 0], "thick": 0.06}),
                    initial_transform=_pose(0, HEIGHT / 2, THETA_A),
                    color=(0.93, 0.53, 0.18)),
            BodyDef("leg_b", MeshSpec("link", {"p0": [-LEG / 2, 0, 0],
                     "p1": [LEG / 2, 0, 0], "thick": 0.06}),
                    initial_transform=_pose(0, HEIGHT / 2, THETA_B),
                    color=(0.36, 0.62, 0.86)),
            BodyDef("roller_top", MeshSpec("box", {"center": [0, 0, 0],
                     "half": [0.03, 0.03, 0.03]}),
                    initial_transform=_translation(-BASE_HALF, HEIGHT),
                    color=(0.72, 0.74, 0.78)),
            BodyDef("platform", MeshSpec("link", {"p0": [-0.3, 0, 0],
                     "p1": [0.3, 0, 0], "thick": 0.06}),
                    initial_transform=_translation(0, HEIGHT),
                    color=(0.35, 0.72, 0.45)),
        ],
        joints=[
            JointDef("base_pin", "revolute", "ground_pin", "leg_a",
                     pivot=[-BASE_HALF, 0, 0], axis=Z),
            JointDef("base_rail", "prismatic", "ground_rail", "roller_bottom",
                     pivot=[BASE_HALF, 0, 0], axis=X),
            JointDef("bottom_roller", "revolute", "roller_bottom", "leg_b",
                     pivot=[BASE_HALF, 0, 0], axis=Z),
            JointDef("cross_pin", "revolute", "leg_a", "leg_b",
                     pivot=[0, HEIGHT / 2, 0], axis=Z),
            JointDef("platform_pin", "revolute", "leg_a", "platform",
                     pivot=[BASE_HALF, HEIGHT, 0], axis=Z),
            JointDef("top_roller", "revolute", "leg_b", "roller_top",
                     pivot=[-BASE_HALF, HEIGHT, 0], axis=Z),
            JointDef("platform_rail", "prismatic", "roller_top", "platform",
                     pivot=[-BASE_HALF, HEIGHT, 0], axis=X),
        ],
        workspace_bounds=[WorkspaceBound("platform_height", body="platform",
                                         axis="y", lo=MIN_HEIGHT, hi=1.02)],
        up_axis="y", render_up_axis="z", gravity=9.8,
    ),
    seeds=[], contact_model="none",
    coordinates=[CoordinateDef("lift_angle", topology="bounded", lo=MIN_ANGLE,
                               hi=MAX_ANGLE, source="body_angle",
                               args={"body": "leg_a", "axis": Z,
                                     "reference": "identity"})],
    actuators=[ActuatorDef("lift torque", kind="joint_torque",
                           target="cross_pin", hi=40.0)],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=192, step=0.036, tangent_eps=0.001),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=5000, smooth_lambda=1e-3,
                               closure_weight=1.0),
        dynamics=DynamicsOptions(distill=True, epochs=2500, n_teacher=1200),
        boundary=BoundaryOptions(kind="sdf", k=1e4, margin=0.0,
                                 epochs=3000, width=64, layers=3),
        expect=Expectations(dimension=1, n_bodies=5),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2)),
    scenarios={"default": {"dt": 1e-4, "initial_u": [math.pi / 2]}},
)
