"""Two independent, ideal-joint slider-crank piston mechanisms.

Each crank, rod, and piston forms one physical closed slider-crank loop.  The
two loops share only a fixed base, so simulator observations span both cyclic
motions without providing their topology to the chart-learning contract.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, JointDef, MechanismSpec, MeshSpec, SamplingOptions

X = [1.0, 0.0, 0.0]
Z = [0.0, 0.0, 1.0]
CRANK_RADIUS = 0.45
ROD_LENGTH = 1.20


def _transform(x, y, angle=0.0):
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0, x], [s, c, 0.0, y],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _link(name, length, x, y, *, angle=0.0, color):
    return BodyDef(name, MeshSpec("link", {"p0": [-length / 2, 0, 0],
                                              "p1": [length / 2, 0, 0],
                                              "thick": 0.07}),
                   configuration="planar_rigid",
                   initial_transform=_transform(x, y, angle), color=color)


def _piston_parts(prefix, y, color, initial_angle):
    pivot = np.array([-2.0, y, 0.0])
    crank_pin = pivot + np.array([CRANK_RADIUS * math.cos(initial_angle),
                                  CRANK_RADIUS * math.sin(initial_angle), 0.0])
    piston_x = (pivot[0] + CRANK_RADIUS * math.cos(initial_angle)
                + math.sqrt(ROD_LENGTH ** 2
                            - (CRANK_RADIUS * math.sin(initial_angle)) ** 2))
    rod_angle = math.atan2(y - crank_pin[1], piston_x - crank_pin[0])
    bodies = [
        _link(f"{prefix}_crank", CRANK_RADIUS,
              pivot[0] + CRANK_RADIUS * math.cos(initial_angle) / 2,
              y + CRANK_RADIUS * math.sin(initial_angle) / 2,
              angle=initial_angle, color=color),
        _link(f"{prefix}_rod", ROD_LENGTH,
              (crank_pin[0] + piston_x) / 2, (crank_pin[1] + y) / 2,
              angle=rod_angle, color=(0.72, 0.74, 0.78)),
        BodyDef(f"{prefix}_piston", MeshSpec("box", {"center": [0, 0, 0],
                    "half": [0.18, 0.12, 0.10]}), configuration="planar_rigid",
                initial_transform=_transform(piston_x, y), color=color),
    ]
    joints = [
        JointDef(f"{prefix}_crank_axle", "revolute", "ground", f"{prefix}_crank",
                 pivot=pivot, axis=Z),
        JointDef(f"{prefix}_rod_pin", "revolute", f"{prefix}_crank", f"{prefix}_rod",
                 pivot=crank_pin, axis=Z),
        JointDef(f"{prefix}_wrist_pin", "revolute", f"{prefix}_rod", f"{prefix}_piston",
                 pivot=[piston_x, y, 0.0], axis=Z),
        JointDef(f"{prefix}_cylinder", "prismatic", "ground", f"{prefix}_piston",
                 pivot=[piston_x, y, 0.0], axis=X),
    ]
    return bodies, joints


_a_bodies, _a_joints = _piston_parts(
    "piston_a", -0.75, (0.93, 0.53, 0.18), initial_angle=0.71)
_b_bodies, _b_joints = _piston_parts(
    "piston_b", 0.75, (0.36, 0.62, 0.86), initial_angle=-0.93)


def VIEW_EMBED(q):
    """Display-only observed angles of the two independently driven cranks."""
    transforms = np.asarray(q, float).reshape(len(q), -1, 3, 4)
    frames = transforms[:, (0, 3), :, :3]
    return np.arctan2(frames[:, :, 1, 0], frames[:, :, 0, 0])


CASE = AtlasCaseSpec(
    name="twin_independent_pistons",
    mechanism=MechanismSpec(
        bodies=[BodyDef("ground", MeshSpec("box", {"center": [0, 0, -0.14],
                                                      "half": [2.8, 1.8, 0.12]}),
                        fixed=True, color=(0.45, 0.47, 0.50))]
               + _a_bodies + _b_bodies,
        joints=_a_joints + _b_joints,
        up_axis="y", render_up_axis="z", gravity=0.0,
    ),
    contact_model="none",
    coordinates=[
        CoordinateDef("crank_a", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "piston_a_crank_axle"}),
        CoordinateDef("crank_b", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "piston_b_crank_axle"}),
    ],
    actuators=[
        ActuatorDef("piston A crank torque", kind="joint_torque",
                    target="piston_a_crank_axle", hi=40.0),
        ActuatorDef("piston B crank torque", kind="joint_torque",
                    target="piston_b_crank_axle", hi=40.0),
    ],
    build=BuildOptions(
        # Both physical motors complete a 2*pi cycle. Their observed coverage
        # has area (2*pi)^2, so 2*pi/sqrt(1000) targets the requested density.
        # The frontier itself, rather than a sample or time cap, decides completion.
        sampling=SamplingOptions(n_samples=None, step=.41286615,
                                 max_seconds=None,
                                 max_probes_per_node=4,
                                 # Measure tangent rank at a strictly local scale;
                                 # the larger front step is for surface coverage.
                                 tangent_eps=0.04,
                                 tangent_rel_tol=0.20),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
