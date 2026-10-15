"""Rigid IPC ``092-piston``: one slider-crank held together by IPC contact.

Only the crank axle is a declared joint (the fixture's single stated
constraint).  Rod and head are free rigid bodies; the pin, wrist pin, and bore
act through contact alone.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BuildOptions, CoordinateDef, MechanismSpec, SamplingOptions
from atlas_rom.cases._rigid_ipc_piston_common import D_HAT, piston_unit

_bodies, _joints = piston_unit("piston")


def VIEW_EMBED(q):
    """Display-only observed crank angle."""
    transforms = np.asarray(q, float).reshape(len(q), -1, 3, 4)
    frame = transforms[:, 0, :, :3]
    return np.arctan2(frame[:, 1, 0], frame[:, 0, 0])[:, None]


CASE = AtlasCaseSpec(
    name="piston_ipc",
    mechanism=MechanismSpec(bodies=_bodies, joints=_joints,
                            up_axis="y", render_up_axis="y", gravity=0.0),
    contact_model="exact_clearance",
    coordinates=[
        CoordinateDef("crank", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "piston_crank_axle"}),
    ],
    actuators=[ActuatorDef("crank torque", kind="joint_torque",
                           target="piston_crank_axle", hi=40.0)],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=None, max_seconds=None, d_hat=D_HAT),
    ),
    scenarios={"default": {"dt": 1e-2}},
)
