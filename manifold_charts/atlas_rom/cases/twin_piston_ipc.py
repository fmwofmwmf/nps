"""Two independent Rigid IPC ``092-piston`` slider-cranks side by side.

Each unit is the `piston_ipc` mechanism; the two share nothing but the fixed
world.  Only each crank axle is declared; rods and heads are contact-held.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BuildOptions, CoordinateDef, MechanismSpec, SamplingOptions
from atlas_rom.cases._rigid_ipc_piston_common import D_HAT, piston_unit

# Crank discs have radius 1.0; 3.0 leaves a clear gap between the two units.
SPACING = 3.0

_a_bodies, _a_joints = piston_unit("piston_a", (-SPACING / 2, 0.0, 0.0),
                                   color=(0.93, 0.53, 0.18))
_b_bodies, _b_joints = piston_unit("piston_b", (SPACING / 2, 0.0, 0.0),
                                   color=(0.36, 0.62, 0.86))


def VIEW_EMBED(q):
    """Display-only observed angles of the two cranks."""
    transforms = np.asarray(q, float).reshape(len(q), -1, 3, 4)
    frames = transforms[:, (0, 3), :, :3]
    return np.arctan2(frames[:, :, 1, 0], frames[:, :, 0, 0])


CASE = AtlasCaseSpec(
    name="twin_piston_ipc",
    mechanism=MechanismSpec(bodies=_a_bodies + _b_bodies,
                            joints=_a_joints + _b_joints,
                            up_axis="y", render_up_axis="y", gravity=0.0),
    contact_model="exact_clearance",
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
        sampling=SamplingOptions(n_samples=None, max_seconds=None, d_hat=D_HAT),
    ),
    scenarios={"default": {"dt": 1e-2}},
)
