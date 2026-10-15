"""Rigid IPC ``398-intermitten-motion``: a curved wheel intermittently drives a
piston between fixed walls.

Source: ``assets/rigid_ipc/fixtures/398-intermitten-motion.json`` (scale 0.002).
The fixture states two constraints: the curved wheel and the end piece each
keep only their z rotation, about their own centre of mass.  The axis and the
piston are free rigid bodies held by IPC contact alone; the walls are static.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, JointDef, MechanismSpec, MeshSpec, SamplingOptions

_ROOT = "assets/rigid_ipc/meshes/507-movements/398-intermitten-motion"
_SCALE = 0.002
Z = [0.0, 0.0, 1.0]
# Rigid IPC rotates a DOF-fixed body about its centre of mass.  These are the
# volume centroids of the scaled, watertight meshes (trimesh `center_mass`).
_WHEEL_COM = [0.0927, 0.2130, 0.0]
_END_COM = [0.7567, 0.2143, -0.0093]
# The fixture's initial barrier activation distance.
_D_HAT = 1e-4


def _body(name, mesh, *, fixed=False, color=None):
    return BodyDef(name, MeshSpec("file", {"path": f"{_ROOT}/{mesh}.off",
                                              "scale": _SCALE}),
                   fixed=fixed, color=color)


def VIEW_EMBED(q):
    """Display-only observed angles of the wheel and the end piece."""
    transforms = np.asarray(q, float).reshape(len(q), -1, 3, 4)
    frames = transforms[:, (1, 2), :, :3]
    return np.arctan2(frames[:, :, 1, 0], frames[:, :, 0, 0])


CASE = AtlasCaseSpec(
    name="intermittent_motion_398_ipc",
    mechanism=MechanismSpec(
        bodies=[
            _body("axis", "axis", color=(0.72, 0.74, 0.78)),
            _body("wheel", "curved_wheel", color=(0.93, 0.53, 0.18)),
            _body("end", "end", color=(0.36, 0.62, 0.86)),
            _body("piston", "piston", color=(0.55, 0.78, 0.45)),
            _body("walls", "walls", fixed=True, color=(0.45, 0.47, 0.50)),
        ],
        joints=[
            JointDef("wheel_axle", "revolute", "wheel", pivot=_WHEEL_COM, axis=Z),
            JointDef("end_axle", "revolute", "end", pivot=_END_COM, axis=Z),
        ],
        up_axis="y", render_up_axis="y", gravity=0.0,
    ),
    contact_model="exact_clearance",
    coordinates=[
        CoordinateDef("wheel", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle", args={"joint": "wheel_axle"}),
    ],
    actuators=[ActuatorDef("wheel torque", kind="joint_torque",
                           target="wheel_axle", hi=40.0)],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=None, max_seconds=None, d_hat=_D_HAT),
    ),
    scenarios={"default": {"dt": 1e-2}},
)
