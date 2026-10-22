"""Rigid IPC ``398-intermitten-motion``: a curved wheel intermittently drives a
piston between fixed walls.

Source: ``assets/rigid_ipc/fixtures/398-intermitten-motion.json`` (scale 0.002).
Geometry is read from ``398.blend`` (one object per body,
named like the body, raw fixture units, assembled pose) so it can be edited in
Blender; it was converted from the fixture's ``.off`` meshes (kept), and the end
and piston were then edited by the user.
The fixture states two constraints: the curved wheel and the end piece each
keep only their z rotation, about their own centre of mass.  The axis and the
piston are free rigid bodies held by IPC contact alone; the walls are static.

System-definition change (user decision): the mechanism is declared PLANAR.
The fixture leaves the axis and the piston free in 3D, so nothing held them in
the xy plane except contacts that never cap z.  Each is now carried by a
physical x/y stage -- prismatic x from the walls to a carriage, prismatic y to
a second carriage, revolute z at the body's centre of mass -- the same serial
construction as ``generic_annulus`` (a ``planar_rigid`` label would not
constrain the solver).  The carriages are tiny boxes at the pivot, listed last
so the original body order is unchanged, and are excluded from contact by the
explicit ``contact_pairs``.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, JointDef, MechanismSpec, MeshSpec, SamplingOptions

_BLEND = "assets/rigid_ipc/meshes/507-movements/398-intermitten-motion/398.blend"
_SCALE = 0.002
Z = [0.0, 0.0, 1.0]
# Rigid IPC rotates a DOF-fixed body about its centre of mass.  These are the
# volume centroids of the scaled, watertight meshes (trimesh `center_mass`).
_WHEEL_COM = [0.0927, 0.2130, 0.0]
_END_COM = [0.7567, 0.2143, -0.0093]
# Planar-stage pivots of the two contact-held bodies (same centroid rule).
_AXIS_COM = [0.5848, 0.2357, 0.0156]
_PISTON_COM = [0.3554, 0.2152, 0.0083]
X, Y = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
# The fixture's initial barrier activation distance.
_D_HAT = 1e-4
_CARRIAGE_HALF = 0.004


def _body(name, *, fixed=False, color=None):
    return BodyDef(name, MeshSpec("file", {"path": _BLEND, "node": name,
                                              "scale": _SCALE}),
                   fixed=fixed, color=color)


def _carriage(name, pivot):
    T = np.eye(4)
    T[:3, 3] = pivot
    return BodyDef(name, MeshSpec("box", {"center": [0.0, 0.0, 0.0],
                                          "half": [_CARRIAGE_HALF] * 3}),
                   initial_transform=T, color=(0.45, 0.55, 0.70))


def _planar_stage(body, pivot):
    """walls -x-> <body>_xc -y-> <body>_yc -rot z-> body, all at ``pivot``."""
    xc, yc = f"{body}_xc", f"{body}_yc"
    return ([_carriage(xc, pivot), _carriage(yc, pivot)],
            [JointDef(f"{body}_stage_x", "prismatic", "walls", xc, pivot=pivot, axis=X),
             JointDef(f"{body}_stage_y", "prismatic", xc, yc, pivot=pivot, axis=Y),
             JointDef(f"{body}_stage_rot", "revolute", body, yc, pivot=pivot, axis=Z)])


_axis_stage_bodies, _axis_stage_joints = _planar_stage("axis", _AXIS_COM)
_piston_stage_bodies, _piston_stage_joints = _planar_stage("piston", _PISTON_COM)
_REAL = ("axis", "wheel", "end", "piston", "walls")


def VIEW_EMBED(q):
    """Display-only observed angles of the wheel and the end piece."""
    transforms = np.asarray(q, float).reshape(len(q), -1, 3, 4)
    frames = transforms[:, (1, 2), :, :3]
    return np.arctan2(frames[:, :, 1, 0], frames[:, :, 0, 0])


CASE = AtlasCaseSpec(
    name="intermittent_motion_398_ipc",
    mechanism=MechanismSpec(
        bodies=[
            _body("axis", color=(0.72, 0.74, 0.78)),
            _body("wheel", color=(0.93, 0.53, 0.18)),
            _body("end", color=(0.36, 0.62, 0.86)),
            _body("piston", color=(0.55, 0.78, 0.45)),
            _body("walls", fixed=True, color=(0.45, 0.47, 0.50)),
            *_axis_stage_bodies, *_piston_stage_bodies,
        ],
        joints=[
            JointDef("wheel_axle", "revolute", "wheel", pivot=_WHEEL_COM, axis=Z),
            JointDef("end_axle", "revolute", "end", pivot=_END_COM, axis=Z),
            *_axis_stage_joints, *_piston_stage_joints,
        ],
        # Every pair of real bodies may touch (as before); carriages touch nothing.
        contact_pairs=[(a, b) for i, a in enumerate(_REAL) for b in _REAL[i + 1:]],
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
