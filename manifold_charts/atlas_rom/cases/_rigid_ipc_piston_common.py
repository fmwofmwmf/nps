"""Shared builder for the Rigid IPC ``092-piston`` slider-crank.

Source: ``assets/rigid_ipc/fixtures/092-piston.json``.  The fixture states one
constraint: the crank keeps only its z rotation.  The connecting rod and head
are free rigid bodies held by IPC contact alone (crank pin, wrist pin, bore),
and the cylinder is static.

The STLs are authored as one assembly about the crank axle at the origin (crank
pin at y=+0.7, wrist pin at y=-1.63, scale 0.001).  The fixture turns every body
by 180 deg about z and lifts the cylinder by 0.3; in the STL's own frame that
lift is -0.3 along y, which centres the bore on the head's stroke.  Every pin,
eye, and bore has 0.005 radial clearance, so `D_HAT` stays below it.

System-definition change (user decision): the mechanism is declared PLANAR.
The fixture leaves rod and head free in 3D, so they slid along the pin axes and
spun in the bore.  `planar_stages` carries each on a physical x/y stage --
prismatic x from the cylinder to a carriage, prismatic y to a second carriage,
revolute z at the body's centre of mass -- the same construction as
``intermittent_motion_398_ipc``.  Cases list the stage bodies after all real
bodies and restrict contact to real bodies with `real_contact_pairs`.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import BodyDef, JointDef, MeshSpec

_ROOT = "assets/rigid_ipc/meshes/507-movements/092-piston"
SCALE = 0.001
D_HAT = 0.002
CYLINDER_SHIFT = -0.3
X, Y, Z = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
# Volume centroids (trimesh `center_mass`) of the scaled rod and head at a unit
# offset of zero; the planar-stage pivots.
_ROD_COM = (0.0, -0.4192, 0.23)
_HEAD_COM = (0.0, -1.7432, 0.23)
_CARRIAGE_HALF = 0.01


def _translate(x, y, z):
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def _part(name, stl, offset, *, fixed=False, color=None):
    return BodyDef(name, MeshSpec("file", {"path": f"{_ROOT}/{stl}.stl",
                                              "scale": SCALE}),
                   fixed=fixed, initial_transform=_translate(*offset), color=color)


def piston_unit(prefix, offset=(0.0, 0.0, 0.0), color=(0.93, 0.53, 0.18)):
    """One 092 slider-crank with its crank axle at `offset`."""
    x, y, z = offset
    bodies = [
        _part(f"{prefix}_crank", "crank", offset, color=color),
        _part(f"{prefix}_rod", "connecting-rod", offset, color=(0.72, 0.74, 0.78)),
        _part(f"{prefix}_head", "head", offset, color=color),
        _part(f"{prefix}_cylinder", "cylinder", (x, y + CYLINDER_SHIFT, z),
              fixed=True, color=(0.45, 0.47, 0.50)),
    ]
    joints = [JointDef(f"{prefix}_crank_axle", "revolute", f"{prefix}_crank",
                       pivot=[x, y, z], axis=Z)]
    return bodies, joints


def _carriage(name, pivot):
    T = np.eye(4)
    T[:3, 3] = pivot
    return BodyDef(name, MeshSpec("box", {"center": [0.0, 0.0, 0.0],
                                          "half": [_CARRIAGE_HALF] * 3}),
                   initial_transform=T, color=(0.45, 0.55, 0.70))


def planar_stages(prefix, offset=(0.0, 0.0, 0.0)):
    """Planar x/y/rot-z stages for the rod and head of the unit at `offset`:
    cylinder -x-> <body>_xc -y-> <body>_yc -rot z-> body, all at the body's COM."""
    bodies, joints = [], []
    for part, com in (("rod", _ROD_COM), ("head", _HEAD_COM)):
        body = f"{prefix}_{part}"
        pivot = [c + o for c, o in zip(com, offset)]
        xc, yc = f"{body}_xc", f"{body}_yc"
        bodies += [_carriage(xc, pivot), _carriage(yc, pivot)]
        joints += [JointDef(f"{body}_stage_x", "prismatic", f"{prefix}_cylinder", xc,
                            pivot=pivot, axis=X),
                   JointDef(f"{body}_stage_y", "prismatic", xc, yc, pivot=pivot, axis=Y),
                   JointDef(f"{body}_stage_rot", "revolute", body, yc, pivot=pivot, axis=Z)]
    return bodies, joints


def real_contact_pairs(bodies):
    """Every pair of the given (real) bodies may touch; stage carriages touch nothing."""
    names = [b.name for b in bodies]
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
