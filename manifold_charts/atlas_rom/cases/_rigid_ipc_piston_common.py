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
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import BodyDef, JointDef, MeshSpec

_ROOT = "assets/rigid_ipc/meshes/507-movements/092-piston"
SCALE = 0.001
D_HAT = 0.002
CYLINDER_SHIFT = -0.3
Z = [0.0, 0.0, 1.0]


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
