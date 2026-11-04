"""cases/circle_arm_contact.py - the circle-drawing arm, its links located by SuperDex contact.

Geometry is `assets/circle_arm/circle_arm.py`, exported by its
`export_sim_meshes.py` at the rest pose, in mm, and scaled here to 1 unit = 12 mm
so the oracle's 0.0625-unit SDF voxel is 0.75 mm, two per 1.5 mm ring clearance.

The arm and the circle crank (pink) are hinged to ground, coaxially, because no
ground geometry exists for them to touch.  Every other body (blue, yellow and the
four rods) is held only by pins in rings.  The arm's collider is its bar; its
cage is display only and never meets another moving part.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import (ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions,
                                      CanonicalOptions, Expectations, JointDef, MechanismSpec,
                                      MeshSpec, SamplingOptions)

_DIR = "manifold_charts/atlas_rom/assets/circle_arm/sim"
_SCALE = 1.0 / 12.0
_Z = [0.0, 0.0, 1.0]
_C = [0.0, 0.0, 0.0]

_COLORS = {"arm": (0.43, 0.73, 0.46), "pink": (0.91, 0.60, 0.75), "blue": (0.54, 0.71, 0.90),
           "yellow": (0.91, 0.81, 0.24)}

_GROUND = BodyDef(name="__ground", mesh=MeshSpec("sphere", {"radius": 0.3, "center": [0.0, 0.0, 8.0]}),
                  fixed=True, color=(0.45, 0.47, 0.50))


def _body(name):
    return BodyDef(name=name, mesh=MeshSpec("file", {"path": f"{_DIR}/{name}.stl", "scale": _SCALE}),
                   color=_COLORS.get(name, (0.65, 0.85, 0.63)))


_MOVING = ["arm", "pink", "blue", "yellow", "coupling_A", "coupling_B", "output_A", "output_B"]

CASE = AtlasCaseSpec(
    name="circle_arm_contact",
    mechanism=MechanismSpec(
        bodies=[_GROUND] + [_body(name) for name in _MOVING],
        joints=[
            JointDef("arm_hinge", "revolute", "__ground", "arm", pivot=_C, axis=_Z),
            JointDef("circle_hinge", "revolute", "__ground", "pink", pivot=_C, axis=_Z),
        ],
        up_axis="z",
        gravity=0.0,
    ),
    seeds=[],
    contact_model="maintained_contact",
    coordinates=[],
    actuators=[
        ActuatorDef("arm crank torque", kind="joint_torque", target="arm_hinge", hi=40.0),
        ActuatorDef("circle motor torque", kind="joint_torque", target="circle_hinge", hi=40.0),
    ],
    build=BuildOptions(
        # The oracle's default 0.0625-unit voxel is 0.75 mm, two per ring clearance.
        # Halving both (0.375 mm voxels, 0.75 mm clearance, together with a halved
        # step cap) did not sharpen the probe rank gap and cost 5x the memory and
        # build time.
        #
        # The front measures in the mass metric, so step and probe sizes are RMS
        # material displacement over the mechanism, in units (12 mm).  In the
        # packed metric a rigid turn about the motor axis is nearly invisible at
        # rest, and the frame read rank 1 there.  Generic probes push every body
        # by tangent_eps; aimed re-measurement along a certified frame uses the
        # smaller aimed_probe_scale.  Step 3.0 is ~8 deg of arm swing or ~36 deg
        # of crank turn: the arm moves far more mass per degree.
        sampling=SamplingOptions(sampler="contact", oracle_backend="superdex",
                                 ambient_metric="mass", tangent_eps=12.0, generic_probe_count=16,
                                 aimed_probe_scale=2.5, aimed_random_count=3,
                                 oracle_workers=4, keep_disagreeing_frames=True, step=3.0,
                                 # the 80 mm cranks put blue and yellow at 1.93M cells, near
                                 # the 2.1M default; a step moves no material point more than
                                 # half the 1.5 mm ring clearance (no continuous collision)
                                 oracle_options={"max_sdf_cells": 3_000_000, "max_point_step": 0.0625,
                                                 "max_steps": 3000},
                                 n_samples=None, max_seconds=18000., checkpoint_seconds=60.),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True),
        expect=Expectations(n_bodies=8),
    ),
)
