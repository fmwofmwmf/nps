"""cases/rotate_contact.py - rotate.glb with its links located by SuperDex contact.

The couplers are held only by mesh contact (A-B, B-C, C-D).  The two cranks are
hinged to ground, because no ground geometry exists for them to touch; the axes
are the exact ones in `cases/rotate.py`.
"""
from __future__ import annotations

from atlas_rom.cases.rotate import (_GROUND, _GROUND_A_AXIS, _GROUND_A_PIVOT, _GROUND_D_AXIS,
                                    _GROUND_D_PIVOT)
from atlas_rom.spec.case_spec import (ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions,
                                      CanonicalOptions, Expectations, JointDef, MechanismSpec,
                                      MeshSpec, SamplingOptions)

_ASSET = "old-random-stuff/testing_old/testing/assets/rotate.glb"


def _link(name):
    return BodyDef(name=name, mesh=MeshSpec("file", {"path": _ASSET, "node": f"rotate-{name}"}),
                   color=(0.36, 0.62, 0.86) if name in ("B", "C") else (0.93, 0.53, 0.18))


CASE = AtlasCaseSpec(
    name="rotate_contact",
    mechanism=MechanismSpec(
        bodies=[_GROUND, _link("A"), _link("B"), _link("C"), _link("D")],
        joints=[
            JointDef("crank_A", "revolute", "__ground", "A",
                     pivot=_GROUND_A_PIVOT, axis=_GROUND_A_AXIS),
            JointDef("crank_D", "revolute", "__ground", "D",
                     pivot=_GROUND_D_PIVOT, axis=_GROUND_D_AXIS),
        ],
        up_axis="z",
        gravity=0.0,
    ),
    seeds=[],
    contact_model="maintained_contact",
    coordinates=[],
    actuators=[
        ActuatorDef("crank A torque", kind="joint_torque", target="crank_A", hi=40.0),
        ActuatorDef("crank D torque", kind="joint_torque", target="crank_D", hi=40.0),
    ],
    build=BuildOptions(
        # A random 48-D probe at 2.0 moves the mechanism about 0.8 deg in plane,
        # just clear of the 0.45-0.7 deg joint wriggle.
        sampling=SamplingOptions(sampler="contact", oracle_backend="superdex", tangent_eps=2.0,
                                 oracle_workers=8, keep_disagreeing_frames=True, step=6.0, n_samples=2500, max_seconds=5400.,
                                 checkpoint_seconds=60.),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True),
        expect=Expectations(n_bodies=4),
    ),
)
