"""The ``gears_1`` planetary train, migrated to the generic Atlas ROM build.

The source chart was recorded by the prior gear explorer.  Its config stack also
contains the fixed ring gear; the generic sampler uses the recorded ``body_order``
metadata to retain only Hand, Gear 1, and Gear 2 while keeping Gear 3 as a static
contact body in the mechanism.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

_ASSET = "assets/gears_1.glb"

# ``gears_1.glb`` is authored at display scale.  The established analytic gear
# model uses unit mass for each moving body (and derives its inertia directly
# from the mesh), so its carrier torque is in a useful N m range.  These are
# 1 / volume for the watertight imported meshes: density * volume == 1.0.
# Using the global ABD default (1000) made the reduced inertia ~13,000 and a
# 40 N m hand torque effectively imperceptible.
_UNIT_MASS_DENSITY = {
    "Hand": 1.4983528804960107,
    "Gear 3": 0.5926648790562989,
    "Gear 1": 0.8580692081636363,
    "Gear 2": 4.180838956280251,
}


def _gear_body(name, *, fixed=False, color=None):
    return BodyDef(name=name, mesh=MeshSpec("file", {"path": _ASSET, "node": name}),
                   fixed=fixed, density=_UNIT_MASS_DENSITY[name], color=color)


CASE = AtlasCaseSpec(
    name="gears_1",
    mechanism=MechanismSpec(
        bodies=[
            _gear_body("Hand", color=(0.93, 0.53, 0.18)),
            _gear_body("Gear 3", fixed=True, color=(0.45, 0.47, 0.50)),
            _gear_body("Gear 1", color=(0.36, 0.62, 0.86)),
            _gear_body("Gear 2", color=(0.36, 0.62, 0.86)),
        ],
        joints=[JointDef("pin-Hand", "revolute", "Hand", pivot=[0, 0, 0],
                         axis=[0, 1, 0])],
        up_axis="y",
        gravity=0.0,
    ),
    seeds=[],
    contact_model="maintained_contact",
    coordinates=[
        CoordinateDef(name="phase", topology="periodic", period=6.283185307179586,
                      source="body_angle",
                      args={"body": "Hand", "axis": [0, 1, 0], "reference": "first"}),
    ],
    actuators=[ActuatorDef("hand torque", kind="joint_torque", target="pin-Hand", hi=40.0)],
    build=BuildOptions(
        sampling=SamplingOptions(reuse="output/gear_loop_chart.npz"),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=6000, n_harmonics=3, smooth_lambda=1e-4),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=720),
        expect=Expectations(dimension=1, n_charts=1, n_bodies=3),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=2.0),
    ),
    scenarios={"default": {"dt": 1e-3, "initial_u": [0.0]}},
)
