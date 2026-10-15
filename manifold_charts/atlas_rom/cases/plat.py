"""The original ``plat`` contact-driven linkage, expressed as a generic atlas case.

The GLB already carries each node's rest-world transform; ``MeshSpec(file)``
bakes that transform into the corresponding body mesh, so every body begins at
identity here.  The three pivots are the explicit ideal constraints from
``testing/output/test_plat/plat.json``.  All other coupling is deliberately
left to contact, as in the source scene.
"""
from __future__ import annotations

import math

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

_ASSET = "assets/plat.glb"
_AXIS = [0.0, 1.0, 0.0]
# ``simulate_headless.py`` passes the rig's [0, 0, 0] pivot to
# ``Simulation.pin_rotation``, whose API is body-local.  The atlas loader bakes
# every GLB node transform into its mesh, so the generic joint API needs the
# corresponding *world* points instead.  These are the GLB rest-node origins.
_PIVOT = {
    "plat_driver": [3.1153829097747803, 0.007626569364219904, -1.9984486103057861],
    "plat_g_1": [1.2445554733276367, 0.0, -1.9984142780303955],
    "plat_g_2": [-1.2594289779663086, 0.0, -1.9984142780303955],
}
_NODES = ("plat_fixed", "plat_plat", "plat_bar_2", "plat_bar_1",
          "plat_g_2", "plat_driver", "plat_g_1")
# ``plat.json`` declares mass=1 for every body.  The generic pipeline declares
# density and derives mass from the imported mesh volume, so these are 1/volume.
_UNIT_MASS_DENSITY = {
    "plat_plat": 0.3944416920108838,
    "plat_bar_2": 1.8980349584138915,
    "plat_bar_1": 1.8980349584138915,
    "plat_g_2": 0.8515130770772084,
    "plat_driver": 4.180836599339991,
    "plat_g_1": 0.8515130770772084,
}


def _body(name: str, *, fixed: bool = False) -> BodyDef:
    return BodyDef(name=name, mesh=MeshSpec("file", {"path": _ASSET,
                                                       "node": name}),
                   fixed=fixed, density=_UNIT_MASS_DENSITY.get(name, 1.0))


CASE = AtlasCaseSpec(
    name="plat",
    mechanism=MechanismSpec(
        bodies=[_body(name, fixed=name == "plat_fixed") for name in _NODES],
        joints=[
            JointDef("driver_pivot", "revolute", "plat_fixed", "plat_driver",
                     pivot=_PIVOT["plat_driver"], axis=_AXIS),
            JointDef("gear_1_pivot", "revolute", "plat_fixed", "plat_g_1",
                     pivot=_PIVOT["plat_g_1"], axis=_AXIS),
            JointDef("gear_2_pivot", "revolute", "plat_fixed", "plat_g_2",
                     pivot=_PIVOT["plat_g_2"], axis=_AXIS),
        ],
        up_axis="y", gravity=0.0,
    ),
    seeds=[],
    contact_model="exact_clearance",
    coordinates=[
        CoordinateDef("driver_phase", topology="periodic", period=4.0 * math.pi,
                      source="sample_field",
                      args={"key": "recorded_angle:plat_driver:y"}),
    ],
    # The unwrapped driver angle completes two turns (4*pi) per configuration
    # loop, so it is an injective physical coordinate and torque has unit gain.
    actuators=[ActuatorDef("driver torque", kind="coordinate_force",
                           target="driver_phase", hi=8000.0)],
    build=BuildOptions(
        # The explorer now lifts the driver angle along its continuation tree, so
        # it can construct this case's 4*pi periodic coordinate itself instead
        # of importing the old torque-driven recording for its sample field.
        # The contact stack is stiff: a 0.25-body predictor can request a target
        # beyond IPC's local Newton basin.  Walk it in small, reliable pieces.
        # The explicit hinge predictor commands this physical angle directly.
        # 0.05 rad is outside IPC's local contact basin; 0.02 rad retains a
        # safe continuation margin while requiring about 630 samples per 4*pi
        # driver loop.
        sampling=SamplingOptions(n_samples=700, step=0.02, max_seconds=3600.0,
                                  predictor_body_displacement=0.1,
                                  fault_tolerant=True, continuation_min_step=0.00125,
                                  continuation_replay_window=8,
                                  d_hat=0.005,
                                  hold_drive_during_projection=True,
                                  actuated_bodies=["plat_driver"]),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=3000, smooth_lambda=1e-3),
        dynamics=DynamicsOptions(distill=True, epochs=2500, n_teacher=256),
        expect=Expectations(dimension=1, n_bodies=6),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2),
    ),
    scenarios={"default": {"dt": 1e-3, "initial_u": [0.0],
                           "dynamics_backend": "distilled"}},
)
