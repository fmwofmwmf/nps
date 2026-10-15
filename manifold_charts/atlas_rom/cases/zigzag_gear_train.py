"""Contact-only, non-planetary zigzag compound spur-gear reduction.

All transmission behaviour comes from the generated involute tooth meshes and
the three revolute axle joints.  There is intentionally no prescribed gear
ratio, gear constraint, or analytic follower relation in this case.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

_ASSET = "assets/zigzag_gear_train.glb"


def _gear(name, color):
    return BodyDef(name=name, mesh=MeshSpec("file", {"path": _ASSET, "node": name}),
                   density=8.0, color=color)


def _ground():
    """A fixed affine counterpart for axle torques; it has no contact role."""
    return BodyDef(name="__ground", mesh=MeshSpec("sphere", {"radius": 0.01}),
                   fixed=True, density=1.0, color=(0.18, 0.18, 0.18))


CASE = AtlasCaseSpec(
    name="zigzag_gear_train",
    mechanism=MechanismSpec(
        bodies=[
            _ground(),
            _gear("Input", (0.93, 0.53, 0.18)),
            _gear("Compound", (0.36, 0.62, 0.86)),
            _gear("Output", (0.38, 0.75, 0.45)),
        ],
        joints=[
            JointDef("input_axle", "revolute", "__ground", "Input",
                     pivot=[0.0, 0.0, 0.0],
                     axis=[0.0, 1.0, 0.0]),
            JointDef("compound_axle", "revolute", "__ground", "Compound",
                     pivot=[2.3591578875522963, 0.0, 1.6519001366910127],
                     axis=[0.0, 1.0, 0.0]),
            JointDef("output_axle", "revolute", "__ground", "Output",
                     pivot=[4.7183157751045925, 0.0, 0.0], axis=[0.0, 1.0, 0.0]),
        ],
        up_axis="y",
        gravity=0.0,
        # Only adjacent layers are intended to mesh.  Do not let broad phase
        # invent an Input--Output contact relation across the train.
        contact_pairs=[("Input", "Compound"), ("Compound", "Output")],
    ),
    seeds=[],
    contact_model="maintained_contact",
    coordinates=[
        CoordinateDef(name="input_phase", topology="periodic", period=6.283185307179586,
                      source="body_angle",
                      args={"body": "Input", "axis": [0.0, 1.0, 0.0],
                            "reference": "first"}),
    ],
    actuators=[ActuatorDef("input torque", kind="joint_torque", target="input_axle", hi=40.0)],
    build=BuildOptions(
        sampling=SamplingOptions(
            n_samples=1000,
            # The meshed rest gap is ~0.00603, so this activates contact with
            # ~0.004 clearance margin rather than requiring initial overlap.
            d_hat=0.010,
            max_kappa=1e9,
            newton_max_iter=64,
        ),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=6000, n_harmonics=3, smooth_lambda=1e-4),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=720),
        expect=Expectations(dimension=1, n_charts=1, n_bodies=3),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=2.0),
    ),
    scenarios={"default": {"dt": 1e-3, "initial_u": [0.0]}},
)
