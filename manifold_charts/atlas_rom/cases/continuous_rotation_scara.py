"""Compact four-DOF SCARA with three continuous bearings and a vertical slide."""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, ChartOptions, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances


_Z = (0.0, 0.0, 1.0)
_L1, _L2 = .72, .58


CASE = AtlasCaseSpec(
    name="continuous_rotation_scara",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef("__ground", MeshSpec("cylinder", {"radius": .16, "height": .12, "axis": _Z}),
                    fixed=True, density=220.0, color=(.30, .32, .35)),
            BodyDef("shoulder_link", MeshSpec("link", {"p0": (0., 0., 0.), "p1": (_L1, 0., 0.), "thick": .07}),
                    density=220.0, color=(.90, .52, .15)),
            BodyDef("elbow_link", MeshSpec("link", {"p0": (_L1, 0., 0.), "p1": (_L1 + _L2, 0., 0.), "thick": .06}),
                    density=220.0, color=(.25, .55, .90)),
            BodyDef("wrist", MeshSpec("cylinder", {"center": (_L1 + _L2, 0., 0.), "radius": .11,
                                                       "height": .16, "axis": _Z}),
                    density=220.0, color=(.32, .76, .48)),
            BodyDef("vertical_tool", MeshSpec("compound", {"parts": [
                MeshSpec("cylinder", {"center": (_L1 + _L2, 0., -.18), "radius": .045,
                                      "height": .36, "axis": _Z}),
                MeshSpec("sphere", {"center": (_L1 + _L2, 0., -.37), "radius": .08, "subdiv": 2}),
            ]}), density=220.0, color=(.82, .32, .28)),
        ],
        joints=[
            JointDef("shoulder", "revolute", "shoulder_link", "__ground", pivot=(0., 0., 0.), axis=_Z),
            JointDef("elbow", "revolute", "elbow_link", "shoulder_link", pivot=(_L1, 0., 0.), axis=_Z),
            JointDef("wrist_spin", "revolute", "wrist", "elbow_link", pivot=(_L1 + _L2, 0., 0.), axis=_Z),
            JointDef("vertical_slide", "prismatic", "vertical_tool", "wrist", pivot=(_L1 + _L2, 0., 0.),
                     axis=_Z, limit_lo=-.32, limit_hi=.32),
        ],
        up_axis="z", gravity=0.0,
    ),
    contact_model="none",
    coordinates=[],
    actuators=[
        ActuatorDef("shoulder torque", kind="joint_torque", target="shoulder", hi=20.0),
        ActuatorDef("elbow torque", kind="joint_torque", target="elbow", hi=20.0),
        ActuatorDef("wrist torque", kind="joint_torque", target="wrist_spin", hi=12.0),
    ],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=700, step=.24, tangent_eps=.025, max_seconds=600.),
        charts=ChartOptions(mode="mesh_atlas", overlap=.25),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
        decoder=DecoderOptions(epochs=2500, smooth_lambda=1e-4),
        dynamics=DynamicsOptions(distill=True, epochs=1200, n_teacher=600),
        expect=Expectations(n_bodies=4),
        tolerances=Tolerances(decoder_translation_rmse=.06, decoder_rotation_deg=3.0),
    ),
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "direct_constrained"}},
)

# Evaluator-only statement; this never enters the sampled evidence or runtime.
EVALUATOR_REFERENCE = {"configuration_space": "T^3 x I", "intrinsic_dimension": 4,
                       "continuous_bearings": ("shoulder", "elbow", "wrist_spin"),
                       "bounded_joint": "vertical_slide"}
