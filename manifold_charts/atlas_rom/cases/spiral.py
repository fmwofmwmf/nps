"""Atlas case for the spiral telescoping actuator.

The tube is constrained by a Y-axis cylindrical joint to the fixed wedge.  It
permits axial translation and axial spin, while every transverse translation
and tilt is locked.  Tube--wedge clearance is retained as the physical bound on
that two-dimensional motion.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

_ASSET = "assets/spiral.glb"
_AXIS = [0.0, 1.0, 0.0]
# Both GLB node transforms are baked into their meshes.  This is their shared
# rest-frame axis, expressed in world coordinates before that bake.
_AXIS_POINT = [10.00000095, -6.25812912, 16.0]


CASE = AtlasCaseSpec(
    name="spiral",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef("spiral_tube", MeshSpec("file", {"path": _ASSET,
                                                        "node": "spiral_tube"}),
                    # Unit moving mass: volume is 19.71564076130568.
                    density=0.050721, color=(0.36, 0.62, 0.86)),
            BodyDef("spiral_wedge", MeshSpec("file", {"path": _ASSET,
                                                         "node": "spiral_wedge"}),
                    fixed=True, density=32.381, color=(0.45, 0.47, 0.50)),
        ],
        joints=[
            JointDef("tube_y_cylindrical", "cylindrical", "spiral_wedge", "spiral_tube",
                     pivot=_AXIS_POINT, axis=_AXIS, exclude_collision=False),
        ],
        up_axis="y", gravity=0.0,
        contact_pairs=[("spiral_tube", "spiral_wedge")],
    ),
    seeds=[],
    contact_model="exact_clearance",
    coordinates=[
        CoordinateDef("tube_y", source="body_translation",
                      args={"body": "spiral_tube", "axis": "y"}),
        CoordinateDef("tube_spin", source="body_angle",
                      args={"body": "spiral_tube", "axis": _AXIS,
                            "reference": "first"}),
    ],
    actuators=[
        ActuatorDef("tube axial force", kind="coordinate_force", target="tube_y", hi=20.0),
        ActuatorDef("tube spin torque", kind="coordinate_force", target="tube_spin", hi=20.0),
    ],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=2000, step=0.06, max_seconds=3600.0,
                                  probe_count=4, drive_iters=12, relax_iters=4,
                                  predictor_body_displacement=0.08,
                                  d_hat=0.01, max_kappa=1e9, newton_max_iter=64,
                                  actuated_bodies=["spiral_tube"]),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=1500, smooth_lambda=1e-3),
        dynamics=DynamicsOptions(distill=True, epochs=1500, n_teacher=512),
        expect=Expectations(dimension=2, n_charts=1, n_bodies=1,
                            contact_pairs=[("spiral_tube", "spiral_wedge")]),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0),
    ),
    scenarios={"default": {"dt": 1e-3, "initial_u": [-6.25812912, 0.0]}},
)
