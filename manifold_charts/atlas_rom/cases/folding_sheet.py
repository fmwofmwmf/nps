"""A small rigid-panel folding sheet.

Eight square panels form a strip in the flat rest pose.  Adjacent panels are
connected by revolute creases, so the declarative mechanism exposes the seven
independent fold angles and keeps the sheet's geometry/hinge layout explicit.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

PANEL_HALF = [0.5, 0.5, 0.025]
PANEL_X = [i - 3.5 for i in range(8)]
Y = [0.0, 1.0, 0.0]                 # crease axes in the flat sheet


def _panel(i):
    return BodyDef(name=f"panel_{i}",
                   mesh=MeshSpec("box", {"center": [0, 0, 0],
                                          "half": PANEL_HALF}),
                   initial_transform=[[1, 0, 0, PANEL_X[i]],
                                      [0, 1, 0, 0], [0, 0, 1, 0],
                                      [0, 0, 0, 1]],
                   color=(0.25 + 0.12 * (i % 2), 0.55, 0.82))


CASE = AtlasCaseSpec(
    name="folding_sheet",
    mechanism=MechanismSpec(
        bodies=[_panel(i) for i in range(8)],
        joints=[JointDef(f"crease_{i:02d}{i + 1:02d}", "revolute",
                         f"panel_{i}", f"panel_{i + 1}",
                         pivot=[-3.0 + i, 0.0, 0.0], axis=Y)
                for i in range(7)],
        up_axis="z", render_up_axis="z", gravity=9.8,
    ),
    seeds=[], contact_model="none",
    coordinates=[
        CoordinateDef(f"fold_{i:02d}{i + 1:02d}", topology="bounded",
                      lo=-1.57, hi=1.57, source="joint_angle",
                      args={"joint": f"crease_{i:02d}{i + 1:02d}"})
        for i in range(7)
    ],
    actuators=[ActuatorDef(f"crease {i:02d}{i + 1:02d} torque",
                           kind="joint_torque",
                           target=f"crease_{i:02d}{i + 1:02d}", hi=5.0)
               for i in range(7)],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=3000, step=0.05),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=5000, smooth_lambda=1e-3,
                               closure_weight=1.0),
        dynamics=DynamicsOptions(distill=True, epochs=2500, n_teacher=1500),
        expect=Expectations(dimension=7, n_bodies=8),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2),
    ),
    scenarios={"default": {"dt": 1e-3, "initial_u": [0.0] * 7}},
)
