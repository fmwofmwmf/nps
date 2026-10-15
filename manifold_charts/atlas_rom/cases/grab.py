"""The recorded ``grab`` mechanism, migrated from ``output/test_grab/grab.json``.

The source JSON records a 38-part, contact-assembled grabber in ``spher_1.glb``.
Its two fixed parts are retained as static context; the other 36 meshes are the
decoder pose slots.  Rather than re-exploring this tightly coupled assembly,
the atlas adopts the 154 recorded frames from the legacy case.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

_ASSET = "assets/spher_1.glb"
_FIXED = {"Fixed 1", "Fixed 2"}

# The legacy simulation assigned mass=1 to every mesh.  Atlas cases declare
# density, so these are reciprocal imported volumes and preserve those masses.
_UNIT_MASS_DENSITY = {
    "adad": 5.26670730157307,
    "Fixed 2": 1.75978754889165,
    "Fixed 1": 1.72440480058292,
    "Pin 2": 10.7399632441052,
    "Pin 1": 16.5300095508601,
    "Body_1_35_Mtl14_0": 4.02903407692239,
    "Driver": 3.81438081283507,
    "Body_1_33_Mtl8_0": 16.3960783352373,
    "Body_1_32_Mtla_0": 15.3423940739536,
    "Body_1_31_Mtla_0": 15.3549597376366,
    "Body_1_30_Mtl8_0": 16.3942561531068,
    "Body_1_29_Mtl10_0": 2530.48551001907,
    "Body_1_28_Mtlb_0": 1849.78240535448,
    "Body_1_27_Mtl7_0": 114.683138615192,
    "Body_1_26_Mtl10_0": 2197.59074540763,
    "Body_1_25_Mtlb_0": 1638.95879999362,
    "Body_1_24_Mtl7_0": 114.683153633619,
    "Body_1_23_Mtl9_0": 147.014427292799,
    "Body_1_22_Mtl9_0": 147.014375164084,
    "Body_1_21_Mtla_0": 8.18809419953336,
    "Body_1_20_Mtl9_0": 147.014267762434,
    "Body_1_19_Mtl9_0": 147.014310658249,
    "Body_1_18_Mtla_0": 8.18506689204053,
    "Body_1_17_Mtl9_0": 147.0144265168,
    "Body_1_16_Mtl9_0": 147.014279940693,
    "Body_1_15_Mtl8_0": 8.76948668690222,
    "Body_1_14_Mtl9_0": 147.014351544972,
    "Body_1_13_Mtl9_0": 147.014276685717,
    "Body_1_12_Mtl8_0": 8.76394349840634,
    "Body_1_11_Mtl7_0": 124.012513346819,
    "Body_1_10_Mtl6_0": 9.60770980307588,
    "Body_1_9_Mtl7_0": 127.325785392661,
    "Body_1_8_Mtl6_0": 9.64370944341414,
    "Body_1_7_Mtl3_0": 755.99839696666,
    "Body_1_6_Mtl5_0": 7.65338545577598,
    "Body_1_4_Mtl3_0": 667.488153106093,
    "Body_1_3_Mtl2_0": 8.27236916611915,
    "Body_1_2_Mtl1_0": 6.52407754623909,
}


def _body(name: str) -> BodyDef:
    return BodyDef(name=name, mesh=MeshSpec("file", {"path": _ASSET, "node": name}),
                   fixed=name in _FIXED, density=_UNIT_MASS_DENSITY[name])


CASE = AtlasCaseSpec(
    name="grab",
    mechanism=MechanismSpec(
        bodies=[_body(name) for name in _UNIT_MASS_DENSITY],
        joints=[],
        up_axis="z", gravity=5.0,
    ),
    seeds=[],
    contact_model="exact_clearance",
    coordinates=[
        CoordinateDef("driver_phase", topology="bounded", lo=-0.155, hi=0.155,
                      source="sample_field",
                      args={"key": "recorded_angle:Driver:z"}),
    ],
    actuators=[
        # The recorded coordinate is the driver's z-axis angle, so generalized
        # force is the source driver's applied z-axis torque.
        ActuatorDef("driver torque", kind="coordinate_force",
                    target="driver_phase", hi=50000.0),
    ],
    build=BuildOptions(
        sampling=SamplingOptions(reuse="../output/test_grab/grab.npz"),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=3000, hidden_width=128, hidden_layers=3,
                               smooth_lambda=1.0),
        dynamics=DynamicsOptions(distill=True, epochs=3000),
        expect=Expectations(dimension=1, n_charts=1, n_bodies=36),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0),
    ),
    scenarios={"default": {"dt": 0.01}},
)
