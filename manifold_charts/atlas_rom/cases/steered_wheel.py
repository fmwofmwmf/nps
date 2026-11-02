"""One steered, driven wheel: a half-shaft turning in a bearing, a knuckle
swivelling on a kingpin between stops, and a wheel turning in the knuckle,
driven from the shaft through a constant-velocity joint on the kingpin axis.

It is one corner of `drivetrain_steer` on its own: the smallest mechanism with
a body that moves with spin alone (the shaft), one that moves with steer alone
(the knuckle), and one that moves with both (the wheel).

Nothing about its freedoms is told to the learner: no coordinates, no
dimension, no topology, no chart count.  `EVALUATOR_REFERENCE` below is read
only by tests.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import (
    AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, JointDef, MechanismSpec, MeshSpec,
    SamplingOptions)

EVALUATOR_REFERENCE = {
    "configuration_space": "S^1 x I",
    "intrinsic_dimension": 2,
    "separable": "S^1 (wheel spin) x I (steer)",
}

X, Z = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
KINGPIN_X = 1.0                    # kingpin axis and CV centre
HUB_X, WHEEL_X = 1.20, 1.40
STEER_LIMIT = 0.35                 # rad each way, ~20 deg
STEEL, WHEEL = 120.0, 30.0


def _cyl(radius, height, center, axis=X):
    return MeshSpec("cylinder", {"radius": radius, "height": height, "center": center, "axis": axis})


def _box(center, half):
    return MeshSpec("box", {"center": center, "half": half})


_GROUND = BodyDef("__ground", MeshSpec("compound", {"parts": [
    _box((0.0, 0.0, -0.9), (1.2, 0.5, 0.04)),                                   # base plate
    _box((0.35, 0.0, -0.45), (0.05, 0.05, 0.43)),                               # shaft pedestal
    _box((KINGPIN_X, 0.0, -0.62), (0.05, 0.05, 0.26)),                          # kingpin post
]}), fixed=True, color=(.34, .36, .40))

_SHAFT = BodyDef("shaft", MeshSpec("compound", {"parts": [
    _cyl(0.05, KINGPIN_X - 0.30, ((KINGPIN_X + 0.10) / 2 + 0.06, 0.0, 0.0)),
    _cyl(0.12, 0.04, (0.20, 0.0, 0.0)),                                         # drive flange
]}), density=STEEL, color=(.25, .58, .88))

_KNUCKLE = BodyDef("knuckle", MeshSpec("compound", {"parts": [
    _cyl(0.06, 0.12, (KINGPIN_X, 0.0, 0.30), axis=Z),                           # upper boss
    _cyl(0.06, 0.12, (KINGPIN_X, 0.0, -0.30), axis=Z),                          # lower boss
    _box((KINGPIN_X + 0.10, 0.0, 0.0), (0.025, 0.09, 0.34)),                    # upright
    _box((KINGPIN_X + 0.05, 0.0, 0.30), (0.05, 0.05, 0.04)),
    _box((KINGPIN_X + 0.05, 0.0, -0.30), (0.05, 0.05, 0.04)),
    MeshSpec("link", {"p0": (KINGPIN_X, 0.0, -0.30), "p1": (KINGPIN_X - 0.12, -0.45, -0.30),
                      "thick": 0.06}),                                           # steering arm
]}), density=STEEL, color=(.55, .57, .60))

_WHEEL = BodyDef("wheel", MeshSpec("compound", {"parts": [
    MeshSpec("sphere", {"radius": 0.10, "center": (KINGPIN_X, 0.0, 0.0)}),       # CV bell
    _cyl(0.05, HUB_X - KINGPIN_X + 0.05, ((KINGPIN_X + HUB_X) / 2 + 0.025, 0.0, 0.0)),
    _cyl(0.18, 0.06, (HUB_X, 0.0, 0.0)),                                         # hub flange
    MeshSpec("torus", {"radius": 0.42, "tube": 0.12, "center": (WHEEL_X, 0.0, 0.0), "axis": X}),
]}), density=WHEEL, color=(.20, .22, .26))

CASE = AtlasCaseSpec(
    name="steered_wheel",
    mechanism=MechanismSpec(
        bodies=[_GROUND, _SHAFT, _KNUCKLE, _WHEEL],
        joints=[
            JointDef("shaft_bearing", "revolute", "shaft", "__ground", pivot=(0.0, 0.0, 0.0), axis=X),
            JointDef("kingpin", "revolute", "knuckle", "__ground", pivot=(KINGPIN_X, 0.0, 0.0), axis=Z,
                     limit_lo=-STEER_LIMIT, limit_hi=STEER_LIMIT),
            JointDef("hub", "revolute", "wheel", "knuckle", pivot=(HUB_X, 0.0, 0.0), axis=X),
            JointDef("cv", "constant_velocity", "shaft", "wheel", pivot=(KINGPIN_X, 0.0, 0.0), axis=X),
        ],
        up_axis="z", gravity=0.0,
    ),
    coordinates=[],
    contact_model="none",
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=None, step=0.4, max_seconds=600,
                                 tangent_eps=0.02, tangent_rel_tol=0.20),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True, autoencoder_charts=True),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
