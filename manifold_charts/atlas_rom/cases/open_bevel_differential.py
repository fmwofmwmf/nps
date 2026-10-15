"""Open bevel differential — a carrier, two bevel side gears and a spider pinion.

The carrier turns on the main axis; each side gear turns on that same axis; the
spider pinion turns on a radial axle carried by the carrier and rolls on BOTH
side gears at once.  The two rolling meshes are the only constraints across four
continuous bearings, leaving a connected two-dimensional configuration manifold
(the familiar ``theta_left + theta_right = 2 theta_carrier`` gear law falls out
of the pair).  The evaluator records its ``T^2`` reference separately; no
coordinate, topology, or chart count is supplied to the learned atlas.

A second, cosmetic spider is welded into the carrier mesh so the section reads
like a real differential; only the one driven spider actually rolls.

Meshes are drawn as a genuine conjugate bevel set: every pitch cone shares its
apex at the centre, the pitch angles sum to 90 deg, and the tooth counts equal
the analytic gear ratio, so the pinion advances exactly one tooth-pitch per
side-gear tooth-pitch and the tooth engagement is the same at every pose on the
manifold (clean at rest => clean everywhere).
"""
from __future__ import annotations

import math as _math

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, ChartOptions, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances


EVALUATOR_REFERENCE = {
    "configuration_space": "T^2",
    "intrinsic_dimension": 2,
    "gear_law": "theta_left + theta_right = 2 theta_carrier",
    "assumptions": ("equal side-gear tooth counts", "connected regular component"),
}

_Z = (0.0, 0.0, 1.0)
_X = (1.0, 0.0, 0.0)

# Conjugate bevel pair, crossed axes: pitch angles sum to 90 deg and the tooth
# counts are in the analytic gear ratio, so one tooth-pitch of the side gear is
# exactly one tooth-pitch of the pinion and the mesh never drifts pose to pose.
_N_SIDE, _N_PINION = 16, 12
_RATIO = _N_SIDE / _N_PINION                              # = 4/3, also = tan(gamma_side)
_GAMMA_SIDE = _math.degrees(_math.atan2(_N_SIDE, _N_PINION))    # ~53.13 deg off the main axis
_GAMMA_PINION = 90.0 - _GAMMA_SIDE                              # ~36.87 deg off the cross-pin
_SIDE_PHASE = _math.pi / _N_SIDE                                # gap facing the pinion
# The pinion only spans the mesh zone; the side gear runs on out to its back.
# `clearance` pulls each blank a hair off the shared pitch cone so the block
# teeth interleave with real gaps instead of grazing (a ~0.3 deg conjugacy
# error over the face -- invisible, teeth still register pose to pose).
_PIN = dict(pitch_angle=_GAMMA_PINION, toe=0.34, heel=0.60, teeth=_N_PINION,
            tooth_phase=0.0, addendum=.016, backlash=.25, clearance=.045,
            hub_radius=.075, hub_len=.14)
_SIDE = dict(pitch_angle=_GAMMA_SIDE, toe=0.34, heel=0.80, teeth=_N_SIDE,
             tooth_phase=_SIDE_PHASE, addendum=.020, backlash=.25, clearance=.02,
             hub_radius=.12, hub_len=.24)


def _body(name, mesh, color):
    return BodyDef(name=name, mesh=mesh, density=80.0, color=color)


_GROUND = BodyDef("__ground", MeshSpec("cylinder", {"radius": .055, "height": 2.7,
                                                     "axis": _Z}), fixed=True,
                  density=80.0, color=(.34, .36, .40))

# An open cage: two end hoops around the side-gear hubs, two side bars carrying
# the pinion cross-shaft, and the ring-gear flange bolted OUTBOARD of one hoop
# (not a plate through the centre).
_CARRIER = _body("carrier", MeshSpec("compound", {"parts": [
    MeshSpec("torus", {"radius": .24, "tube": .05, "center": (0., 0., .92), "axis": _Z}),        # end rings
    MeshSpec("torus", {"radius": .24, "tube": .05, "center": (0., 0., -.92), "axis": _Z}),
    MeshSpec("box", {"center": (-.84, 0., 0.), "half": (.05, .05, .94)}),                        # side bars
    MeshSpec("box", {"center": (.84, 0., 0.), "half": (.05, .05, .94)}),
    MeshSpec("cylinder", {"radius": .045, "height": 1.74, "center": (0., 0., 0.), "axis": _X}),  # pinion cross-shaft
    MeshSpec("torus", {"radius": .82, "tube": .06, "center": (0., 0., -1.02), "axis": _Z}),      # ring-gear rim, outboard
    MeshSpec("cylinder", {"radius": .84, "height": .04, "center": (0., 0., -1.02), "axis": _Z}), # ring-gear web
    MeshSpec("bevel_gear", {**_PIN, "center": (0., 0., 0.), "axis": _X, "flip": True}),          # cosmetic pinion (-x)
]}), (.90, .52, .15))

# Every pitch cone shares its apex at the differential centre (origin); the wide
# toothed end and the hub face outboard toward the shaft.
_LEFT = _body("left_side_gear",
              MeshSpec("bevel_gear", {**_SIDE, "center": (0., 0., 0.), "axis": _Z, "flip": False}),
              (.25, .58, .88))
_RIGHT = _body("right_side_gear",
               MeshSpec("bevel_gear", {**_SIDE, "center": (0., 0., 0.), "axis": _Z, "flip": True}),
               (.32, .76, .48))
_SPIDER = _body("spider",
                MeshSpec("bevel_gear", {**_PIN, "center": (0., 0., 0.), "axis": _X, "flip": False}),
                (.88, .30, .34))

CASE = AtlasCaseSpec(
    name="open_bevel_differential",
    mechanism=MechanismSpec(
        bodies=[_GROUND, _CARRIER, _LEFT, _RIGHT, _SPIDER],
        joints=[
            JointDef("carrier_bearing", "revolute", "carrier", "__ground", pivot=(0., 0., 0.), axis=_Z),
            JointDef("left_bearing", "revolute", "left_side_gear", "__ground", pivot=(0., 0., 0.), axis=_Z),
            JointDef("right_bearing", "revolute", "right_side_gear", "__ground", pivot=(0., 0., 0.), axis=_Z),
            JointDef("spider_bearing", "revolute", "spider", "carrier", pivot=(0., 0., 0.), axis=_X),
            JointDef("mesh_left", "bevel_gear", "spider", "left_side_gear",
                     body_c="carrier", axis=_X, axis2=_Z, ratio=_RATIO),
            JointDef("mesh_right", "bevel_gear", "spider", "right_side_gear",
                     body_c="carrier", axis=_X, axis2=_Z, ratio=-_RATIO),
        ],
        up_axis="z", gravity=0.0,
    ),
    coordinates=[],
    contact_model="none",
    actuators=[ActuatorDef("carrier torque", kind="joint_torque", target="carrier_bearing", hi=20.0)],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=None, step=.73868232, tangent_eps=.03,
                                 rank_gap_ratio=100., max_seconds=None,
                                 max_probes_per_node=4),
        charts=ChartOptions(mode="mesh_atlas", overlap=.25),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
        decoder=DecoderOptions(epochs=3000, smooth_lambda=1e-4),
        dynamics=DynamicsOptions(distill=True, epochs=1500, n_teacher=900),
        expect=Expectations(n_bodies=4),
        tolerances=Tolerances(decoder_translation_rmse=.05, decoder_rotation_deg=3.0),
    ),
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "exact_canonical"}},
)
