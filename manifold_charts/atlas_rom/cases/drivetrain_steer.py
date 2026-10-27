"""Drivetrain on a stand with STEERED driven wheels: the `drivetrain` inline-4
transaxle, its open differential, and a rack-and-tie-rod front end.

  * engine, final drive, carrier and spiders exactly as in `drivetrain`;
  * each side's drive line is split in two at a constant-velocity joint:
      - an inner half-shaft (side gear + shaft) turning in a bearing on the
        stand, as `drivetrain`'s axle did;
      - a wheel stub (CV bell, hub, brake disc, tyre) turning in a knuckle;
      the CV joint sits on the kingpin axis, so the shaft line passes through
      it at every steer angle;
  * each knuckle swivels about a vertical kingpin fixed to the stand and
    carries a steering arm;
  * a rack slides along x on the stand, between end stops, and pulls both
    steering arms through tie rods (a Hooke joint at the rack end keeps each
    rod from spinning idly on its own axis, a ball joint at the arm).

Counting what the mechanism closes: `drivetrain`'s two freedoms (engine
rotation, differential split) plus one more -- steering.  The tie rods make
each knuckle's angle a function of the rack position, and the CV joints carry
each half-shaft's spin to its wheel unchanged, so the wheels spin with their
shafts whatever the steer.

Nothing about that is told to the learner: no coordinates, no dimension, no
topology, no chart count.  `EVALUATOR_REFERENCE` below is read only by tests.

Meshes here are the FUNCTIONAL layer -- closed solids that set mass and inertia.
No render-only display assets have been built for this case.
"""
from __future__ import annotations

import math

from atlas_rom.cases import drivetrain as _base
from atlas_rom.cases._drivetrain_params import (
    ARM_BALL_X, BRAKE_X, CV_RADIUS, FINAL_RATIO, HUB_X, KINGPIN_BOSS_Z, KINGPIN_X, NEG_X,
    RACK_END_X, RACK_TRAVEL, RACK_Y, SHAFT_INNER_X, SHAFT_OUTER_X, SHAFT_RADIUS, SIDE_PHASE,
    STAND_Z, STEER_Z, TYRE_RING, TYRE_TUBE, WHEEL_X, X, Z)
from atlas_rom.cases.drivetrain import STEEL, WHEEL, _bevel, _box, _cyl
from atlas_rom.spec.case_spec import (
    ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, JointDef,
    MechanismSpec, MeshSpec, SamplingOptions)

EVALUATOR_REFERENCE = {
    "configuration_space": "T^2 x I",
    "intrinsic_dimension": 3,
    "gear_law": "theta_crank = -2 theta_carrier; theta_left + theta_right = 2 theta_carrier",
    # With CV joints each wheel's rotation is R_z(steer) R_x(shaft angle): the
    # pose metric and the kinetic metric are both block diagonal, an exact
    # Riemannian product of the drivetrain torus and the steering interval.
    "separable": "T^2 (engine, differential) x I (steering)",
}

Y = (0.0, 1.0, 0.0)
ORIGIN = (0.0, 0.0, 0.0)
SIDES = (("left", -1.0), ("right", 1.0))


# ── drive line: inner half-shaft | CV | wheel stub ───────────────────────────
def _half_shaft(side, sx):
    length = SHAFT_OUTER_X - SHAFT_INNER_X
    return BodyDef(f"axle_{side}", MeshSpec("compound", {"parts": [
        _bevel(X if sx > 0 else NEG_X, SIDE_PHASE, 0.12),
        _cyl(SHAFT_RADIUS, length, (sx * (SHAFT_INNER_X + length / 2), 0., 0.)),
    ]}), density=STEEL, color=(.25, .58, .88) if sx < 0 else (.32, .76, .48))


def _wheel(side, sx):
    stub_len = HUB_X - KINGPIN_X
    return BodyDef(f"wheel_{side}", MeshSpec("compound", {"parts": [
        MeshSpec("sphere", {"radius": CV_RADIUS, "center": (sx * KINGPIN_X, 0., 0.)}),  # CV bell
        _cyl(SHAFT_RADIUS, stub_len, (sx * (KINGPIN_X + stub_len / 2), 0., 0.)),         # stub axle
        _cyl(0.26, 0.10, (sx * HUB_X, 0., 0.)),                                          # hub flange
        _cyl(0.60, 0.07, (sx * BRAKE_X, 0., 0.)),                                        # brake disc
        MeshSpec("torus", {"radius": TYRE_RING, "tube": TYRE_TUBE,
                           "center": (sx * WHEEL_X, 0., 0.), "axis": X}),                # tyre
    ]}), density=WHEEL, color=(.20, .22, .26))


# ── steering ─────────────────────────────────────────────────────────────────
def _knuckle(side, sx):
    plate_x = KINGPIN_X + 0.10
    parts = []
    for z in (-KINGPIN_BOSS_Z, KINGPIN_BOSS_Z):
        parts += [_cyl(0.06, 0.12, (sx * KINGPIN_X, 0., z), axis=Z),                     # kingpin boss
                  _box((sx * (KINGPIN_X + 0.05), 0., z), (0.05, 0.05, 0.04))]            # boss -> plate
    parts += [
        _box((sx * plate_x, 0., 0.), (0.025, 0.09, KINGPIN_BOSS_Z + 0.04)),              # upright
        MeshSpec("link", {"p0": (sx * KINGPIN_X, 0., STEER_Z),
                          "p1": (sx * ARM_BALL_X, RACK_Y, STEER_Z), "thick": 0.07}),     # steering arm
    ]
    return BodyDef(f"knuckle_{side}", MeshSpec("compound", {"parts": parts}),
                   density=STEEL, color=(.55, .57, .60))


_RACK = BodyDef("rack", MeshSpec("compound", {"parts": [
    _cyl(0.035, 2 * RACK_END_X, (0., RACK_Y, STEER_Z)),
    *[MeshSpec("sphere", {"radius": 0.05, "center": (sx * RACK_END_X, RACK_Y, STEER_Z)})
      for _, sx in SIDES],
]}), density=STEEL, color=(.70, .72, .30))


def _tie_rod(side, sx):
    length = ARM_BALL_X - RACK_END_X
    return BodyDef(f"tie_rod_{side}", MeshSpec("compound", {"parts": [
        _cyl(0.025, length, (sx * (RACK_END_X + length / 2), RACK_Y, STEER_Z)),
        MeshSpec("sphere", {"radius": 0.04, "center": (sx * ARM_BALL_X, RACK_Y, STEER_Z)}),
    ]}), density=STEEL, color=(.80, .52, .20))


# ── stand: `drivetrain`'s, plus kingpin posts and the rack housing ───────────
_GROUND = BodyDef("__ground", MeshSpec("compound", {"parts": [
    *_base._GROUND.mesh.params["parts"],
    *[_box((sx * KINGPIN_X, 0., (STAND_Z - KINGPIN_BOSS_Z - 0.06) / 2),
           (0.05, 0.05, (-STAND_Z - KINGPIN_BOSS_Z - 0.06) / 2)) for _, sx in SIDES],
    _box((0., RACK_Y, STEER_Z), (0.90, 0.06, 0.06)),                                    # rack housing
    *[_box((sx * 0.70, RACK_Y, (STAND_Z + STEER_Z - 0.06) / 2),
           (0.05, 0.05, (STEER_Z - 0.06 - STAND_Z) / 2)) for _, sx in SIDES],           # housing posts
]}), fixed=True, color=(.34, .36, .40))


def _side_joints(side, sx):
    kingpin = (sx * KINGPIN_X, 0., 0.)
    return [
        JointDef(f"{side}_bearing", "revolute", f"axle_{side}", "__ground", pivot=ORIGIN, axis=X),
        JointDef(f"kingpin_{side}", "revolute", f"knuckle_{side}", "__ground", pivot=kingpin, axis=Z),
        JointDef(f"hub_{side}", "revolute", f"wheel_{side}", f"knuckle_{side}",
                 pivot=(sx * HUB_X, 0., 0.), axis=X),
        JointDef(f"cv_{side}", "constant_velocity", f"axle_{side}", f"wheel_{side}",
                 pivot=kingpin, axis=X),
        # Hooke pins perpendicular to the rod: no idle spin about its own axis.
        JointDef(f"tie_rod_inner_{side}", "universal", "rack", f"tie_rod_{side}",
                 pivot=(sx * RACK_END_X, RACK_Y, STEER_Z), axis=Z, axis2=Y),
        JointDef(f"tie_rod_outer_{side}", "spherical", f"tie_rod_{side}", f"knuckle_{side}",
                 pivot=(sx * ARM_BALL_X, RACK_Y, STEER_Z)),
    ]


CASE = AtlasCaseSpec(
    name="drivetrain_steer",
    mechanism=MechanismSpec(
        bodies=[_GROUND, _base._crankshaft(), *[_base._rod(i) for i in range(4)],
                *[_base._piston(i) for i in range(4)], _base._carrier(),
                _base._SPIDER, _base._SPIDER_B,
                *[_half_shaft(s, sx) for s, sx in SIDES],
                *[_knuckle(s, sx) for s, sx in SIDES],
                *[_wheel(s, sx) for s, sx in SIDES],
                _RACK, *[_tie_rod(s, sx) for s, sx in SIDES]],
        joints=_base._engine_joints() + [
            JointDef("final_drive", "gear_ratio", "crankshaft", "carrier", body_c="__ground",
                     axis=X, ratio=float(FINAL_RATIO)),
            JointDef("carrier_bearing", "revolute", "carrier", "__ground", pivot=ORIGIN, axis=X),
            JointDef("spider_bearing", "revolute", "spider", "carrier", pivot=ORIGIN, axis=Z),
            JointDef("spider_b_bearing", "revolute", "spider_b", "carrier", pivot=ORIGIN,
                     axis=(0.0, 0.0, -1.0)),
            JointDef("mesh_right", "bevel_gear", "spider", "axle_right", body_c="carrier",
                     axis=Z, axis2=X, ratio=1.0),
            JointDef("mesh_left", "bevel_gear", "spider", "axle_left", body_c="carrier",
                     axis=Z, axis2=X, ratio=-1.0),
            JointDef("mesh_b_right", "bevel_gear", "spider_b", "axle_right", body_c="carrier",
                     axis=(0.0, 0.0, -1.0), axis2=X, ratio=1.0),
            JointDef("rack_slide", "prismatic", "__ground", "rack", pivot=(0., RACK_Y, STEER_Z),
                     axis=X, limit_lo=-RACK_TRAVEL, limit_hi=RACK_TRAVEL),
            *[j for s, sx in SIDES for j in _side_joints(s, sx)],
        ],
        up_axis="z", gravity=0.0,
    ),
    coordinates=[],
    contact_model="none",
    actuators=[
        ActuatorDef("engine torque", kind="joint_torque", target="crank_main", hi=40.0),
        ActuatorDef("left wheel brake", kind="joint_torque", target="left_bearing", hi=40.0),
        # `joint_torque` reads a rotation, so steering is driven at a kingpin,
        # not at the rack's slide.
        ActuatorDef("steering torque", kind="joint_torque", target="kingpin_left", hi=40.0),
    ],
    build=BuildOptions(
        # Provisional: `drivetrain`'s step.  The steering extent in the packed
        # pose metric has not been measured yet; set the step from a measured
        # volume density before running a front.  No sample cap: the frontier
        # must close on its own.
        sampling=SamplingOptions(n_samples=None, step=1.0, max_seconds=1800,
                                 tangent_eps=0.04, tangent_rel_tol=0.20),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
