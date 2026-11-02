"""Drivetrain on a stand with STEERED driven wheels: the `drivetrain_v8`
front-mounted 90-degree V8, its bevel final drive and open differential, and
the rack-and-tie-rod front end of `drivetrain_steer`.

  * engine, bevel final drive, carrier and spiders exactly as in
    `drivetrain_v8`;
  * each side's drive line is split in two at a constant-velocity joint, as in
    `drivetrain_steer`:
      - an inner half-shaft (the V8's side gear + shaft) turning in a bearing
        on the stand;
      - a wheel stub (CV bell, hub, brake disc, tyre) turning in a knuckle;
      the CV joint sits on the kingpin axis;
  * knuckles on vertical kingpins, a rack between end stops, and tie rods,
    all taken unchanged from `drivetrain_steer`;
  * a left-hand-drive steering column laid out like a car's: the wheel on an
    upper column in a bearing on the dash crossbar, an intermediate shaft
    with a Hooke joint at each end, and a pinion shaft in the rack housing
    driving the rack (`rack_pinion`: rack travel = pitch radius x pinion
    angle).  The column moves with steering alone;
  * the stand is a car's front body frame (rails, crossmembers, dash
    crossbar, lower arms) instead of a plate.

Counting what the mechanism closes: `drivetrain_v8`'s two freedoms (engine
rotation, differential split) plus steering.

Nothing about that is told to the learner: no coordinates, no dimension, no
topology, no chart count.  `EVALUATOR_REFERENCE` below is read only by tests.

Meshes here are the FUNCTIONAL layer -- closed solids that set mass and inertia.
No render-only display assets have been built for this case.
"""
from __future__ import annotations

import math

from atlas_rom.cases import drivetrain_steer as _steer
from atlas_rom.cases import drivetrain_v8 as _v8
from atlas_rom.cases._drivetrain_params import (
    NEG_X, NEG_Y, SHAFT_OUTER_X, SHAFT_RADIUS, V8_FINAL_RATIO, V8_SHAFT_INNER_X, X, Y)
from atlas_rom.cases.drivetrain import STEEL, _box, _cyl
from atlas_rom.spec.case_spec import (
    ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, JointDef,
    MechanismSpec, MeshSpec, SamplingOptions)

EVALUATOR_REFERENCE = {
    "configuration_space": "T^2 x I",
    "intrinsic_dimension": 3,
    "gear_law": "theta_crank(y) = -2 theta_carrier(x); theta_left + theta_right = 2 theta_carrier",
    "separable": "T^2 (engine, differential) x I (steering)",
}

ORIGIN = _v8.ORIGIN
SIDES = _steer.SIDES

# ── steering: left-hand drive, laid out like a car's ────────────────────────
# One unit ~= 0.25 m.  The wheel sits on the driver's side, ~1.4 m behind the
# front axle and ~0.9 m above the road (tyre contact at z ~= -1.29), on an
# upper column tilted 25 deg from horizontal.  A pinion shaft turns in the
# rack housing at the driver's end of the rack; an intermediate shaft with a
# Hooke joint at each end carries the column's turn down to it, bending ~16
# and ~13 deg.  Steering stays one freedom: the column's angle follows the
# pinion's (with the usual Hooke wobble), and the pinion's follows the rack.
PINION_PITCH = 0.06                 # rack +-0.15 -> pinion +-2.5 rad (inside +-pi)
PINION_LEN = 0.55
COLUMN_LEN = 2.4                    # upper column, U-joint to wheel centre


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _add(p, v, t=1.0):
    return tuple(a + t * b for a, b in zip(p, v))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


_tilt = math.radians(25.0)
COLUMN_AXIS = (0.0, -math.cos(_tilt), math.sin(_tilt))       # toward the driver, rising
WHEEL_CENTER = (-1.5, -5.5, 2.2)
PINION_CENTER = (-0.9, _steer.RACK_Y, _steer.STEER_Z)        # on the rack line
PINION_AXIS = _unit((-0.15, -0.7, 0.7))
U_LOWER = _add(PINION_CENTER, PINION_AXIS, PINION_LEN)        # pinion shaft / intermediate
U_UPPER = _add(WHEEL_CENTER, COLUMN_AXIS, -COLUMN_LEN)       # intermediate / column
_dx = tuple(b - a for a, b in zip(U_LOWER, U_UPPER))
INTER_AXIS = _unit(_dx)
INTER_LEN = math.sqrt(sum(c * c for c in _dx))
# Hooke crosses: the outer shaft's pin is normal to the bend plane, the
# intermediate shaft's lies in it, so the intermediate's two yokes are
# (nearly) in phase.
_N_LOWER = _unit(_cross(PINION_AXIS, INTER_AXIS))
_N_UPPER = _unit(_cross(INTER_AXIS, COLUMN_AXIS))
COWL_Y = -4.5                       # dash crossbar; the upper column bearing hangs from it
COLUMN_BEARING = _add(U_UPPER, COLUMN_AXIS, (COWL_Y - U_UPPER[1]) / COLUMN_AXIS[1])
RIM_RADIUS = 0.75                   # a 37 cm wheel


def _pinion_shaft():
    return BodyDef("pinion_shaft", MeshSpec("compound", {"parts": [
        MeshSpec("spur_gear", {"pitch_radius": PINION_PITCH, "teeth": 12, "width": 0.10,
                               "center": PINION_CENTER, "axis": PINION_AXIS}),        # pinion
        _cyl(0.03, PINION_LEN, _add(PINION_CENTER, PINION_AXIS, PINION_LEN / 2), axis=PINION_AXIS),
        MeshSpec("sphere", {"radius": 0.05, "center": U_LOWER}),                       # lower cross
    ]}), density=STEEL, color=(.62, .64, .68))


def _intermediate_shaft():
    return BodyDef("intermediate_shaft", MeshSpec("compound", {"parts": [
        _cyl(0.028, INTER_LEN, _add(U_LOWER, INTER_AXIS, INTER_LEN / 2), axis=INTER_AXIS),
    ]}), density=STEEL, color=(.62, .64, .68))


def _column():
    e1 = (1.0, 0.0, 0.0)                                       # rim plane, perpendicular to the column
    e2 = (0.0, math.sin(_tilt), math.cos(_tilt))
    spokes = []
    for deg in (270.0, 30.0, 150.0):                           # three spokes, one pointing down
        a = math.radians(deg)
        rim = tuple(w + RIM_RADIUS * (math.cos(a) * u + math.sin(a) * v)
                    for w, u, v in zip(WHEEL_CENTER, e1, e2))
        spokes.append(MeshSpec("link", {"p0": WHEEL_CENTER, "p1": rim, "thick": 0.06}))
    return BodyDef("steering_column", MeshSpec("compound", {"parts": [
        MeshSpec("sphere", {"radius": 0.05, "center": U_UPPER}),                       # upper cross
        _cyl(0.035, COLUMN_LEN, _add(U_UPPER, COLUMN_AXIS, COLUMN_LEN / 2), axis=COLUMN_AXIS),
        _cyl(0.13, 0.10, WHEEL_CENTER, axis=COLUMN_AXIS),                              # hub / horn pad
        MeshSpec("torus", {"radius": RIM_RADIUS, "tube": 0.05, "center": WHEEL_CENTER,
                           "axis": COLUMN_AXIS}),                                      # rim
        *spokes,
    ]}), density=STEEL, color=(.15, .15, .17))


def _half_shaft(side, sx):
    """The V8's side gear (a gap toward the spiders, phase 0) and shaft,
    ending inside the CV bell."""
    length = SHAFT_OUTER_X - V8_SHAFT_INNER_X
    return BodyDef(f"axle_{side}", MeshSpec("compound", {"parts": [
        _v8._bevel(X if sx > 0 else NEG_X, 0.0, 0.10),
        _cyl(SHAFT_RADIUS, length, (sx * (V8_SHAFT_INNER_X + length / 2), 0., 0.)),
    ]}), density=STEEL, color=(.25, .58, .88) if sx < 0 else (.32, .76, .48))


# ── body frame (static): a car's front structure instead of a plate ─────────
# Two longitudinal rails under the half-shafts' pillow blocks; crossmembers
# under the engine, the crank's rear bearing and the rack; a front bumper
# beam; a dash crossbar on A-pillar posts carrying the column; and a fixed
# lower arm from each rail to the bottom of its kingpin.
RAIL_X, RAIL_Z = 1.95, -0.75
RAIL_Y = (-6.4, 4.9)
_RAIL_TOP = RAIL_Z + 0.10
_COWL_Z = COLUMN_BEARING[2]


def _post(x, y, z_top, z_bot=_RAIL_TOP, half=0.06):
    return _box((x, y, (z_top + z_bot) / 2), (half, half, (z_top - z_bot) / 2))


def _cross_member(y, half_y=0.08):
    return _box((0., y, RAIL_Z), (RAIL_X, half_y, 0.08))


_GROUND = BodyDef("__ground", MeshSpec("compound", {"parts": [
    *[_box((sx * RAIL_X, sum(RAIL_Y) / 2, RAIL_Z), (0.08, (RAIL_Y[1] - RAIL_Y[0]) / 2, 0.10))
      for _, sx in SIDES],                                                              # frame rails
    _cross_member(RAIL_Y[1] - 0.10, 0.10),                                              # bumper beam
    _cross_member(3.15), _post(0., 3.15, -0.48),                                        # engine mount
    _cross_member(1.30), _post(0., 1.30, -0.13),                                        # crank rear bearing
    _cross_member(_steer.RACK_Y),                                                       # rack crossmember
    *[_post(sx * 2.00, 0., -0.15) for _, sx in SIDES],                                  # axle pillow posts
    *[MeshSpec("link", {"p0": (sx * (RAIL_X + 0.06), 0., RAIL_Z),
                        "p1": (sx * _steer.KINGPIN_X, 0., -_steer.KINGPIN_BOSS_Z - 0.10),
                        "thick": 0.10}) for _, sx in SIDES],                            # lower arms
    _box((0., _steer.RACK_Y, _steer.STEER_Z), (0.90, 0.06, 0.06)),                      # rack housing
    *[_post(sx * 0.70, _steer.RACK_Y, _steer.STEER_Z - 0.06) for _, sx in SIDES],      # housing posts
    *[_post(sx * RAIL_X, COWL_Y, _COWL_Z + 0.08) for _, sx in SIDES],                  # A-pillar posts
    _box((0., COWL_Y, _COWL_Z), (RAIL_X + 0.06, 0.08, 0.08)),                           # dash crossbar
    MeshSpec("link", {"p0": (COLUMN_BEARING[0], COWL_Y, _COWL_Z), "p1": COLUMN_BEARING,
                      "thick": 0.09}),                                                  # column bracket
]}), fixed=True, color=(.34, .36, .40))

CASE = AtlasCaseSpec(
    name="drivetrain_v8_steer",
    mechanism=MechanismSpec(
        bodies=[_GROUND, _v8._crankshaft(),
                *[_v8._rod(i, bank) for i in range(4) for bank in _v8.BANKS],
                *[_v8._piston(i, bank) for i in range(4) for bank in _v8.BANKS],
                _v8._carrier(), _v8._SPIDER, _v8._SPIDER_B,
                *[_half_shaft(s, sx) for s, sx in SIDES],
                *[_steer._knuckle(s, sx) for s, sx in SIDES],
                *[_steer._wheel(s, sx) for s, sx in SIDES],
                _steer._RACK, *[_steer._tie_rod(s, sx) for s, sx in SIDES],
                _pinion_shaft(), _intermediate_shaft(), _column()],
        joints=_v8._engine_joints() + [
            JointDef("final_drive", "bevel_gear", "crankshaft", "carrier", body_c="__ground",
                     axis=Y, axis2=X, ratio=float(V8_FINAL_RATIO)),
            JointDef("carrier_bearing", "revolute", "carrier", "__ground", pivot=ORIGIN, axis=X),
            JointDef("spider_bearing", "revolute", "spider", "carrier", pivot=ORIGIN, axis=Y),
            JointDef("spider_b_bearing", "revolute", "spider_b", "carrier", pivot=ORIGIN, axis=NEG_Y),
            JointDef("mesh_right", "bevel_gear", "spider", "axle_right", body_c="carrier",
                     axis=Y, axis2=X, ratio=1.0),
            JointDef("mesh_left", "bevel_gear", "spider", "axle_left", body_c="carrier",
                     axis=Y, axis2=X, ratio=-1.0),
            JointDef("mesh_b_right", "bevel_gear", "spider_b", "axle_right", body_c="carrier",
                     axis=NEG_Y, axis2=X, ratio=1.0),
            JointDef("rack_slide", "prismatic", "__ground", "rack",
                     pivot=(0., _steer.RACK_Y, _steer.STEER_Z), axis=X,
                     limit_lo=-_steer.RACK_TRAVEL, limit_hi=_steer.RACK_TRAVEL),
            *[j for s, sx in SIDES for j in _steer._side_joints(s, sx)],
            JointDef("pinion_bearing", "revolute", "pinion_shaft", "__ground",
                     pivot=PINION_CENTER, axis=PINION_AXIS),
            JointDef("steering_gear", "rack_pinion", "pinion_shaft", "rack",
                     pivot=PINION_CENTER, axis=PINION_AXIS, axis2=X, ratio=PINION_PITCH),
            JointDef("lower_hooke", "universal", "pinion_shaft", "intermediate_shaft",
                     pivot=U_LOWER, axis=_N_LOWER, axis2=_unit(_cross(INTER_AXIS, _N_LOWER))),
            JointDef("upper_hooke", "universal", "steering_column", "intermediate_shaft",
                     pivot=U_UPPER, axis=_N_UPPER, axis2=_unit(_cross(INTER_AXIS, _N_UPPER))),
            JointDef("column_bearing", "revolute", "steering_column", "__ground",
                     pivot=COLUMN_BEARING, axis=COLUMN_AXIS),
        ],
        up_axis="z", gravity=0.0,
    ),
    coordinates=[],
    contact_model="none",
    actuators=[
        ActuatorDef("engine torque", kind="joint_torque", target="crank_main", hi=40.0),
        ActuatorDef("left wheel brake", kind="joint_torque", target="left_bearing", hi=40.0),
        ActuatorDef("right wheel brake", kind="joint_torque", target="right_bearing", hi=40.0),
        ActuatorDef("steering torque", kind="joint_torque", target="column_bearing", hi=10.0),
    ],
    build=BuildOptions(
        # Provisional: `drivetrain_v8`'s step.  The steering extent in the
        # packed pose metric has not been measured for this case.  No sample
        # cap: the frontier must close on its own.
        sampling=SamplingOptions(n_samples=None, step=1.0, max_seconds=2400,
                                 checkpoint_seconds=300,
                                 tangent_eps=0.04, tangent_rel_tol=0.20),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
