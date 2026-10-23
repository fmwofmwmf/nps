"""Drivetrain on a stand with a front-mounted, forward-facing 90-degree
flat-plane V8: crank fore-aft on the centreline, bevel final drive, open
differential, two wheels hanging free.

  * one crankshaft with the inline-4's four throws (0-180-180-0, a flat-plane
    V8 crank) on a fore-aft axis; each crank pin is twice as long and carries
    two connecting rods side by side, one per bank;
  * bank A's bores lean 45 deg left, bank B's 45 deg right; each piston slides
    on its own tilted bore (eight closed slider-crank loops, adding no freedom);
  * the crank's long rear shaft ends in a straight bevel pinion meshing a ring
    gear bolted to the differential carrier (`bevel_gear` with the stand as its
    carrier: both bearings are fixed to the frame).  The crank turns
    `V8_FINAL_RATIO` times per carrier turn;
  * an open bevel differential (1:1 mitre side gears, two spiders on a
    cross-pin that points forward at rest) and one rigid axle+wheel body per
    side, as in `drivetrain`.

Five free bearings less three rolling meshes leaves two freedoms (engine
rotation, differential split).  Nothing about that reaches the learner: no
coordinates, dimension, topology or chart count.  `EVALUATOR_REFERENCE` is read
only by tests.

Meshes here are the FUNCTIONAL layer (closed solids for mass and inertia); the
open-topped translucent banks are display-only, built from `_drivetrain_params`
by `scripts/assets/build_drivetrain_display.py --v8`.
"""
from __future__ import annotations

import math

from atlas_rom.cases import drivetrain as _base
from atlas_rom.cases._drivetrain_params import (
    BORE_RADIUS, BRAKE_X, CRANK_RADIUS, HUB_X, JOURNAL_RADIUS, NEG_X, NEG_Y, PULLEY_RADIUS,
    PEDESTAL_X, SHAFT_RADIUS, STAND_Z, THROW_SIGN, TYRE_RING, TYRE_TUBE, V8_BANK_AXIS, V8_CAGE_RADIUS,
    V8_CAGE_X, V8_FINAL_CONE, V8_FINAL_RATIO, V8_JOURNAL_Y, V8_MITRE, V8_PIN_HALF, V8_PIN_Y,
    V8_PINION_ANGLE, V8_PINION_TEETH, V8_PULLEY_Y, V8_RING_BORE, V8_RING_TEETH,
    V8_SHAFT_INNER_X, WHEEL_X, X, Y, v8_crank_pin, v8_wrist_pin)
from atlas_rom.cases.drivetrain import ALLOY, STEEL, WHEEL, _box, _cyl
from atlas_rom.spec.case_spec import (
    ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, JointDef, MechanismSpec, MeshSpec,
    SamplingOptions)

EVALUATOR_REFERENCE = {
    "configuration_space": "T^2",
    "intrinsic_dimension": 2,
    "gear_law": "theta_crank(y) = -2 theta_carrier(x); theta_left + theta_right = 2 theta_carrier",
}
BANKS = ("A", "B")
ORIGIN = (0.0, 0.0, 0.0)


def _bevel(axis, phase, hub_len, cone=None, **extra):
    params = {**(cone or V8_MITRE), "tooth_phase": phase, "hub_len": hub_len,
              "center": ORIGIN, "axis": axis, "flip": False, **extra}
    params.setdefault("hub_radius", 0.045)
    return MeshSpec("bevel_gear", params)


# ── engine ───────────────────────────────────────────────────────────────────
def _crankshaft():
    y0, y1 = V8_JOURNAL_Y
    parts = [_cyl(JOURNAL_RADIUS, y1 - y0, (0., (y0 + y1) / 2, 0.), axis=Y)]
    for y, s in zip(V8_PIN_Y, THROW_SIGN):
        pin_z = s * CRANK_RADIUS
        parts.append(_cyl(0.055, 2 * V8_PIN_HALF, (0., y, pin_z), axis=Y))   # double-width pin
        for dy in (-(V8_PIN_HALF + 0.02), V8_PIN_HALF + 0.02):               # two webs
            lo, hi = sorted((-s * 0.12, pin_z + s * 0.07))
            parts.append(_box((0., y + dy, (lo + hi) / 2), (0.075, 0.02, (hi - lo) / 2)))
            parts.append(_cyl(0.15, 0.04, (0., y + dy, -s * 0.07), axis=Y))  # counterweight
    parts += [
        _cyl(PULLEY_RADIUS, 0.07, (0., V8_PULLEY_Y, 0.), axis=Y),
        # Drive pinion: a tooth faces the ring at the contact line (+x).
        _bevel(Y, 0.0, 0.10, {**V8_FINAL_CONE, "pitch_angle": V8_PINION_ANGLE,
                              "teeth": V8_PINION_TEETH}, hub_radius=0.07),
    ]
    return BodyDef("crankshaft", MeshSpec("compound", {"parts": parts}),
                   density=STEEL, color=(.62, .64, .68))


def _rod(i, bank):
    p, w = v8_crank_pin(i, bank), v8_wrist_pin(i, bank)
    d = tuple(b - a for a, b in zip(p, w))
    length = math.sqrt(sum(c * c for c in d))
    mid = tuple((a + b) / 2 for a, b in zip(p, w))
    return BodyDef(f"rod_{bank}{i + 1}", MeshSpec("compound", {"parts": [
        _cyl(0.035, length - 0.10, mid, axis=d),                             # shank along the rod
        _cyl(0.08, 0.07, p, axis=Y),                                         # big end
        _cyl(0.045, 0.07, w, axis=Y),                                        # small end
    ]}), density=STEEL, color=(.80, .52, .20))


def _piston(i, bank):
    w, a = v8_wrist_pin(i, bank), V8_BANK_AXIS[bank]
    crown = tuple(c + 0.05 * e for c, e in zip(w, a))
    return BodyDef(f"piston_{bank}{i + 1}", MeshSpec("compound", {"parts": [
        _cyl(BORE_RADIUS, 0.20, crown, axis=a),                              # crown + skirt
        _cyl(0.035, 0.26, w, axis=Y),                                        # wrist pin
    ]}), density=ALLOY, color=(.82, .84, .88))


# ── differential: cross-pin forward, ring gear bolted to the +x flange ─────
def _carrier():
    h, r = V8_CAGE_X, V8_CAGE_RADIUS
    parts = [MeshSpec("torus", {"radius": r, "tube": 0.035, "center": (sx * h, 0., 0.), "axis": X})
             for sx in (-1, 1)]                                               # end hoops; +x = flange
    for c in ((0., r, 0.), (0., -r, 0.), (0., 0., r), (0., 0., -r)):
        parts.append(_box(c, (h, 0.03, 0.03)))                                # cage bars
    parts += [
        _cyl(0.025, 2 * r, ORIGIN, axis=Y),                                  # spider cross-pin
        # Ring gear: an annulus whose bore sits on the flange hoop; a gap faces
        # the pinion at the contact line (+y).
        _bevel(X, math.pi / 2 - math.pi / V8_RING_TEETH, 0.0,
               {**V8_FINAL_CONE, "pitch_angle": 90.0 - V8_PINION_ANGLE, "teeth": V8_RING_TEETH},
               bore=V8_RING_BORE, bolts=10),
    ]
    return BodyDef("carrier", MeshSpec("compound", {"parts": parts}),
                   density=STEEL, color=(.90, .52, .15))


def _axle(side):
    """Side gear, half-shaft, brake disc and wheel as one rigid body; a
    side-gear GAP faces the forward/back spiders (tooth phase 0 for an even
    14-tooth gear)."""
    sx = 1.0 if side == "right" else -1.0
    shaft_len = HUB_X - V8_SHAFT_INNER_X
    parts = [
        _bevel(X if side == "right" else NEG_X, 0.0, 0.10),
        _cyl(SHAFT_RADIUS, shaft_len, (sx * (V8_SHAFT_INNER_X + shaft_len / 2), 0., 0.)),
        _cyl(0.26, 0.10, (sx * HUB_X, 0., 0.)),                               # hub flange
        _cyl(0.60, 0.07, (sx * BRAKE_X, 0., 0.)),                             # brake disc
        MeshSpec("torus", {"radius": TYRE_RING, "tube": TYRE_TUBE,
                           "center": (sx * WHEEL_X, 0., 0.), "axis": X}),     # tyre
    ]
    return BodyDef(f"axle_{side}", MeshSpec("compound", {"parts": parts}),
                   density=WHEEL, color=(.25, .58, .88) if side == "left" else (.32, .76, .48))


_SPIDER = BodyDef("spider", _bevel(Y, 0.0, 0.05), density=STEEL, color=(.88, .30, .34))
_SPIDER_B = BodyDef("spider_b", _bevel(NEG_Y, 0.0, 0.05), density=STEEL, color=(.88, .30, .34))

_GROUND = BodyDef("__ground", MeshSpec("compound", {"parts": [
    _box((0., 1.55, STAND_Z), (2.6, 2.95, 0.06)),                             # base plate
    *[_box((sx * PEDESTAL_X, 0., (STAND_Z - SHAFT_RADIUS) / 2), (0.06, 0.06, -STAND_Z / 2))
      for sx in (-1, 1)],                                                     # axle pedestals
    *[_box((0., y, (STAND_Z - 0.40) / 2), (0.06, 0.06, (-STAND_Z - 0.40) / 2))
      for y in (V8_PIN_Y[0] - 0.36, V8_PIN_Y[-1] + 0.36)],                    # engine mounts
]}), fixed=True, color=(.34, .36, .40))


def _engine_joints():
    joints = [JointDef("crank_main", "revolute", "crankshaft", "__ground",
                       pivot=(0., V8_PIN_Y[0], 0.), axis=Y)]
    for i in range(len(V8_PIN_Y)):
        for bank in BANKS:
            n = f"{bank}{i + 1}"
            joints += [
                JointDef(f"rod_pin_{n}", "revolute", "crankshaft", f"rod_{n}",
                         pivot=v8_crank_pin(i, bank), axis=Y),
                JointDef(f"wrist_pin_{n}", "revolute", f"rod_{n}", f"piston_{n}",
                         pivot=v8_wrist_pin(i, bank), axis=Y),
                JointDef(f"bore_{n}", "prismatic", "__ground", f"piston_{n}",
                         pivot=v8_wrist_pin(i, bank), axis=V8_BANK_AXIS[bank]),
            ]
    return joints


CASE = AtlasCaseSpec(
    name="drivetrain_v8",
    mechanism=MechanismSpec(
        bodies=[_GROUND, _crankshaft(),
                *[_rod(i, bank) for i in range(4) for bank in BANKS],
                *[_piston(i, bank) for i in range(4) for bank in BANKS],
                _carrier(), _SPIDER, _SPIDER_B, _axle("left"), _axle("right")],
        joints=_engine_joints() + [
            # Pinion on the crank (axle +y, bearing in the stand) rolling on the
            # ring gear (spin +x); the stand is the carrier.  Contact on the
            # +x/+y pitch line gives crank spin = -2 x carrier spin: ratio +2.
            JointDef("final_drive", "bevel_gear", "crankshaft", "carrier", body_c="__ground",
                     axis=Y, axis2=X, ratio=float(V8_FINAL_RATIO)),
            JointDef("carrier_bearing", "revolute", "carrier", "__ground", pivot=ORIGIN, axis=X),
            JointDef("left_bearing", "revolute", "axle_left", "__ground", pivot=ORIGIN, axis=X),
            JointDef("right_bearing", "revolute", "axle_right", "__ground", pivot=ORIGIN, axis=X),
            JointDef("spider_bearing", "revolute", "spider", "carrier", pivot=ORIGIN, axis=Y),
            JointDef("spider_b_bearing", "revolute", "spider_b", "carrier", pivot=ORIGIN, axis=NEG_Y),
            # Same handedness as the upright cross-pin: +x side gear +1, -x side -1.
            JointDef("mesh_right", "bevel_gear", "spider", "axle_right", body_c="carrier",
                     axis=Y, axis2=X, ratio=1.0),
            JointDef("mesh_left", "bevel_gear", "spider", "axle_left", body_c="carrier",
                     axis=Y, axis2=X, ratio=-1.0),
            JointDef("mesh_b_right", "bevel_gear", "spider_b", "axle_right", body_c="carrier",
                     axis=NEG_Y, axis2=X, ratio=1.0),
        ],
        up_axis="z", gravity=0.0,
    ),
    coordinates=[],
    contact_model="none",
    actuators=list(_base.CASE.actuators) + [
        ActuatorDef("right wheel brake", kind="joint_torque", target="right_bearing", hi=40.0),
    ],
    build=BuildOptions(
        # Pose-metric area measures ~523 (as the inline-4's ~546); each
        # analytic projection costs ~2.4x (264 vs 168 pose columns).  A front
        # at step 1.3 closed at 350 samples (NN spacing 1.2) but was too sparse
        # to chart (211/350 uncovered); the inline-4 charted at spacing 0.95
        # (step 1.0).  The dataset is that closed front re-opened (bare: every
        # frame re-measured) and advanced at step 1.0 until it closes again.
        # No sample cap: the frontier must close on its own; checkpoints let
        # an interrupted front resume.
        sampling=SamplingOptions(n_samples=None, step=1.0, max_seconds=2400,
                                 checkpoint_seconds=300,
                                 tangent_eps=0.04, tangent_rel_tol=0.20),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
