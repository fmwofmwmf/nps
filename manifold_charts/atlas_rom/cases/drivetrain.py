"""Drivetrain on a stand: transverse inline-4 -> spur final drive -> open
differential -> two wheels hanging free.

  * an inline-4 engine: one crankshaft, four connecting rods, four pistons on
    rails (four closed slider-crank loops, which add no freedom);
  * the crankshaft's pinion drives a ring gear bolted to the differential
    carrier through an external spur pair (`gear_ratio` with the stand as its
    carrier: the crank turns `FINAL_RATIO` times per carrier turn);
  * an open bevel differential: two 1:1 mitre side gears on the axle line and
    two spider gears on a cross-pin carried by the carrier;
  * each axle is one rigid body: its side gear, half-shaft and wheel.

The layout is transverse (engine parallel to the axle, as in a front-wheel-drive
transaxle) because the analytic oracle's `bevel_gear` constraint needs a moving
carrier; a fixed-axis spur pair expresses the final drive with `gear_ratio`.

Counting what the mechanism actually closes: five free bearings (crank,
carrier, both axles, spider) less three rolling meshes (final drive and the
two meshes of the driven spider) leaves two freedoms -- engine rotation and the
left/right split of the differential.  The second spider meshes the right axle
only; its mesh with the left axle then holds automatically.  Every gear ratio
is an integer (2 and 1), so every body's angle is an integer combination of the
two free angles and the configuration space closes after one turn of each.

Nothing about that is told to the learner: no coordinates, no dimension, no
topology, no chart count.  `EVALUATOR_REFERENCE` below is read only by tests.

Meshes here are the FUNCTIONAL layer -- closed solids that set mass and inertia.
The detailed render-only parts (tyres, rims, engine block, housings) are built
from the same `_drivetrain_params` by `scripts/assets/build_drivetrain_display.py`.
"""
from __future__ import annotations

import math

from atlas_rom.cases._drivetrain_params import (
    BORE_RADIUS, CARRIER_HOOP_RADIUS, CARRIER_HOOP_X, CRANK_RADIUS, CRANK_Y, CRANK_Z,
    CROSS_PIN_HALF, CYL_X, FINAL_RATIO, FINAL_X, FLYWHEEL_RADIUS, FLYWHEEL_TEETH,
    FLYWHEEL_X, GEAR_WIDTH, HUB_X, JOURNAL_RADIUS, JOURNAL_X, MITRE, NEG_X, NEG_Z,
    NEG_Z_MESH,
    PINION_PITCH, PINION_TEETH, PULLEY_RADIUS, PULLEY_X, RING_PITCH, RING_TEETH,
    ROD_LENGTH, SHAFT_INNER_X, SHAFT_RADIUS, PEDESTAL_X, SIDE_PHASE, STAND_Z, THROW_SIGN, TYRE_RING,
    TYRE_TUBE, BRAKE_X, WHEEL_X, X, Z, crank_pin, wrist_pin)
from atlas_rom.spec.case_spec import (
    ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, JointDef,
    MechanismSpec, MeshSpec, SamplingOptions)

EVALUATOR_REFERENCE = {
    "configuration_space": "T^2",
    "intrinsic_dimension": 2,
    "gear_law": "theta_crank = -2 theta_carrier; theta_left + theta_right = 2 theta_carrier",
}

STEEL, ALLOY, WHEEL = 120.0, 35.0, 30.0       # kg per scene-unit^3 (1 unit^3 ~ 0.0156 m^3)


def _cyl(radius, height, center, axis=X):
    return MeshSpec("cylinder", {"radius": radius, "height": height,
                                 "center": center, "axis": axis})


def _box(center, half):
    return MeshSpec("box", {"center": center, "half": half})


# ── engine ───────────────────────────────────────────────────────────────────
def _crankshaft():
    x0, x1 = JOURNAL_X
    parts = [_cyl(JOURNAL_RADIUS, x1 - x0, ((x0 + x1) / 2, CRANK_Y, CRANK_Z))]
    for x, s in zip(CYL_X, THROW_SIGN):
        pin_z = CRANK_Z + s * CRANK_RADIUS
        parts.append(_cyl(0.055, 0.14, (x, CRANK_Y, pin_z)))                 # crank pin
        for dx in (-0.075, 0.075):                                            # two webs
            lo, hi = sorted((CRANK_Z - s * 0.12, pin_z + s * 0.07))
            parts.append(_box((x + dx, CRANK_Y, (lo + hi) / 2), (0.02, 0.075, (hi - lo) / 2)))
            parts.append(_cyl(0.15, 0.04, (x + dx, CRANK_Y, CRANK_Z - s * 0.07)))  # counterweight
    parts += [
        _cyl(PULLEY_RADIUS, 0.07, (PULLEY_X, CRANK_Y, CRANK_Z)),
        MeshSpec("spur_gear", {"pitch_radius": PINION_PITCH, "teeth": PINION_TEETH,
                               "width": GEAR_WIDTH, "tooth_phase": -math.pi / 2,
                               "center": (FINAL_X, CRANK_Y, CRANK_Z), "axis": X}),
        MeshSpec("spur_gear", {"pitch_radius": FLYWHEEL_RADIUS, "teeth": FLYWHEEL_TEETH,
                               "width": 0.06, "center": (FLYWHEEL_X, CRANK_Y, CRANK_Z),
                               "axis": X}),
    ]
    return BodyDef("crankshaft", MeshSpec("compound", {"parts": parts}),
                   density=STEEL, color=(.62, .64, .68))


def _rod(i):
    (x, y, z0), (_, _, z1) = crank_pin(i), wrist_pin(i)
    return BodyDef(f"rod_{i + 1}", MeshSpec("compound", {"parts": [
        _box((x, y, (z0 + z1) / 2), (0.03, 0.035, (z1 - z0) / 2 - 0.05)),
        _cyl(0.08, 0.07, (x, y, z0)),                                         # big end
        _cyl(0.045, 0.07, (x, y, z1)),                                        # small end
    ]}), density=STEEL, color=(.80, .52, .20))


def _piston(i):
    x, y, z = wrist_pin(i)
    return BodyDef(f"piston_{i + 1}", MeshSpec("compound", {"parts": [
        _cyl(BORE_RADIUS, 0.20, (x, y, z + 0.05), axis=Z),                    # crown + skirt
        _cyl(0.035, 0.26, (x, y, z)),                                         # wrist pin
    ]}), density=ALLOY, color=(.82, .84, .88))


# ── differential ─────────────────────────────────────────────────────────────
def _bevel(axis, phase, hub_len):
    # Never `flip`: it mirrors the mesh inside out.  Point the cone with a proper
    # rotation onto `axis` instead (see NEG_Z_MESH).
    return MeshSpec("bevel_gear", {**MITRE, "tooth_phase": phase, "hub_radius": 0.07,
                                   "hub_len": hub_len, "center": (0., 0., 0.),
                                   "axis": axis, "flip": False})


def _carrier():
    h, r = CARRIER_HOOP_X, CARRIER_HOOP_RADIUS
    parts = [MeshSpec("torus", {"radius": r, "tube": 0.04, "center": (sx * h, 0., 0.), "axis": X})
             for sx in (-1, 1)]
    for c in ((0., r, 0.), (0., -r, 0.), (0., 0., r + 0.02), (0., 0., -r - 0.02)):
        parts.append(_box(c, (h, 0.035, 0.035)))                              # cage bars
    parts += [
        _cyl(0.035, 2 * CROSS_PIN_HALF, (0., 0., 0.), axis=Z),               # spider cross-pin
        _cyl(0.10, FINAL_X - h + 0.02, ((FINAL_X + h) / 2, 0., 0.)),          # ring-gear neck
        _cyl(0.42, 0.03, (FINAL_X, 0., 0.)),                                  # ring-gear web
        MeshSpec("spur_gear", {"pitch_radius": RING_PITCH, "teeth": RING_TEETH,
                               "width": GEAR_WIDTH,
                               "tooth_phase": math.pi / 2 - math.pi / RING_TEETH,
                               "center": (FINAL_X, 0., 0.), "axis": X}),
    ]
    return BodyDef("carrier", MeshSpec("compound", {"parts": parts}),
                   density=STEEL, color=(.90, .52, .15))


def _axle(side):
    sx = 1.0 if side == "right" else -1.0
    shaft_len = HUB_X - SHAFT_INNER_X
    parts = [
        _bevel(X if side == "right" else NEG_X, SIDE_PHASE, 0.12),
        _cyl(SHAFT_RADIUS, shaft_len, (sx * (SHAFT_INNER_X + shaft_len / 2), 0., 0.)),
        _cyl(0.26, 0.10, (sx * HUB_X, 0., 0.)),                               # hub flange
        _cyl(0.60, 0.07, (sx * BRAKE_X, 0., 0.)),                             # brake disc
        MeshSpec("torus", {"radius": TYRE_RING, "tube": TYRE_TUBE,
                           "center": (sx * WHEEL_X, 0., 0.), "axis": X}),     # tyre
    ]
    return BodyDef(f"axle_{side}", MeshSpec("compound", {"parts": parts}),
                   density=WHEEL, color=(.25, .58, .88) if side == "left" else (.32, .76, .48))


_SPIDER = BodyDef("spider", _bevel(Z, 0.0, 0.10), density=STEEL, color=(.88, .30, .34))
_SPIDER_B = BodyDef("spider_b", _bevel(NEG_Z_MESH, 0.0, 0.10), density=STEEL,
                    color=(.88, .30, .34))

# ── stand: render context and the fixed frame every bearing is grounded to ───
_GROUND = BodyDef("__ground", MeshSpec("compound", {"parts": [
    _box((0., 0.35, STAND_Z), (2.6, 1.05, 0.06)),                             # base plate
    *[_box((sx * PEDESTAL_X, 0., (STAND_Z - SHAFT_RADIUS) / 2), (0.06, 0.06, -STAND_Z / 2))
      for sx in (-1, 1)],                                                     # axle pedestals
    *[_box((x, CRANK_Y, (STAND_Z - 0.40) / 2), (0.06, 0.06, (-STAND_Z - 0.40) / 2))
      for x in (-2.15, 0.80)],                                                # engine mounts
]}), fixed=True, color=(.34, .36, .40))


def _engine_joints():
    joints = [JointDef("crank_main", "revolute", "crankshaft", "__ground",
                       pivot=(0., CRANK_Y, CRANK_Z), axis=X)]
    for i in range(len(CYL_X)):
        n = i + 1
        joints += [
            JointDef(f"rod_pin_{n}", "revolute", "crankshaft", f"rod_{n}", pivot=crank_pin(i), axis=X),
            JointDef(f"wrist_pin_{n}", "revolute", f"rod_{n}", f"piston_{n}", pivot=wrist_pin(i), axis=X),
            JointDef(f"bore_{n}", "prismatic", "__ground", f"piston_{n}", pivot=wrist_pin(i), axis=Z),
        ]
    return joints


CASE = AtlasCaseSpec(
    name="drivetrain",
    mechanism=MechanismSpec(
        bodies=[_GROUND, _crankshaft(), *[_rod(i) for i in range(4)],
                *[_piston(i) for i in range(4)], _carrier(), _SPIDER, _SPIDER_B,
                _axle("left"), _axle("right")],
        joints=_engine_joints() + [
            JointDef("final_drive", "gear_ratio", "crankshaft", "carrier", body_c="__ground",
                     axis=X, ratio=float(FINAL_RATIO)),
            JointDef("carrier_bearing", "revolute", "carrier", "__ground", pivot=(0., 0., 0.), axis=X),
            JointDef("left_bearing", "revolute", "axle_left", "__ground", pivot=(0., 0., 0.), axis=X),
            JointDef("right_bearing", "revolute", "axle_right", "__ground", pivot=(0., 0., 0.), axis=X),
            JointDef("spider_bearing", "revolute", "spider", "carrier", pivot=(0., 0., 0.), axis=Z),
            JointDef("spider_b_bearing", "revolute", "spider_b", "carrier", pivot=(0., 0., 0.), axis=NEG_Z),
            # Signed ratios carry mesh handedness: the +x side gear rolls against
            # the spider with +1, the -x side gear with -1.
            JointDef("mesh_right", "bevel_gear", "spider", "axle_right", body_c="carrier",
                     axis=Z, axis2=X, ratio=1.0),
            JointDef("mesh_left", "bevel_gear", "spider", "axle_left", body_c="carrier",
                     axis=Z, axis2=X, ratio=-1.0),
            JointDef("mesh_b_right", "bevel_gear", "spider_b", "axle_right", body_c="carrier",
                     axis=NEG_Z, axis2=X, ratio=1.0),
        ],
        up_axis="z", gravity=0.0,
    ),
    coordinates=[],
    contact_model="none",
    actuators=[
        ActuatorDef("engine torque", kind="joint_torque", target="crank_main", hi=40.0),
        ActuatorDef("left wheel brake", kind="joint_torque", target="left_bearing", hi=40.0),
    ],
    build=BuildOptions(
        # The manifold's area in the packed 168-D pose metric measures ~530
        # (tangent-frame area density 13.4 +- 0.4 over 12 walked poses).  A
        # first front at step 0.80 closed at 1393 samples (~0.38*step^2 of area
        # per sample) in ~1550 s; step 1.0 targets ~900 samples so the front
        # closes well inside its time budget.  No sample cap: the frontier
        # must close on its own.
        sampling=SamplingOptions(n_samples=None, step=1.0, max_seconds=1800,
                                 tangent_eps=0.04, tangent_rel_tol=0.20),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                   autoencoder_charts=True),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
