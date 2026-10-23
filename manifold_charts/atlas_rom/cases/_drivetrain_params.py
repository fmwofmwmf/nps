"""Shared dimensions for the `drivetrain` case and its display-asset generator.

One source of truth: the case file builds its functional meshes and joints from
these numbers, and `scripts/assets/build_drivetrain_display.py` builds the
detailed render-only parts from the same numbers, so a display wheel can never
drift off the axle the oracle actually turns.

Frame: X lateral (the axle line), Y forward, Z up.  One scene unit ~= 0.25 m.
The rest pose is fully assembled with every body at transform = identity:
cylinders 1 and 4 at top dead centre, 2 and 3 at bottom dead centre.
"""
from __future__ import annotations

import math

X = (1.0, 0.0, 0.0)
Y = (0.0, 1.0, 0.0)
Z = (0.0, 0.0, 1.0)
NEG_Z = (0.0, 0.0, -1.0)
NEG_X = (-1.0, 0.0, 0.0)
# `_orient` turns +z onto an exactly antiparallel axis by POINT INVERSION, which
# mirrors the mesh and turns it inside out (negative volume -> wrong inertia).
# A micro-radian tilt keeps it a proper rotation; it is mesh-only, never a joint axis.
NEG_Z_MESH = (0.0, 1e-6, -1.0)

# ── engine: transverse inline-4, crank axis parallel to the axle ─────────────
CRANK_Y, CRANK_Z = 0.75, 0.0          # crank axis line (x free)
CRANK_RADIUS = 0.16                   # half the stroke
ROD_LENGTH = 0.56
BORE_RADIUS = 0.17
CYL_X = (-1.62, -1.18, -0.74, -0.30)  # cylinder centre lines
THROW_SIGN = (1, -1, -1, 1)           # 0-180-180-0 flat-plane inline-4 crank
JOURNAL_RADIUS = 0.065
JOURNAL_X = (-2.05, 1.02)             # front pulley end .. flywheel end
PULLEY_X, PULLEY_RADIUS = -1.98, 0.13
FLYWHEEL_X, FLYWHEEL_RADIUS, FLYWHEEL_TEETH = 0.95, 0.41, 72

# ── final drive: external spur pair, crank pinion -> ring gear on the carrier
# Equal module, centre distance = sum of pitch radii = the crank-axis offset.
FINAL_X = 0.62
PINION_TEETH, RING_TEETH = 15, 30
FINAL_RATIO = RING_TEETH // PINION_TEETH          # crank turns twice per carrier turn
RING_PITCH = 0.50
PINION_PITCH = RING_PITCH * PINION_TEETH / RING_TEETH
GEAR_WIDTH = 0.10
assert abs((RING_PITCH + PINION_PITCH) - math.hypot(CRANK_Y, CRANK_Z)) < 1e-12

# ── open differential: 1:1 mitre bevel set, apex at the origin ───────────────
MITRE_TEETH = 14                      # even, so a tooth/gap faces both mates
MITRE = dict(pitch_angle=45.0, toe=0.18, heel=0.42, teeth=MITRE_TEETH,
             addendum=0.022, backlash=0.25, clearance=0.03)
SIDE_PHASE = math.pi / MITRE_TEETH    # side gears present a GAP to each spider
CARRIER_HOOP_X, CARRIER_HOOP_RADIUS = 0.48, 0.50
CROSS_PIN_HALF = 0.52

# ── axles and wheels ─────────────────────────────────────────────────────────
SHAFT_RADIUS = 0.05
SHAFT_INNER_X = 0.40
WHEEL_X = 3.15                        # wheel centre plane, |x|
WHEEL_RADIUS = 1.29                   # display tyre outer radius
# The functional tyre (inertia only) nests inside the display carcass: its
# torus spans radius 0.83..1.23 < the display tread base (1.255).
TYRE_TUBE = 0.20
TYRE_RING = WHEEL_RADIUS - 0.26
BRAKE_X = WHEEL_X - 0.24              # brake disc plane, |x|
HUB_X = 2.90

# ── stand (static) ───────────────────────────────────────────────────────────
STAND_Z = -1.75
# Along each half-shaft (|x|): inner CV boot 0.70, axle pedestal + pillow block,
# outer CV boot, then the upright/caliper bracket and the brake disc.
PEDESTAL_X = 2.00
BOOT_INNER_X, BOOT_OUTER_X = 0.70, 2.40


def crank_pin(i):
    """World crank-pin centre of cylinder i at rest."""
    return (CYL_X[i], CRANK_Y, CRANK_Z + THROW_SIGN[i] * CRANK_RADIUS)


def wrist_pin(i):
    """World wrist-pin centre of cylinder i at rest (rods vertical at dead centre)."""
    return (CYL_X[i], CRANK_Y, CRANK_Z + THROW_SIGN[i] * CRANK_RADIUS + ROD_LENGTH)


# ── drivetrain_v8: front-mounted, forward-facing 90-degree flat-plane V8 ────
# The crank runs fore-aft on the centreline at axle height (+y forward) and
# drives the differential through a 2:1 bevel pinion/ring pair whose bearings
# are both fixed to the stand.  Same stroke, rod length and throws as the
# inline-4; each crank pin carries two rods side by side, one per bank.  Bank A
# leans left (-x), bank B right (+x).  The engine sits about half an axle
# length ahead of the axle.  The diff's spider cross-pin points forward.
NEG_Y = (0.0, -1.0, 0.0)
V8_BANK_ANGLE = math.radians(45.0)                    # each bank from vertical
V8_BANK_AXIS = {"A": (-math.sin(V8_BANK_ANGLE), 0.0, math.cos(V8_BANK_ANGLE)),
                "B": (math.sin(V8_BANK_ANGLE), 0.0, math.cos(V8_BANK_ANGLE))}
V8_PIN_Y = (2.49, 2.93, 3.37, 3.81)                   # crank-pin centre planes
V8_ROD_OFFSET = {"A": -0.05, "B": 0.05}               # big ends side by side on one pin
V8_PIN_HALF = 0.11                                    # half the (double-width) crank pin
V8_JOURNAL_Y = (0.52, 4.20)                           # pinion hub .. front pulley
V8_PULLEY_Y = 4.15
# Final drive: straight bevel pinion (on the crank) and ring (on the carrier),
# apexes at the differential centre, pitch angles summing to 90 deg, teeth in
# the integer ratio so the rolling constraint stays continuous through +-pi.
V8_PINION_TEETH, V8_RING_TEETH = 15, 30
V8_FINAL_RATIO = V8_RING_TEETH // V8_PINION_TEETH
V8_PINION_ANGLE = math.degrees(math.atan2(V8_PINION_TEETH, V8_RING_TEETH))   # ~26.6 deg
# Sized like a real axle: ring gear ~1/3 of the wheel diameter.  The ring is an
# annulus (`bore`) seated on the carrier's +x flange hoop, the diff inside it.
V8_FINAL_CONE = dict(toe=0.38, heel=0.48, addendum=0.02, backlash=0.25, clearance=0.02,
                     normal_recess=True)
V8_RING_BORE = 0.29                                   # < flange hoop's outer edge: seated on it
# A smaller mitre set and cage so the whole differential nests inside the
# ring's bore and the pinion toe clears the cage.
V8_MITRE = dict(pitch_angle=45.0, toe=0.10, heel=0.24, teeth=MITRE_TEETH,
                addendum=0.013, backlash=0.25, clearance=0.03, normal_recess=True)
V8_CAGE_X, V8_CAGE_RADIUS = 0.30, 0.28
V8_SHAFT_INNER_X = 0.22                               # half-shaft starts inside the side-gear hub


def v8_crank_pin(i, bank):
    """World centre of bank `bank`'s big end on crank pin i, at rest."""
    return (0.0, V8_PIN_Y[i] + V8_ROD_OFFSET[bank], THROW_SIGN[i] * CRANK_RADIUS)


def v8_wrist_pin(i, bank):
    """World wrist-pin centre at rest: on the bank's bore axis through the crank
    axis, one rod length from the crank pin."""
    a = V8_BANK_AXIS[bank]
    along = THROW_SIGN[i] * CRANK_RADIUS * a[2]                      # (pin - crank axis) . a
    across = CRANK_RADIUS * math.hypot(a[0], a[1])                 # |(pin - crank axis) x a|
    t = along + math.sqrt(ROD_LENGTH ** 2 - across ** 2)
    return (t * a[0], V8_PIN_Y[i] + V8_ROD_OFFSET[bank], t * a[2])
