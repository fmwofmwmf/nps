"""cases/delta_robot.py — a Clavel delta robot with true parallelogram forearms.

Each of the three legs is

    ground --(R, motor)-- upper arm --(S, S)-- two parallel rods --(U, U)-- platform

The motor axis is the horizontal TANGENTIAL direction at the leg's base anchor,
so the arm swings in the vertical plane through the anchor's radial direction.
The forearm is a parallelogram: two equal rods spaced along that same tangent,
ball-jointed at the elbow crossbar and Hooke-jointed at the platform.  Each
parallelogram holds one platform edge parallel to its motor axis, and three
non-parallel motor axes together hold the platform's orientation, so the moving
platform only translates.

The wrist ends are universal, not spherical, for the same reason as in
`stewart_platform`: a rod with balls at both ends is free to spin about its own
long axis, six energy-free directions that would inflate the configuration
manifold from 3 to 9.  One pin of each Hooke joint lies along the leg tangent
(perpendicular to the rod, which lies in the leg's radial-vertical plane at
rest), the other across both.

Row count (10 moving bodies, 60 ambient dofs): 3*5 revolute + 6*3 spherical +
6*4 universal = 57, leaving 3.

A ball-jointed parallelogram can also fold into a crossed (anti-parallelogram)
assembly, in which the platform twists.  That is real kinematics of this
mechanism, not something the spec rules out; whether the sampler reaches it is
for the pipeline to discover.

Nothing about its freedoms is told to the learner: no coordinates, no
dimension, no topology, no chart count.  `EVALUATOR_REFERENCE` below is read
only by tests.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import (
    AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, JointDef, MechanismSpec, MeshSpec,
    SamplingOptions)

EVALUATOR_REFERENCE = {
    "configuration_space": "translational workspace (a patch of R^3)",
    "intrinsic_dimension": 3,
}

# ─── geometry ─────────────────────────────────────────────────────────────────

BASE_RADIUS = 1.0
BASE_ANGLES_DEG = (90.0, 210.0, 330.0)
ARM_LENGTH = 0.6
ROD_LENGTH = 1.4
PLATFORM_RADIUS = 0.3
ROD_HALF_SPACING = 0.12            # parallelogram rods sit +-this along the tangent
HOME_ARM_DROP = math.radians(30.0)  # below horizontal, outward

# Motor stops, measured from the rest pose about the tangential axis.  Positive
# rotation about t = z x r carries r toward -z, so it LOWERS the arm: the range
# below runs from 40 deg above horizontal to 80 deg below it, a servo range
# that stays clear of the fully stretched rod.
MOTOR_LIMIT_LO = -math.radians(70.0)
MOTOR_LIMIT_HI = math.radians(50.0)

ARM_RADIUS = 0.04
ROD_RADIUS = 0.02
PLATFORM_BAR_RADIUS = 0.035
BASE_PLATE_THICK = 0.06

_Z = np.array([0.0, 0.0, 1.0])


def _leg_frame(angle_deg):
    """(radial, tangential) unit vectors at one base anchor."""
    a = math.radians(angle_deg)
    radial = np.array([math.cos(a), math.sin(a), 0.0])
    return radial, np.cross(_Z, radial)


_LEGS = [_leg_frame(a) for a in BASE_ANGLES_DEG]

BASE_PIVOTS = [BASE_RADIUS * r for r, _t in _LEGS]
ELBOWS = [b + ARM_LENGTH * (math.cos(HOME_ARM_DROP) * r - math.sin(HOME_ARM_DROP) * _Z)
          for b, (r, _t) in zip(BASE_PIVOTS, _LEGS)]

# The home platform height is DERIVED: whatever puts each wrist exactly one rod
# length below-inward of its elbow, so the rest pose is assembled by
# construction and needs no pre-relaxation.
_radial_gap = BASE_RADIUS + ARM_LENGTH * math.cos(HOME_ARM_DROP) - PLATFORM_RADIUS
_drop_sq = ROD_LENGTH ** 2 - _radial_gap ** 2
if _drop_sq <= 0.0:
    raise ValueError("delta_robot: rod is too short to close the home leg "
                     f"(radial gap {_radial_gap:.4f} >= rod {ROD_LENGTH})")
HOME_HEIGHT = -ARM_LENGTH * math.sin(HOME_ARM_DROP) - math.sqrt(_drop_sq)

WRISTS = [PLATFORM_RADIUS * r + HOME_HEIGHT * _Z for r, _t in _LEGS]


def _rod_ends(i, side):
    """(elbow end, wrist end) of rod `side` (+1 / -1) of leg i at rest."""
    offset = side * ROD_HALF_SPACING * _LEGS[i][1]
    return ELBOWS[i] + offset, WRISTS[i] + offset


for _i in range(3):
    for _s in (1, -1):
        _e, _w = _rod_ends(_i, _s)
        assert abs(float(np.linalg.norm(_w - _e)) - ROD_LENGTH) < 1e-9


def _bar_mesh(p0, p1, radius):
    """A cylindrical link spanning p0 -> p1, verts baked at the rest world pose."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    return MeshSpec("cylinder", {"radius": radius,
                                 "height": float(np.linalg.norm(d)),
                                 "center": (0.5 * (p0 + p1)).tolist(),
                                 "axis": d.tolist(), "sections": 16})


def _crossbar(center, tangent, radius):
    c, t = np.asarray(center, float), np.asarray(tangent, float)
    return _bar_mesh(c - 1.2 * ROD_HALF_SPACING * t, c + 1.2 * ROD_HALF_SPACING * t, radius)


# ─── bodies ───────────────────────────────────────────────────────────────────

_GROUND = BodyDef(
    name="__ground",
    mesh=MeshSpec("cylinder", {"radius": BASE_RADIUS + 0.15,
                               "height": BASE_PLATE_THICK,
                               "center": [0.0, 0.0, 0.5 * BASE_PLATE_THICK + ARM_RADIUS],
                               "axis": [0.0, 0.0, 1.0], "sections": 48}),
    fixed=True, color=(0.45, 0.47, 0.50))

_ARMS = [BodyDef(f"arm_{i}", MeshSpec("compound", {"parts": [
             _bar_mesh(BASE_PIVOTS[i], ELBOWS[i], ARM_RADIUS),
             _crossbar(ELBOWS[i], _LEGS[i][1], ARM_RADIUS)]}),
                 color=(0.93, 0.53, 0.18))
         for i in range(3)]

_ROD_SIDES = (("a", 1), ("b", -1))
_RODS = [BodyDef(f"rod_{i}{tag}", _bar_mesh(*_rod_ends(i, side), ROD_RADIUS),
                 color=(0.36, 0.62, 0.86))
         for i in range(3) for tag, side in _ROD_SIDES]

# One welded triangle through the three wrist centres plus a crossbar at each
# wrist, as a compound so its mass and inertia come from the geometry.
_PLATFORM = BodyDef(
    name="platform",
    mesh=MeshSpec("compound", {"parts": [
        _bar_mesh(WRISTS[i], WRISTS[j], PLATFORM_BAR_RADIUS)
        for i, j in ((0, 1), (1, 2), (2, 0))] + [
        _crossbar(WRISTS[i], _LEGS[i][1], PLATFORM_BAR_RADIUS) for i in range(3)]}),
    color=(0.35, 0.72, 0.45))


# ─── joints ───────────────────────────────────────────────────────────────────

_JOINTS = []
for _i in range(3):
    _tangent = _LEGS[_i][1]
    _JOINTS.append(JointDef(f"motor_{_i}", "revolute", "__ground", f"arm_{_i}",
                            pivot=BASE_PIVOTS[_i], axis=_tangent,
                            limit_lo=MOTOR_LIMIT_LO, limit_hi=MOTOR_LIMIT_HI))
    for _tag, _side in _ROD_SIDES:
        _elbow, _wrist = _rod_ends(_i, _side)
        _rod_dir = (_wrist - _elbow) / ROD_LENGTH
        _cross_pin = np.cross(_rod_dir, _tangent)
        _JOINTS += [
            JointDef(f"elbow_{_i}{_tag}", "spherical", f"arm_{_i}", f"rod_{_i}{_tag}",
                     pivot=_elbow),
            # Hooke joint: `axis` is the pin fixed in the rod, `axis2` the pin
            # fixed in the platform.  It removes only the rod's idle spin.
            JointDef(f"wrist_{_i}{_tag}", "universal", f"rod_{_i}{_tag}", "platform",
                     pivot=_wrist, axis=_tangent,
                     axis2=_cross_pin / np.linalg.norm(_cross_pin)),
        ]


# ─── the case ─────────────────────────────────────────────────────────────────

CASE = AtlasCaseSpec(
    name="delta_robot",
    mechanism=MechanismSpec(
        bodies=[_GROUND] + _ARMS + _RODS + [_PLATFORM],
        joints=_JOINTS,
        up_axis="z",
        gravity=9.8,
    ),
    coordinates=[],
    # Ideal joints only; nothing is meant to touch.  This puts the case on the
    # analytic path in `ideal_oracle.py`.
    contact_model="none",
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=None, step=0.3, max_seconds=3600,
                                 tangent_eps=0.02, tangent_rel_tol=0.20),
        canonical=CanonicalOptions(enabled=True, residual_discovery=True, autoencoder_charts=True),
    ),
    scenarios={"default": {"dt": 1e-3}},
)
