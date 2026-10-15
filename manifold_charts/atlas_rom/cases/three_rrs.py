"""cases/three_rrs.py — a 3-RRS platform: rotary servo cranks replacing the
prismatic legs of a 3-RPS.

Each of the three legs is

    ground --(R, motor)-- crank --(R, passive elbow)-- rod --(S)-- platform

with the leg's two revolutes sharing ONE axis: the horizontal TANGENTIAL
direction at that base anchor.  That shared axis is the architecture.  It
confines the crank and the rod to the single vertical plane spanned by the
anchor's radial direction and world +z, so the leg's ball anchor is trapped in
that plane no matter what the platform does.  Three plane constraints on a free
6-DOF platform leave 3, and — unlike the variants below — those three survive
finite travel.

WHY THE AXES ARE TANGENTIAL AND NOT RADIAL
------------------------------------------
Turning the leg axes 90 degrees, so they lie along the base circle's edge normal
and the cranks sweep tangentially, looks equivalent and is not.  With radial
axes each leg's constraint wrench is (r_i, h t_i); the three radial directions
sum to zero and so do the three tangents, so the three wrenches sum to zero and
span only rank 2.  The unconstrained direction is a yaw about +z, which slides
every anchor tangentially — inside its own plane — so no leg resists it.  Ball
joints there give 4 DOF on 3 motors.

Repairing that by pinning platform orientation fails in the other direction.
Three revolutes about three different radial axes force R = I exactly, freezing
x and y with it: a 1-DOF elevator.  That is `general/rotary_stewart.py`, whose
`forward_kinematics` closes only when all three motor angles are equal.

A universal joint at the platform appears to thread the needle — it blocks the
free yaw without over-pinning the tilts, and reports mobility 3 by Gruebler
count, by constraint-Jacobian rank, by wrench rank, and by libuipc's tangent
probe.  ALL OF THOSE ARE FIRST ORDER, AND ALL OF THEM ARE WRONG HERE.  Built and
sampled, the radial+universal mechanism is a FIRST-ORDER FLEX: a genuine 3-D
Jacobian null space that is not integrable.  Stepping a finite distance along
every tangent direction and re-projecting moves all three motors identically to
within 2.6e-06, its 992-sample cloud had motor-angle correlation exactly 1.000
and effective dimension 1, and its platform rotation stayed within 4e-6 of
identity at every sample — it never tilted once.

The lesson is in this file because it is not visible from any rank: an
architecture is only 3-DOF if it MOVES three ways.  The check that separates
them is finite travel, not spectra.

    radial + universal   finite motor spread 2.6e-06  -> 1 DOF, pure heave
    tangential + ball    finite motor spread 1.7e-02  -> 3 DOF, independent

This case is the second row.

BUILD STATUS: mechanism verified, COORDINATE verified, dynamics NOT gated
-------------------------------------------------------------------------
Verified, by measurement rather than by declaration:

    finite mobility        3 (motor spreads 1.0e-2 .. 2.0e-2 on 0.05 steps)
    closure at the seed    0.0 exactly; libuipc's project() leaves it fixed
    tangent rank           3, agreeing with libuipc to 0.29 degrees
    platform motion        tilts to 53.2 deg, height 0.320 .. 1.394 (home 0.933)
    reachable motor range  theta_0 [-0.998, +0.630] and still growing at the
                           30k-sample budget; asymmetric because the rod reaches
                           full extension sooner going up than down

NOT yet solved: the three motor angles are not a GLOBAL coordinate, because a
3-RRS has several forward-kinematics solutions per motor triple.  The branches
here are MIRRORED PLATFORM TILTS, not elbow-up/elbow-down — all 29921 samples in
the mapped cloud share the same elbow signs, while pairs 0.012 rad apart in motor
angle differ by 2.9 in pose with platform normals [+0.63,-0.31,0.71] against
[-0.64,-0.35,0.69].  About 39% of near-coincident pairs conflict, and shrinking
the coordinate box does NOT help (41% at +-0.4), so no motor-angle bound repairs
it.  `MOTOR_SWING` below therefore only keeps the workspace to a sane servo
range; it is not a branch fix, and must not be described as one.

MEASURED ANSWER: charts are NOT needed — reconstruct the coordinate instead.
The motor angles are an ACTUATOR map pi(q) = theta, many-to-one; the mistake was
adopting them as the ROM state because they are physically convenient and
locally 3-D.  A coordinate rebuilt from CONTINUATION distance in pose space
(landmark Isomap over the explorer's own graph) is single-valued on this domain,
so the manifold needs no cover at all.  On 6000 samples in the +-0.5 box:

    geodesic MDS spectrum   0.371, 0.311, 0.281, then 0.010 — a 28x drop after
                            three, so the manifold is 3-D and the embedding is
                            very nearly isometric, not a squashed projection
    false neighbours        motors 1.18% of 72000 coordinate-nearest pairs are
                            pose-far; the embedding 0.00%
    worst neighbour gap     motors 12.4x the cloud spacing, embedding 3.9x
    local conditioning      s3/s1 median 0.93 and never below 0.81, against
                            0.29 and 0.0046 for motors
    the decisive one        all 849 pairs the motor coordinate confuses (median
                            0.031 rad apart in theta, 13.6x spacing apart in
                            pose) separate by >= 1.80 in the embedding, where a
                            true neighbour sits at 0.30 — a 6x margin at the
                            WORST pair, stable across three landmark seeds

So the earlier plan — label assembly modes and import a cover via `import_mesh`,
as `cases/five_bar.py` does — was solving a problem the coordinate created.  Keep
that route in mind only if a wider box folds the embedding; over this domain it
does not.

WIRED UP, AND RE-CERTIFIED END TO END
-------------------------------------
The coordinate the ROM ships is now `build.canonical`'s z, built by the same
stages every canonical case uses.  The one thing that had to be built for it was
generality: all of `canonical.py` assumed a certified TRIANGLE complex, which a
3-DOF manifold does not have.  A cotan weight needs a face to take an angle in,
and the disk covers and Tutte parameterizations above it need a surface to be a
disk in.  What generalizes is the operator: the heat kernel on the CERTIFIED
graph converges to the same Laplace-Beltrami operator, with the alpha=1
normalization of diffusion maps dividing out the sampling density — which matters
here, because an advancing front does not deposit samples uniformly and without
it the low modes track where the explorer lingered.  Restricting the kernel to
certified edges is what keeps it safe: a plain epsilon-ball kernel is the rule
that connects two assembly modes, and a Laplacian that connects them has
eigenfunctions that average across them.

Measured on 15000 samples filling the +-0.5 box, d_z = 2d = 6:

    certified graph      14999 tree edges -> 148793 edges, 0 of 133912 chord
                         candidates rejected, 100% agreement on a 400-chord
                         audit by full projection (200 of them straddling the
                         cut, where the test's errors live)
    THE SAME REFINEMENT TEST, run on both coordinates by the same function:
      motor angles       distortion grows as N^(+0.318) [+0.281,+0.354], against
                         a collision prediction of +0.337 and a bar of +0.168
                         -> FAILED
      canonical z        distortion SHRINKS, N^(-0.370) [-0.410,-0.335]
                         -> VERIFIED
    immersion            sigma_min 0.0591 at the single worst sample, but 1st
                         percentile 0.1752 and median 0.7158, and that worst
                         sample's own 916-sample geodesic ball has median
                         0.6162.  So the low value is ONE sample's Jacobian, not
                         a degenerate region — which is the distinction the
                         worst-region readout exists to make, and the reason it
                         is reported instead of the scalar alone.
    witnesses            all 15262 pairs the motor coordinate confuses come
                         apart in z; the WORST sits 2.74 local neighbour
                         spacings away (median 6.89), where a genuinely adjacent
                         sample sits at 1.0
    injectivity          neighbour purity 1.0000, 0 impure of 15000
    chart maps           117 projection patches, encode RMSE 4.2e-06 (5.8e-06 of
                         the z scale), round trip 1.2e-15, patch-to-patch
                         agreement 5.7e-06, sigma_min(dE/du) = 1.0000000000
    decoder D(z)         validated on HELD-OUT REGIONS — eight geodesic balls of
                         the certified graph, interiors never trained on —
                         because a random split of a continuation cloud measures
                         interpolation between training neighbours, not
                         generalization.  The gap is real and it is 3x:

                             held-out regions    0.0179 / 0.83 deg
                             scattered points    0.0059 / 0.29 deg
                             tolerance           0.0500 / 3.00 deg
                             noise floor         0.0278 / 1.22 deg

                         A random holdout would have reported 0.29 deg and
                         overstated this decoder by threefold.  What matters is
                         the first row, and it is BELOW the sampling's own noise
                         floor: on manifold it has never seen, D(z) reconstructs
                         the pose more accurately than the recorded poses agree
                         with each other.  Loop closure 0.0060.

    reproduction         an INDEPENDENT 15000-sample run at sampling seed 17
                         reproduces every verdict and margin: verified /
                         verified, separated / separated, sigma_min 0.0591 ->
                         0.0797, median 0.7158 -> 0.6996, separation 1.526 ->
                         1.516, purity 1.0 -> 1.0, and both clouds' distortion
                         slopes sit far under their own bars (-0.370 vs +0.171,
                         -0.710 vs +0.172).  See `atlas_rom/reproduce.py`; it
                         compares VERDICTS and margins, never the embedding,
                         because two clouds do not share a z.

`atlas_metrics.json` keeps BOTH verdicts — `declared_coordinate` (rejected, with
its growth statistics) and `canonical_coordinate` (the full recertification) —
and names the one that ships in `shipped_coordinate`.  Overwriting the first with
the second would leave a ROM whose metrics say its coordinate is fine and no
record that a different one was tried and failed, which is precisely the missing
provenance that let the original mistake ship.

WHAT IS STILL NOT DONE: the reduced DYNAMICS does not pass its gate
------------------------------------------------------------------
The coordinate is certified and the ROM does not ship.  Those are compatible
statements and the distinction matters, so read the split before trusting either
half.  Everything that is an EXACT identity passes, most of them at round-off:

    chart round trip 1.2e-15, sigma_min(dE/du) exactly 1, mass asymmetry 0,
    mass SPD (min eig 0.037, max cond 5.6e+04), the mass/kinetic-energy
    identity 1.5e-15, and every handoff invariance — pose, velocity, kinetic
    energy, potential, z, zdot — EXACTLY 0.0, with 0.0 drift over 200 closed
    cycles.  Rollout energy drift converges with dt (0.0463 -> 0.0029 at dt/4).

Everything that fails is a DERIVATIVE or an off-tangent velocity:

    dM/du step-size consistency     0.144   (tol 5e-3)
    |grad V - central difference|   70.8    (tol 0.44, |grad V| ~ 270)
    |dtheta/du - central difference| 0.463  (tol 0.027, |dtheta/du| ~ 0.99)
    zdot off-tangent after handoff  6.10    (tol 1e-6)
    max zdot off-tangent in rollout 1.49    (tol 1e-6)

That pattern does not implicate z: an embedding that folded or collided would
break the invariances first, and those are the ones reading exactly zero.  It
points at the reduced dynamics' derivative layer — finite-difference step sizing
against THIS mechanism's scale, and a velocity that leaves the tangent bundle
during integration and is only exposed at a handoff (the spike is at patch 39
with sigma_min 1.000 and condition 2.31, so it is not a conditioning failure).
The overlap-physics numbers are a third thing again: 4.1e-06 kinetic and 9.0e-06
control power against a 1e-8 tolerance, which tracks the Hermite fit residual
(8.2e-05 here against the five-bar's 1.1e-07) on patches of up to 1398 nodes.

Anyone continuing this should start with the off-tangent velocity, because it is
the only failure whose magnitude cannot be explained by fit accuracy.

ON d_z.  2d = 6 is Whitney's minimum and the five-bar found 2d+1 worth a factor
of two there (d_z=4 sigma_min 0.21, d_z=5 sigma_min 0.73), so d_z=7 was tried
here.  It is not worth it: matched at refinement step 374 the two runs sit at
sigma_min 0.0426/separation 1.41 (d_z=6) against 0.0504/1.38 (d_z=7), and both
converge to the same place.  The immersion floor on this manifold is not set by
how much room the embedding has; it is set by how smooth it is required to be,
because z is drawn from a low-frequency spectral span by construction.  Spend a
column only if a measurement says so.

A TRAP IN THE RECONSTRUCTION, since it produced a confident wrong answer first:
the explorer records ONE PARENT LINK per accepted sample, so `edges` is a
spanning TREE.  Geodesics on a tree are search order, not geometry — siblings
detour through their common ancestor — and Isomap on it returned a flat spectrum
(0.325, 0.115, 0.097, ...) with no 3-D structure at all.  The graph has to be
completed with short chords, and the rule for doing that safely now lives in
`sample_graph.py` as a build stage.  Plain kNN is the rule that merges sheets, so
each chord is certified by its MIDPOINT: on one sheet the midpoint is a secant's
and sits off the manifold by curvature x length^2, while a cross-sheet chord has
its midpoint in empty configuration space.  Calibrate the cut from the tree edges
themselves — those are chords the explorer walked, so their residuals are the
same-sheet distribution.  (Here 0 of 55034 candidates were rejected at 6000
samples and 0 of 133912 at 15000: the sheets never come within kNN range at this
density, so the certificate is insurance rather than load-bearing.  It becomes
load-bearing as density rises, which is the direction this cloud is going.)

A METHOD NOTE, because it cost two wrong turns: to decide whether a coordinate is
single-valued, explore with the pipeline's ADVANCING FRONT, not a random walk.  A
2500-point random walk on this mechanism reported 0% branch conflict; the 30k
advancing front reported 39%.  The same under-exploration made a coarse walk
declare the radial variant clean at +-0.8 when half the step showed 52.8%.
"""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, CanonicalOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

# ─── geometry ─────────────────────────────────────────────────────────────────

BASE_RADIUS = 1.0
BASE_ANGLES_DEG = (90.0, 210.0, 330.0)
CRANK_LENGTH = 0.55
ROD_LENGTH = 1.0
PLATFORM_RADIUS = 0.55
HOME_CRANK_ANGLE = math.radians(45.0)      # from the outward radial, toward +z

CRANK_RADIUS = 0.05
ROD_RADIUS = 0.04
PLATFORM_BAR_RADIUS = 0.05
BASE_PLATE_THICK = 0.06

_Z = np.array([0.0, 0.0, 1.0])


def _leg_frame(angle_deg):
    """(radial, tangential) unit vectors at one base anchor.

    Both of the leg's revolutes turn about TANGENTIAL, so the leg lives in the
    vertical plane spanned by `radial` and +z and the crank sweeps radially.
    """
    a = math.radians(angle_deg)
    radial = np.array([math.cos(a), math.sin(a), 0.0])
    return radial, np.cross(_Z, radial)


_LEGS = [_leg_frame(a) for a in BASE_ANGLES_DEG]

BASE_PIVOTS = [BASE_RADIUS * radial for radial, _t in _LEGS]
CRANK_TIPS = [b + CRANK_LENGTH * (math.cos(HOME_CRANK_ANGLE) * radial
                                  + math.sin(HOME_CRANK_ANGLE) * _Z)
              for b, (radial, _t) in zip(BASE_PIVOTS, _LEGS)]

# The home platform height is DERIVED, not chosen: it is whatever puts each ball
# anchor exactly one rod-length from its crank tip, in-plane.  Solving it rather
# than declaring it keeps the mechanism assembled by construction, so the rest
# pose needs no pre-relaxation and the recorded seed is exact.
_radial_gap = BASE_RADIUS + CRANK_LENGTH * math.cos(HOME_CRANK_ANGLE) - PLATFORM_RADIUS
_rise_sq = ROD_LENGTH ** 2 - _radial_gap ** 2
if _rise_sq <= 0.0:
    raise ValueError("three_rrs: rod is too short to close the home leg "
                     f"(radial gap {_radial_gap:.4f} >= rod {ROD_LENGTH})")
HOME_HEIGHT = CRANK_LENGTH * math.sin(HOME_CRANK_ANGLE) + math.sqrt(_rise_sq)

PLATFORM_ANCHORS = [PLATFORM_RADIUS * radial + HOME_HEIGHT * _Z
                    for radial, _t in _LEGS]

for _tip, _anchor in zip(CRANK_TIPS, PLATFORM_ANCHORS):
    assert abs(float(np.linalg.norm(_anchor - _tip)) - ROD_LENGTH) < 1e-9


def _bar_mesh(p0, p1, radius):
    """A cylindrical link spanning p0 -> p1, verts baked at the rest world pose.

    `MeshSpec("link", ...)` is a planar helper: it orients the bar with
    atan2(dy, dx) about +z only, which silently flattens any bar that rises out
    of the xy-plane.  Every bar here does, so `cylinder` is the correct one.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    return MeshSpec("cylinder", {"radius": radius,
                                 "height": float(np.linalg.norm(d)),
                                 "center": (0.5 * (p0 + p1)).tolist(),
                                 "axis": d.tolist(), "sections": 16})


# ─── bodies ───────────────────────────────────────────────────────────────────

_GROUND = BodyDef(
    name="__ground",
    mesh=MeshSpec("cylinder", {"radius": BASE_RADIUS + 0.15,
                               "height": BASE_PLATE_THICK,
                               "center": [0.0, 0.0, -0.5 * BASE_PLATE_THICK],
                               "axis": [0.0, 0.0, 1.0], "sections": 48}),
    fixed=True, color=(0.45, 0.47, 0.50))

_CRANKS = [BodyDef(f"crank_{i}", _bar_mesh(BASE_PIVOTS[i], CRANK_TIPS[i],
                                           CRANK_RADIUS),
                   color=(0.93, 0.53, 0.18))
           for i in range(3)]
_RODS = [BodyDef(f"rod_{i}", _bar_mesh(CRANK_TIPS[i], PLATFORM_ANCHORS[i],
                                       ROD_RADIUS),
                 color=(0.36, 0.62, 0.86))
         for i in range(3)]

# The platform is one welded triangle of bars through its three ball anchors,
# built as a compound so its mass and full inertia tensor come from the actual
# geometry the same way every other body's does.
_PLATFORM = BodyDef(
    name="platform",
    mesh=MeshSpec("compound", {"parts": [
        _bar_mesh(PLATFORM_ANCHORS[i], PLATFORM_ANCHORS[j], PLATFORM_BAR_RADIUS)
        for i, j in ((0, 1), (1, 2), (2, 0))]}),
    color=(0.35, 0.72, 0.45))


# ─── joints ───────────────────────────────────────────────────────────────────

_JOINTS = []
for _i in range(3):
    _tangent = _LEGS[_i][1]
    _JOINTS += [
        JointDef(f"motor_{_i}", "revolute", "__ground", f"crank_{_i}",
                 pivot=BASE_PIVOTS[_i], axis=_tangent),
        JointDef(f"elbow_{_i}", "revolute", f"crank_{_i}", f"rod_{_i}",
                 pivot=CRANK_TIPS[_i], axis=_tangent),
        # Position only, orientation free.  A revolute here would additionally
        # pin the platform's orientation to the leg axis; see the module note.
        JointDef(f"ball_{_i}", "spherical", f"rod_{_i}", "platform",
                 pivot=PLATFORM_ANCHORS[_i]),
    ]


# ─── the case ─────────────────────────────────────────────────────────────────

# A servo working range, NOT a branch fix — see BUILD STATUS.  The assembly modes
# coexist at every motor triple, so no value here makes the MOTOR coordinate
# single-valued; that needs a coordinate reconstructed from pose, not a bound.
#
# 0.5 rad (~29 deg of crank swing) sits well inside the mapped reachable extent,
# which runs to -0.998 on the tightest axis and +0.630 on the loosest.  Keeping
# the box strictly interior means the DECLARED bound, not the mechanism's ragged
# asymmetric workspace edge, limits the domain on all six faces — which is what
# makes the atlas stage's coverage assertion satisfiable.  A box reaching for the
# true edge fails that assertion, because the front stops where the legs run out
# rather than where the box says it should.
MOTOR_SWING = 0.5

CASE = AtlasCaseSpec(
    name="three_rrs",
    mechanism=MechanismSpec(
        bodies=[_GROUND] + _CRANKS + _RODS + [_PLATFORM],
        joints=_JOINTS,
        up_axis="z",          # a genuinely spatial mechanism: gravity is world -z
        gravity=9.8,
    ),
    seeds=[],
    # Every coupling here is an ideal joint; nothing is meant to touch.  This
    # also puts the case on the analytic path in `ideal_oracle.py` — closed-form
    # projection and an exact SVD tangent space, ~75x faster than driving
    # libuipc, and verified against it to 0.29 degrees of tangent subspace.
    contact_model="none",
    coordinates=[
        CoordinateDef(f"theta_{i}", topology="bounded",
                      lo=-MOTOR_SWING, hi=MOTOR_SWING,
                      source="joint_angle", args={"joint": f"motor_{i}"})
        for i in range(3)
    ],
    actuators=[
        ActuatorDef(f"motor {i} torque", kind="joint_torque", target=f"motor_{i}",
                    hi=400.0)
        for i in range(3)
    ],
    build=BuildOptions(
        # `step` is a distance in PACKED POSE space, not in the coordinates, and
        # the conversion is per-mechanism: here one ambient step buys
        # du ~ 0.31*step of motor angle (measured, linear over 0.03..0.30).
        # Filling a 3-D box costs (range/du)^3, so the conversion matters enormously
        # — step 0.03 gives du 0.009 and would need 3.5 MILLION samples to fill
        # +-0.7, which is why it covered only +-0.13 and failed the bounded-
        # coordinate coverage check.  step 0.30 gives du 0.093, ~15 per axis and
        # ~3400 samples to fill; 6000 leaves margin to reach the workspace edge,
        # and the analytic oracle samples at ~90/s so it costs about a minute.
        sampling=SamplingOptions(n_samples=15000, step=0.30, max_seconds=5400),
        charts=ChartOptions(mode="single"),
        # The ROM's coordinate.  `mode="single"` above is the DECLARED motor
        # coordinate, kept because the atlas stage's verdict on it is this case's
        # provenance — it is measured, rejected, and recorded, not quietly
        # dropped.  What ships is z: 2*d = 6 columns of the certified sample
        # graph's Laplace-Beltrami spectrum, refined against the same separation
        # and immersion targets every canonical case uses.
        #
        # `planarity` is what a projection chart may bend by across its own
        # extent.  0.2 is the five-bar's value and it is not free here: a 3-D
        # manifold's charts are graphs over a 3-plane in R^6, so a looser bound
        # buys larger patches and a worse Hermite condition number.
        canonical=CanonicalOptions(enabled=True, dimension=6, planarity=0.2),
        decoder=DecoderOptions(epochs=6000, smooth_lambda=1e-3,
                               closure_weight=1.0),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=3000),
        # No barrier field.  The SDF boundary this replaces was trained on the
        # DECLARED coordinate — a signed distance to the motor box — and a
        # canonical ROM never visits that coordinate: its state is z, and the
        # edge of the explored region is the edge of the projection cover, which
        # patch membership already enforces.  Keeping the field would ship a
        # barrier in units the runtime does not use.
        boundary=BoundaryOptions(kind="none"),
        expect=Expectations(dimension=3, n_bodies=7),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2),
    ),
    # A canonical ROM has no `initial_u` to declare: the state is z, and the
    # runtime starts from the patch containing the sample nearest the home pose.
    # `boundary_solver` likewise belongs to the chart runtime's barrier, which a
    # projection cover replaces with patch membership.
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "exact_canonical"},
               "exact": {"dt": 1e-3, "dynamics_backend": "exact_canonical"},
               "distilled": {"dt": 1e-3, "dynamics_backend": "distilled_canonical"}},
)
