"""cases/rotate.py — the rotate.glb spatial 4-bar as an IDEAL-JOINT case.

`general/libuipc_oracle.build_rotate_oracle` treats this mechanism as
genuine contact: cranks A, D are pinned revolutes, but couplers B, C are
"free coupler links, located ONLY by IPC contact" — no joint is declared
between A-B, B-C, C-D, so every project() pays a ~6.6 s/step libuipc settle
(`rotate/CONTACT_STEPPER.md`, `profile_rotate.py`).

That contact turns out not to be doing anything a joint couldn't say directly,
and the mesh itself says exactly what: each `rotate-*` geometry (glTF names
"Circle.0xx") is built from perfectly circular edge loops — pin shafts and
their mating bores — and every one of them comes out axis-aligned to
noise-floor precision when fit in world space (`fit_circle_3d` below: planarity
and circularity residuals both 0.0 to 1e-8, the float roundoff on a baked
transform). That is a stronger statement than a statistical fit: it is the
CAD, not an estimate from noisy motion data. Trusting it directly (rather than
least-squares fitting axes from the recorded contact trajectory, which is what
an earlier version of this file did) removed the ~1-2% axis error that a fit
against compliant/backlash-y contact motion carries, and which was enough to
occasionally close the analytic Jacobian's rank gap below the pipeline's
1e3 threshold far from rest.

  * A-B: rotate-A carries a stepped Z-axis shaft at (x,y)=(10, 0); rotate-B
    carries the mating Z-axis bore (r=0.670) spanning z=[20.742, 21.258] —
    axis exactly [0,0,1], transverse position exactly (10, 0), bore
    midpoint z=21.0 (a round number, not a coincidence).
  * B-C: rotate-B's sleeve and rotate-C's bore (r=0.583) share axis [0,1,0]
    at (x,z)=(10, 16); rotate-C's bore spans y=[3.742, 4.258], midpoint
    y=4.0.
  * C-D: rotate-C's shaft and rotate-D's bore share axis [1,0,0] at
    (y,z)=(0, 16); rotate-C's two rings sit at x={4.742, 5.258}, midpoint
    x=5.0.

The two ground revolutes (crank_A, crank_D) were never contact features —
`build_rotate_oracle` pins each crank about its own node origin with a
hardcoded world axis, so those pivots/axes are exact by construction too:
read straight off the GLB scene graph, not fit.

A joint's pivot is only determined up to sliding along its own axis (the
constraint Jacobian is degenerate in that one direction — any point on the
hinge line is an equally valid `pivot`), so the axial coordinate picked for
each interior joint (the bore midpoint) is a convention, not a measurement;
only the two transverse coordinates and the axis direction are physical, and
those come out exact.

So this case declares the whole loop as five ideal revolutes:

    ground --(R, world +x, actuated)-- A --(R, world +z)-- B
      --(R, world +y)-- C --(R, world +x)-- D --(R, world +y, actuated)-- ground

with contact_model="none", which puts it on `ideal_oracle.py`'s analytic path:
closed-form Gauss-Newton project() and an exact SVD tangent_frame(), instead
of a GPU IPC settle per sample (measured ~75x on three_rru; the win here
should be similar or larger since rotate's contact settle was tuned to be
this slow ON PURPOSE — see build_rotate_oracle's docstring on why the cheap
levers all failed).

Pivots and axes below are MEASURED, not modeled from CAD: each of the 5
world-frame numbers is a least-squares fit against the recorded relative
motion, expressed at the mechanism's rest pose (theta_A = theta_D = 0, where
every body's rotation is exactly identity in the recorded packing — see
`node_thetas`/`configs` in the fill dataset). Rounding is to the fit's own
precision (~1e-3), not to invented "nice" values.

Grubler's count for a 5-link, 5-revolute spatial loop is -1 (generically
rigid); this mechanism moves anyway, the way a Bennett-family linkage does —
by a special geometric relation between the axes rather than by joint variety.
`ideal_oracle`'s rank-by-spectral-gap already handles that (it measures the
Jacobian's null space at each configuration, it does not count DOF from the
joint list), so nothing case-specific has to detect it.

COORDINATES ARE DECLARED EMPTY ON PURPOSE. Prior work coordinatizing this
manifold by the two crank angles (theta_A, theta_D) found it is NOT injective
there — the passive elbow (bodies B, C) has an up/down assembly mode, so the
manifold double-covers the (theta_A, theta_D) rectangle (`rotate/NOTES.md`,
the "spherical 4-bar whose manifold is a RING" section; `rotate_elbow.py`'s
`s` pseudoscalar detects the branch). The atlas_rom canonical/residual-
discovery path (same one `five_bar` and `three_rrs` use) reconstructs an
injective coordinate from the sampled poses and their certified neighbour
graph instead of being handed the non-injective angle pair, which is the
generic fix for exactly this failure mode.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, ChartOptions, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

_ASSET = "assets/rotate.glb"

# Rest-pose world translations of each node's origin, read directly off the
# GLB scene graph (`scene.graph["rotate-A"] -> (transform, geometry)`), which
# is also the packing convention this pipeline uses (MeshSpec("file", ...)
# bakes the GLB node's rest transform into the mesh, so BodyDef.initial_transform
# stays identity and "world frame" and "body-local frame" coincide at rest).
_T_A = [16.25811958, 0.0, 16.0]
_T_D = [10.00000095, -6.25812912, 15.99999905]

# Ground revolutes: axes as declared/verified in `build_rotate_oracle`
# (pin_rotation pivot=[0,0,0] local, i.e. the node's own rest origin) —
# exact by construction, not fit.
_GROUND_A_PIVOT = _T_A
_GROUND_A_AXIS = [1.0, 0.0, 0.0]
# `rotate-D` spins about world Y.  Its exported node transform carries ordinary
# float32 bake noise in the two *transverse* coordinates (x=10.00000095,
# z=15.99999905), while the fitted circular geometry gives the exact hinge line
# x=10, z=16.  Keeping those errors invents a 4.7e-9 constraint singular value:
# one nominal tangent direction then projects back onto the starting branch and
# the 2-D explorer degenerates into a few coordinate rays.  The coordinate
# along Y is free for a revolute pivot, so retain the node value there only.
_GROUND_D_PIVOT = [10.0, _T_D[1], 16.0]
_GROUND_D_AXIS = [0.0, 1.0, 0.0]

# Coupler revolutes: read from the mesh's own circular pin/bore edge loops
# (see module docstring) — axis and transverse position are exact; the
# coordinate along the axis is the bore midpoint, a free convention.
_AB_PIVOT = [10.0, 0.0, 21.0]
_AB_AXIS = [0.0, 0.0, 1.0]
_BC_PIVOT = [10.0, 4.0, 16.0]
_BC_AXIS = [0.0, 1.0, 0.0]
_CD_PIVOT = [5.0, 0.0, 16.0]
_CD_AXIS = [1.0, 0.0, 0.0]


def _link(name):
    return BodyDef(name=name, mesh=MeshSpec("file", {"path": _ASSET,
                                                      "node": f"rotate-{name}"}),
                   color=(0.36, 0.62, 0.86) if name in ("B", "C")
                   else (0.93, 0.53, 0.18))


_GROUND = BodyDef(
    name="__ground",
    mesh=MeshSpec("sphere", {"radius": 0.3, "center": [13.0, -3.0, 16.0]}),
    fixed=True, color=(0.45, 0.47, 0.50))

CASE = AtlasCaseSpec(
    name="rotate",
    mechanism=MechanismSpec(
        bodies=[_GROUND, _link("A"), _link("B"), _link("C"), _link("D")],
        joints=[
            JointDef("crank_A", "revolute", "__ground", "A",
                     pivot=_GROUND_A_PIVOT, axis=_GROUND_A_AXIS),
            JointDef("crank_D", "revolute", "__ground", "D",
                     pivot=_GROUND_D_PIVOT, axis=_GROUND_D_AXIS),
            JointDef("elbow_AB", "revolute", "A", "B",
                     pivot=_AB_PIVOT, axis=_AB_AXIS),
            JointDef("elbow_BC", "revolute", "B", "C",
                     pivot=_BC_PIVOT, axis=_BC_AXIS),
            JointDef("elbow_CD", "revolute", "C", "D",
                     pivot=_CD_PIVOT, axis=_CD_AXIS),  # the loop closure
        ],
        up_axis="z",
        gravity=9.8,
    ),
    seeds=[],
    contact_model="none",       # ideal joints only -> ideal_oracle's analytic path
    coordinates=[],             # non-injective in (theta_A, theta_D); see docstring
    actuators=[
        ActuatorDef("crank A torque", kind="joint_torque", target="crank_A", hi=40.0),
        ActuatorDef("crank D torque", kind="joint_torque", target="crank_D", hi=40.0),
    ],
    build=BuildOptions(
        sampling=SamplingOptions(
            # NOT the old contact dataset: its packed translations are in a
            # different frame convention (node-origin-relative, not baked-
            # mesh-at-rest — see the module docstring's t_new conversion) and
            # feeding it straight to this case's oracle would look like a
            # ~30-unit residual explosion. Explore fresh with the analytic
            # oracle instead, which is the whole point of this case.
            # step=0.08, then 0.2, were cautious steps up while a spurious
            # 4.68e-9 singular value (float32 GLB bake noise on ground-D's
            # transverse pivot coords, fixed above) made the Jacobian's null
            # space slightly wrong and large steps risky. With that gone, both
            # rest-pose tangent directions realize a full step exactly and need
            # no `hold` at all, so there's no more reason to stay well under
            # five_bar's own step=1.0 (which covered its full 360x360
            # actuator-angle range in 626 samples, frontier_emptied=True) — use
            # the same value here.
            # step=1.0 alone wasn't the real lever: `sample.py`'s
            # MAX_BODY_DISPLACEMENT (0.25) silently clamps every predictor step
            # regardless of `step`, which is why realized spacing topped out
            # around 0.4-0.7 instead of ~1.0 (median NN distance 0.40 on the
            # 164k-sample run). `predictor_body_displacement` is the per-case
            # override for that cap. A much coarser step (10, vs the cautious
            # 0.08 -> 0.2 -> 1.0 escalation) still converges cleanly post-
            # pivot-fix: residual stayed <=6e-12 and realized displacement
            # tracked the target (9.5 of 10) rather than snapping back, tested
            # at rest before committing to a full run.
            n_samples=5000, step=6.0, tangent_eps=0.01, max_seconds=360, graph_k=32,
            checkpoint_seconds=30,
            predictor_body_displacement=6.0),
        charts=ChartOptions(mode="mesh_atlas", overlap=0.25),
        canonical=CanonicalOptions(enabled=True, dimension=2, planarity=0.2,
                                   residual_discovery=True,
                                   autoencoder_charts=True),
        decoder=DecoderOptions(epochs=6000, smooth_lambda=1e-3),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=1500),
        expect=Expectations(dimension=2, n_bodies=4),
        tolerances=Tolerances(decoder_translation_rmse=0.1,
                              decoder_rotation_deg=3.0),
    ),
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "exact_canonical"},
               "exact": {"dt": 1e-3, "dynamics_backend": "exact_canonical"},
               "distilled": {"dt": 1e-3, "dynamics_backend": "distilled_canonical"}},
)


def VIEW_EMBED(q):
    """Viewer-only 3-D embedding: the moving B-link origin in physical space.

    The stored pose convention records non-fixed bodies in
    ``CASE.mechanism.body_order``.  This projection is deliberately only a
    physical display coordinate; it is not supplied to sampling or CAE fitting.
    """
    poses = np.asarray(q, float).reshape(len(q), -1, 3, 4)
    b_index = CASE.mechanism.body_order.index("B")
    return poses[:, b_index, :, 3]
