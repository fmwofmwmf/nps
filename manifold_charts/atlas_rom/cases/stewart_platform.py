"""cases/stewart_platform.py — a rotary-actuated Stewart-Gough platform.

Ported from the `stewart` system in Sharp et al., "Data-Free Learning of
Reduced-Order Kinematics" (SIGGRAPH 2023), `src/rigid3d_model.py`.  Their model
is 14 rigid bodies (a fixed base, six cranks, six coupler struts, a moving top
plate) wired together by penalty joints; the base->crank joints carry a real
axis-alignment term (a hinge) while the crank->strut and strut->top joints pass
`0.01*v` as the alignment axis, so their alignment penalty is ~constant and
those ends behave as ball joints.  The mechanism is therefore a 6-RSS parallel
platform driven by six base cranks -- a 6-DOF moving plate.

This port keeps all thirteen moving bodies and the exact rest geometry
(`scale = 5.0`, the 2*pi/3 symmetry, the attachment vectors straight out of
`rigid3d_model.py`), and translates the joints to ideal ones for the analytic
oracle (`contact_model="none"`):

  * base -> crank : revolute  (the actuated hinge, axis from the paper's `v`)
  * crank -> strut: spherical (the paper's near-zero alignment)
  * strut -> plate: universal (Hooke) -- a spherical joint with the idle strut
    spin removed.  A plain 6-RSS leaves each strut free to roll about its own
    long axis: six energy-free nullspace directions that would inflate the
    ideal-joint configuration manifold from 6-D to 12-D.  The universal joint's
    one extra perpendicularity constraint (pin_a fixed in the strut stays
    perpendicular to pin_b fixed in the plate; both chosen across the strut
    axis and mutually perpendicular at rest) removes exactly that spin, so the
    manifold is a clean 6-D -- the platform's own SE(3) freedom.

Row count check (13 bodies, 156 ambient dofs): 78 orthonormality + 6*5 revolute
+ 6*3 spherical + 6*4 universal = 150 independent constraints => dim 6.

The declared coordinates are the moving plate's pose (three translations, three
rotation angles about the world axes).  Over the platform's bounded workspace
that is single-valued -- unlike the closed-chain planar cases, there is no
elbow-in / elbow-out branch here -- so it serves as a global-ish chart; the
atlas still verifies the dimension from measured pose secants.

NOTE: this is a large case (13 bodies, a 6-D manifold).  The build options
below are conservative starting points, not a tuned recipe; expect the sampling
and decoder budgets to need raising for a ROM that passes every gate.  A trial
explore confirmed the intrinsic dimension: 5535 samples, tangent rank 6 at
*every* one (a plain 6-RSS -- both strut ends spherical -- would read 12).  The
advancing front is slow here: ~1000 s for those 5535 samples, since each
frontier pop takes a full 156x156 Jacobian SVD per tangent probe, and
`max_seconds` is only checked once per pop.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

_ASSET = "assets/neural_physics_subspaces/stewart"
_SCALE = 5.0

# One entry per leg, from the rest geometry in `rigid3d_model.py` (scale 5.0,
# R = rot_y(2*pi/3)).  `pin_a`/`pin_b` are the universal joint's cross-pins:
# both perpendicular to the rest strut axis and to each other, so the joint
# holds the strut's roll while leaving two bending freedoms.
_LEGS = [
    dict(base=(-0.090000, 0.107500, -0.224280), hinge=(0.0, 0.0, 1.0),
         mid=(-0.015000, 0.107500, -0.259280), top=(-0.160795, 0.411110, -0.113430),
         pin_a=(-0.707240, 0.000000, -0.706973), pin_b=(-0.584822, -0.561880, 0.585042)),
    dict(base=(-0.239235, 0.107500, 0.034200), hinge=(0.866025, 0.0, -0.5),
         mid=(-0.232045, 0.107500, 0.116650), top=(-0.178560, 0.411110, -0.082440),
         pin_a=(0.965757, 0.000000, 0.259448), pin_b=(0.214645, -0.561740, -0.798984)),
    dict(base=(-0.149232, 0.107500, 0.190082), hinge=(0.866025, 0.0, -0.5),
         mid=(-0.217043, 0.107500, 0.142630), top=(-0.017836, 0.411110, 0.195968),
         pin_a=(-0.258637, 0.000000, 0.965975), pin_b=(0.799072, -0.561880, 0.213949)),
    dict(base=(0.149236, 0.107500, 0.190084), hinge=(-0.866025, 0.0, -0.5),
         mid=(0.217044, 0.107500, 0.142632), top=(0.017885, 0.411110, 0.195857),
         pin_a=(-0.258190, 0.000000, -0.966094), pin_b=(-0.799263, -0.561740, 0.213604)),
    dict(base=(0.239232, 0.107500, 0.034198), hinge=(-0.866025, 0.0, -0.5),
         mid=(0.232043, 0.107500, 0.116650), top=(0.178631, 0.411110, -0.082538),
         pin_a=(0.965877, 0.000000, -0.259001), pin_b=(-0.214251, -0.561880, -0.798991)),
    dict(base=(0.089999, 0.107500, -0.224284), hinge=(0.0, 0.0, 1.0),
         mid=(0.015001, 0.107500, -0.259282), top=(0.160675, 0.411110, -0.113417),
         pin_a=(-0.707567, 0.000000, 0.706646), pin_b=(0.584618, -0.561740, 0.585380)),
]

_CRANK_COLOR = (0.93, 0.53, 0.18)
_STRUT_COLOR = (0.36, 0.62, 0.86)


def _file_body(name, obj, color, *, fixed=False):
    return BodyDef(name=name,
                   mesh=MeshSpec("file", {"path": f"{_ASSET}/{obj}", "scale": _SCALE}),
                   fixed=fixed, color=color)


_bodies = [_file_body("__ground", "stewart-base.obj", (0.45, 0.47, 0.50), fixed=True)]
for _i in range(1, 7):
    _bodies.append(_file_body(f"crank_{_i}", f"stewart-arm{_i}.obj", _CRANK_COLOR))
for _i in range(1, 7):
    _bodies.append(_file_body(f"strut_{_i}", f"stewart-strut{_i}.obj", _STRUT_COLOR))
_bodies.append(_file_body("platform", "stewart-top.obj", (0.80, 0.80, 0.83)))

_joints = []
for _i, _leg in enumerate(_LEGS, start=1):
    _joints.append(JointDef(f"drive_{_i}", "revolute", "__ground", f"crank_{_i}",
                            pivot=_leg["base"], axis=_leg["hinge"]))
    _joints.append(JointDef(f"knee_{_i}", "spherical", f"crank_{_i}", f"strut_{_i}",
                            pivot=_leg["mid"]))
    _joints.append(JointDef(f"ankle_{_i}", "universal", f"strut_{_i}", "platform",
                            pivot=_leg["top"], axis=_leg["pin_a"], axis2=_leg["pin_b"]))

CASE = AtlasCaseSpec(
    name="stewart_platform",
    mechanism=MechanismSpec(
        bodies=_bodies,
        joints=_joints,
        up_axis="y",                      # the paper poses the platform along +y
        gravity=9.8,
    ),
    seeds=[],
    contact_model="none",                 # ideal joints -> ideal_oracle analytic path
    coordinates=[
        CoordinateDef("plate_x", source="body_translation",
                      args={"body": "platform", "axis": "x"}),
        CoordinateDef("plate_y", source="body_translation",
                      args={"body": "platform", "axis": "y"}),
        CoordinateDef("plate_z", source="body_translation",
                      args={"body": "platform", "axis": "z"}),
        CoordinateDef("plate_rx", source="body_angle",
                      args={"body": "platform", "axis": (1, 0, 0), "reference": "identity"}),
        CoordinateDef("plate_ry", source="body_angle",
                      args={"body": "platform", "axis": (0, 1, 0), "reference": "identity"}),
        CoordinateDef("plate_rz", source="body_angle",
                      args={"body": "platform", "axis": (0, 0, 1), "reference": "identity"}),
    ],
    actuators=[
        ActuatorDef(f"drive {_i} torque", kind="joint_torque",
                    target=f"drive_{_i}", hi=30.0)
        for _i in range(1, 7)
    ],
    build=BuildOptions(
        # Coarse full-6-D cover: at step ~1.1 (spacing 0.6*step ~= 0.66 against a
        # manifold of pose-diameter ~4) the advancing front empties its frontier
        # around 10k samples in well under an hour.  This is a connectivity /
        # rank smoke test over the WHOLE manifold, not decoder-grade data -- a
        # linear tangent step this large overshoots the curvature.  Drop step to
        # ~0.15 (and the budget to hours) for a cover a chart fit can use.
        # `tangent_eps` stays small so the secant-PCA rank read is unaffected by
        # the coarse walking step.
        sampling=SamplingOptions(n_samples=12000, step=1.1, max_seconds=3000,
                                 tangent_eps=0.03),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=6000, smooth_lambda=1e-3, closure_weight=1.0),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=2000),
        expect=Expectations(dimension=6, n_bodies=13),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2),
    ),
    scenarios={"default": {"dt": 1e-3}},
)


def VIEW_EMBED(q):
    """Display-only: place each sample by the moving plate's translation.

    The 6-D platform C-space has no faithful 3-D picture; the plate centroid
    path at least shows the workspace shell for `viewers.view_sample_set`.
    Recomputed from `q` alone, never passed to the fitter.
    """
    import numpy as np
    T = np.asarray(q, float).reshape(len(q), 13, 3, 4)
    return T[:, 12, :, 3]
