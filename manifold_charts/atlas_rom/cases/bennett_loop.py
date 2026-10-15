"""A Bennett linkage: the classical spatial overconstrained 4R loop.

Grubler's count for a 4-revolute spatial loop (3 moving links) is
6*3 - 5*4 = -2 -- generically rigid, the same story as `rotate.py`'s 5-bar.
Bennett's linkage moves anyway because opposite links share length and twist
(a1=a3=a, alpha1=alpha3=alpha, a2=a4=b, alpha2=alpha4=beta) under the classical
condition a/sin(alpha) = b/sin(beta).  Unlike `rotate.py` (whose special axes
were MEASURED off a recorded mesh), this one is the textbook closed-form
condition, so the four joint axes/pivots below are DERIVED, then verified
numerically before being written down:

  * built the loop as a chain of DH transforms (theta_i variable, alpha_i/a_i
    from the Bennett condition above, d_i=0),
  * solved the 6 loop-closure equations (T1 T2 T3 T4 = I) for (theta2,theta3,
    theta4) at a driven theta1 via Gauss-Newton, and
  * continued that solution across theta1 in [-3.0, 3.0] rad -- residual stays
    <1e-8 throughout, so this is a genuine finite 1-parameter family, not a
    first-order flex that only looks mobile in a linearized Jacobian (the
    same "instantaneous vs finite" trap noted in `rotate.py`'s neighbor,
    `revolute-manifold-needs-screw-continuation`).

a=1.0, alpha=0.9 rad, beta=0.6 rad, b = a*sin(beta)/sin(alpha) = 0.720826...
The rest pose below is theta1=1.0 rad (chosen off the theta1~0 configuration,
which is a real point on the curve but geometrically flat -- all 4 pivots
coplanar, volume 0 -- and not a useful rest pose to render or step away from).
The mechanism's own motion is periodic (closure holds continuously over the
full 2*pi range swept), so the driven coordinate is declared periodic, not
bounded.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

# Frame origins/z-axes at theta1=1.0 rad, from the DH loop-closure solve
# (T0=I is ground; T4 closes back to I).  Verified: |T4 - I| < 3e-10, and the
# four origin-to-origin rod lengths reproduce a/b to 1e-6.
_P0 = [0.0, 0.0, 0.0]
_Z0 = [0.0, 0.0, 1.0]
_P1 = [0.5403023059, 0.8414709848, 0.0]
_Z1 = [0.6590546141, -0.4231979425, 0.6216099683]
_P2 = [0.2509024281, 0.1948976424, -0.1333145236]
_Z2 = [0.1849814360, -0.2772261156, 0.9428086943]
_P3 = [-0.7208260899, 0.0, 0.0]
_Z3 = [0.0, 0.5645998747, 0.8253187759]

_THICK = 0.08


def _rod(name, p0, p1, color):
    p0a, p1a = np.array(p0), np.array(p1)
    mid = 0.5 * (p0a + p1a)
    d = p1a - p0a
    length = float(np.linalg.norm(d))
    axis = (d / length).tolist()
    return BodyDef(name=name,
                   mesh=MeshSpec("cylinder", {"radius": _THICK / 2,
                                              "height": length,
                                              "center": mid.tolist(),
                                              "axis": axis}),
                   color=color)


_GROUND = BodyDef(
    name="__ground",
    # Includes the loop's fourth (fixed) rod, P3->P0, purely as visual context
    # for the closed loop -- it carries no freedom, ground is one rigid body.
    mesh=MeshSpec("compound", {"parts": [
        MeshSpec("sphere", {"radius": 0.08, "center": _P0}),
        MeshSpec("cylinder", {"radius": _THICK / 2,
                              "height": 0.7208260899,
                              "center": [(_P3[0] + _P0[0]) / 2,
                                        (_P3[1] + _P0[1]) / 2,
                                        (_P3[2] + _P0[2]) / 2],
                              "axis": [(_P0[0] - _P3[0]) / 0.7208260899,
                                      (_P0[1] - _P3[1]) / 0.7208260899,
                                      (_P0[2] - _P3[2]) / 0.7208260899]}),
    ]}),
    fixed=True, color=(0.45, 0.47, 0.50))

CASE = AtlasCaseSpec(
    name="bennett_loop",
    mechanism=MechanismSpec(
        bodies=[_GROUND,
               _rod("link_1", _P0, _P1, (0.93, 0.53, 0.18)),
               _rod("link_2", _P1, _P2, (0.36, 0.62, 0.86)),
               _rod("link_3", _P2, _P3, (0.93, 0.53, 0.18))],
        joints=[
            JointDef("joint_1", "revolute", "__ground", "link_1",
                     pivot=_P0, axis=_Z0),
            JointDef("joint_2", "revolute", "link_1", "link_2",
                     pivot=_P1, axis=_Z1),
            JointDef("joint_3", "revolute", "link_2", "link_3",
                     pivot=_P2, axis=_Z2),
            JointDef("joint_4", "revolute", "link_3", "__ground",
                     pivot=_P3, axis=_Z3),      # the loop closure
        ],
        up_axis="z",
        gravity=9.8,
    ),
    seeds=[],
    contact_model="none",       # ideal joints only -> ideal_oracle's analytic path
    coordinates=[
        CoordinateDef("theta_1", topology="periodic", source="joint_angle",
                      args={"joint": "joint_1"}),
    ],
    actuators=[
        ActuatorDef("joint 1 torque", kind="joint_torque", target="joint_1", hi=20.0),
    ],
    build=BuildOptions(
        sampling=SamplingOptions(n_samples=1500, step=0.05, max_seconds=1800),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=4000, smooth_lambda=1e-3),
        dynamics=DynamicsOptions(distill=True, epochs=2000, n_teacher=1000),
        expect=Expectations(dimension=1, n_bodies=3),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0),
    ),
    scenarios={"default": {"dt": 1e-3, "initial_u": [1.0]}},
)
