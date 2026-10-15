"""cases/pulley.py -- a five-roller compound pulley (block-and-tackle), one DOF.

A single rope threads sliderA -> roller1 -> roller2 -> roller3 -> roller4 ->
roller5 -> sliderB, both ends riding a prismatic joint.  A real rope is
inextensible but PAYS OUT between segments as the mechanism moves, which no
`JointDef` can express -- a joint fixes the relative pose of two named
bodies, not a scalar length-conservation law along a whole routed path.  So
this is not a cable simulation: every rope segment is a rigid rod of fixed
length ("rod1".."rod6"), and every roller a short rigid crank ("crank1"..
"crank5") pinned to a fixed ground axle, with each rod free to pivot at the
crank's tip.  That is the standard kinematic idealisation of a taut rope
redirected by a frictionless sheave -- the two segments leaving a pulley are
independently free to swing -- and it chains into a run of slider-crank /
four-bar loops sharing bodies, the exact technique `jansen_leg.py` uses for
its eight-bar leg.  Grubler confirms the chain stays at exactly one DOF for
any number of rollers, because every new rod's length equation is matched by
exactly one new crank angle:

    bodies   sliderA, sliderB, crank1..5, rod1..6   13 moving x 3 planar DOF = 39
    joints   2 prismatic (sliders to ground)
             5 revolute  (crank axles to ground)
            12 revolute  (each rod pinned at both ends)     19 joints, 38 removed
    DOF = 39 - 38 = 1

Geometry matches `pulley_leg.py` exactly (REST_POINTS there is this case's own
source of truth): five ground axles G1..G5 at y=2.0, evenly spaced x=-2..2.
Every crank points straight down at rest, so its tip sits at (x, 1.7) and the
rope threads tip-to-tip in one straight horizontal line at the rest pose --
every rod is exactly 1.0 except the closing one to sliderB, whose rail sits
0.7 (not 1.0) off the last tip so that hop is a genuine circle/line crossing
rather than a tangency.  Numbers below are written out rather than imported,
because a spec file is data (`cases/jansen.py` follows the same rule).

`pull` -- sliderA's own y-translation -- IS the coordinate: at d=1 there is no
separate reduced quantity to define, and `body_translation` reads it straight
off the recorded transforms with no per-frame sample field required.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

N_ROLLERS = 5
GROUND_X = (-2.0, -1.0, 0.0, 1.0, 2.0)
GROUND_Y = 2.0
CRANK_R = 0.3
REST_Y = GROUND_Y - CRANK_R                       # 1.7 -- crank tip height at rest
RAIL_SA_X = -3.0
RAIL_SB_X = 2.7
REST_SB_Y = 0.9858571571457155                     # circle(T5, 1.0) x line(x=2.7), lower root
SAFE_MARGIN = 0.5                                   # matches pulley_sample.py's SAFE_MARGIN
Z = [0.0, 0.0, 1.0]                                 # crank axles: planar hinge about +z

REST = {"SA": [RAIL_SA_X, REST_Y, 0.0], "SB": [RAIL_SB_X, REST_SB_Y, 0.0]}
for _i, _x in enumerate(GROUND_X):
    REST[f"G{_i+1}"] = [_x, GROUND_Y, 0.0]
    REST[f"T{_i+1}"] = [_x, REST_Y, 0.0]

ROLLER_COLOR = (0.85, 0.32, 0.36)
ROPE_COLOR = (0.36, 0.62, 0.86)
SLIDER_COLOR = (0.93, 0.53, 0.18)


def _crank(i):
    """Roller i's crank arm, with a flat disk (the roller wheel) welded at its axle."""
    name, g_name, t_name = f"crank{i+1}", f"G{i+1}", f"T{i+1}"
    arm = MeshSpec("link", {"p0": REST[g_name], "p1": REST[t_name], "thick": 0.06})
    wheel = MeshSpec("cylinder", {"radius": 0.35, "height": 0.12,
                                  "center": REST[g_name], "axis": Z, "sections": 24})
    return BodyDef(name=name, mesh=MeshSpec("compound", {"parts": [arm, wheel]}),
                   color=ROLLER_COLOR)


def _rod(i, a_name, b_name):
    return BodyDef(name=f"rod{i+1}",
                   mesh=MeshSpec("link", {"p0": REST[a_name], "p1": REST[b_name],
                                          "thick": 0.05}),
                   color=ROPE_COLOR)


def _slider(name, point_name):
    return BodyDef(name=name, mesh=MeshSpec("box", {"center": REST[point_name],
                                                     "half": [0.15, 0.15, 0.15]}),
                   color=SLIDER_COLOR)


ROD_ENDPOINTS = (("SA", "T1"), ("T1", "T2"), ("T2", "T3"), ("T3", "T4"),
                 ("T4", "T5"), ("T5", "SB"))


def _owner(joint_name):
    """The body that physically carries a rope-path point: a slider, or the
    crank whose tip that point is."""
    if joint_name == "SA":
        return "sliderA"
    if joint_name == "SB":
        return "sliderB"
    return f"crank{int(joint_name[1:])}"           # "T3" -> "crank3"

CASE = AtlasCaseSpec(
    name="pulley",
    mechanism=MechanismSpec(
        bodies=(
            [_crank(i) for i in range(N_ROLLERS)]
            + [_rod(i, a, b) for i, (a, b) in enumerate(ROD_ENDPOINTS)]
            + [_slider("sliderA", "SA"), _slider("sliderB", "SB")]
            + [BodyDef(name="__frame",
                      mesh=MeshSpec("link", {"p0": REST["G1"], "p1": REST["G5"],
                                             "thick": 0.15}),
                      fixed=True, color=(0.45, 0.47, 0.50))]
        ),
        joints=(
            [JointDef(f"axle{i+1}", "revolute", "__frame", f"crank{i+1}",
                      pivot=REST[f"G{i+1}"], axis=Z) for i in range(N_ROLLERS)]
            + [JointDef(f"rod{i+1}_a", "revolute", _owner(a), f"rod{i+1}",
                       pivot=REST[a], axis=Z)
               for i, (a, b) in enumerate(ROD_ENDPOINTS)]
            + [JointDef(f"rod{i+1}_b", "revolute", _owner(b), f"rod{i+1}",
                       pivot=REST[b], axis=Z)
               for i, (a, b) in enumerate(ROD_ENDPOINTS)]
            + [JointDef("railA", "prismatic", "__frame", "sliderA",
                       pivot=REST["SA"], axis=[0.0, 1.0, 0.0],
                       limit_lo=-SAFE_MARGIN, limit_hi=SAFE_MARGIN),
               JointDef("railB", "prismatic", "__frame", "sliderB",
                       pivot=REST["SB"], axis=[0.0, 1.0, 0.0],
                       limit_lo=-SAFE_MARGIN, limit_hi=SAFE_MARGIN)]
        ),
        up_axis="y",                     # rope travel is in-plane; gravity pulls it taut
        render_up_axis="z",
        gravity=9.8,
    ),
    seeds=[],
    contact_model="none",                # ideal joints only; no rope/roller contact modelled
    # Recorded transforms are the rigid motion FROM the rest assembly (mesh
    # verts are baked at rest-pose world coordinates, transform = identity
    # there), so `body_translation`'s T[..., 3] reads sliderA's DISPLACEMENT
    # from REST_Y, not its absolute height -- hence [-SAFE_MARGIN, SAFE_MARGIN].
    coordinates=[
        CoordinateDef(name="pull", topology="bounded",
                      lo=-SAFE_MARGIN, hi=SAFE_MARGIN,
                      source="body_translation", args={"body": "sliderA", "axis": "y"}),
    ],
    actuators=[
        ActuatorDef("pull force", kind="coordinate_force", target="pull", hi=200.0),
    ],
    build=BuildOptions(
        sampling=SamplingOptions(reuse="output/pulley_cycle_si.npz"),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=6000, smooth_lambda=1e-4, closure_weight=1.0),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=800),
        boundary=BoundaryOptions(kind="interval", margin=0.0),
        expect=Expectations(dimension=1, n_charts=1, n_bodies=13),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2),
    ),
    scenarios={"default": {"dt": 1e-3, "initial_u": [0.0]}},
)
