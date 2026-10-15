"""cases/jansen.py — the crank-coupled Jansen leg, one PERIODIC coordinate.

The whole eight-bar linkage collapses to a single degree of freedom: the crank
phase.  That makes it the case that proves the periodic half of the contract —
`phi` is declared `periodic` with period 2*pi, which is what makes the decoder
encode it as Fourier features (exactly seam-continuous at the wrap, rather than
approximately) and what makes the runtime wrap it every step instead of letting
the coordinate run off to infinity.

Body frames follow the recorded transforms: each body's origin is at its centre of
mass with +x along its first two joints, so the meshes below are bars laid out in
that frame.  The numbers are the rest-frame joint coordinates of the standard
Jansen linkage (LENGTHS a..m), written out rather than recomputed, because a spec
file is data.

Bars are given a physical cross-section here, so mass and inertia come from the
mesh like every other case, rather than from a 2-D "mass per unit length".  That
means the coordinates below must be METRES, not Jansen's dimensionless linkage
units: a `link` mesh is fed to `density * volume`, so the unit-scale table (bars
15..150 across a 3.0 section) makes a 4450-tonne leg whose gravity torque is
7e8 N*m, and no plausible crank actuator can move it.  Everything here is the
standard linkage at scale 1/100 -- bars 0.15..0.62 m on a 3 cm section, a 4.45 kg
leg, peak gravity torque ~7.4 N*m -- which is the same metre scale every other
case is written in.  The reused cycle is recorded at the matching scale
(`jansen_sample.py --scale 0.01`), because the body-local meshes here and the
recorded body frames have to be in one unit system.
"""
from __future__ import annotations
import math

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

THICK = 0.03                      # bar cross-section, metres

# rest-frame joint coordinates in METRES, per body, measured from that body's
# centre of mass (the standard Jansen LENGTHS at scale 1/100)
LOCAL = {
    "crank": {"O": [-0.075, 0.0], "A": [0.075, 0.0]},
    "AC":    {"A": [-0.25, 0.0], "C": [0.25, 0.0]},
    "AD":    {"A": [-0.3095, 0.0], "D": [0.3095, 0.0]},
    "BD":    {"B": [-0.1965, 0.0], "D": [0.1965, 0.0]},
    "BCE":   {"B": [-0.156049398, -0.139644704], "C": [0.258950602, -0.139644704],
              "E": [-0.129950602, 0.260505084]},
    "EF":    {"E": [-0.197, 0.0], "F": [0.197, 0.0]},
    "DFG":   {"D": [-0.09476703, -0.18327674], "F": [0.27223297, -0.18327674],
              "G": [-0.17223297, 0.300561071]},
}
# which joint pairs are actually welded bars within each body
BARS = {
    "crank": [("O", "A")], "AC": [("A", "C")], "AD": [("A", "D")],
    "BD": [("B", "D")], "BCE": [("B", "C"), ("C", "E"), ("B", "E")],
    "EF": [("E", "F")], "DFG": [("D", "F"), ("F", "G"), ("D", "G")],
}
COLORS = {
    "crank": (0.93, 0.53, 0.18), "AC": (0.36, 0.62, 0.86), "AD": (0.24, 0.47, 0.74),
    "BD": (0.55, 0.72, 0.92), "BCE": (0.35, 0.74, 0.55), "EF": (0.72, 0.60, 0.88),
    "DFG": (0.85, 0.32, 0.36),
}


def _body(name):
    """A body is the union of its bars, each a thin box between two joints."""
    parts = [MeshSpec("link", {"p0": LOCAL[name][a] + [0.0],
                               "p1": LOCAL[name][b] + [0.0], "thick": THICK})
             for a, b in BARS[name]]
    mesh = parts[0] if len(parts) == 1 else MeshSpec("compound", {"parts": parts})
    return BodyDef(name=name, mesh=mesh, color=COLORS[name],
                   configuration="planar_rigid",
                   initial_transform=_pose_transform(name))


# ─── declared joints, derived from body membership at shared LOCAL points ──
#
# jansen.py originally declared no joints: "the recorded cycle is the
# linkage's own solved motion, and the generic pipeline only needs joints for
# the loop-closure loss and for re-exploring with the oracle." Re-exploring
# WITH the oracle (rather than only replaying the recorded cycle) requires
# those joints to actually exist, so they are derived here -- not guessed --
# from which bodies share which named LOCAL point, cross-checked against the
# specialized generator's own recording (output/jansen_cycle_si.npz).
#
# POSE0 is each body's rest-pose (tx, ty, theta) lifted from that recording's
# frame 0 (`transforms[0]`, world-frame). It is needed because every body's
# LOCAL dict is expressed relative to that body's OWN centre of mass, and
# JointDef.pivot is a WORLD-frame point at the declared rest pose -- without
# a real per-body world placement, two bodies sharing a physical pin (e.g.
# crank and AC at "A") would each convert the same declared pivot to a
# DIFFERENT (wrong) local offset. Verified against the recording directly:
#   - transforming every body's LOCAL[body][letter] by POSE0[body] reproduces
#     the recording's own `points[frame, joint_index]` to machine precision
#     (<1e-9), for ALL 721 recorded frames, not just frame 0 -- confirming
#     both the pivot values below and the body/point membership they encode.
#   - `points[:, i, :].std(axis=0)` is exactly zero only for joints "O" and
#     "B": these are the mechanism's two frame-fixed pivots (Jansen's linkage
#     needs two, not one -- Grubler's equation only closes to 1 DOF with 10
#     revolute joints, not 9, and the missing constraint is exactly a second
#     ground pivot at "B", shared by BD and BCE with no direct joint between
#     them otherwise).
#   - every body transform is a pure planar rotation about world Z with zero
#     z-translation across all 721 frames (checked directly on `transforms`),
#     confirming `configuration="planar_rigid"` and every joint axis below.
POSE0 = {
    "crank": (0.075, 0.0, 0.0),
    "AC": (-0.04506767548563903, 0.1563604872742133, 2.4658948443085644),
    "AD": (-0.059760535157864836, -0.22757585085040577, -2.315481044639538),
    "BD": (-0.32476053515786485, -0.2665758508504058, -1.2858386142179512),
    "BCE": (-0.45888270032686684, 0.11598344632829814, 1.2270429474429436),
    "EF": (-0.6701294017117553, -0.0995488001242581, -1.1647326082437524),
    "DFG": (-0.4400781830174872, -0.5712612399547841, 2.6457075227029914),
}


def _pose_transform(name):
    tx, ty, theta = POSE0[name]
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0, tx], [s, c, 0.0, ty], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _world_point(body, letter):
    """LOCAL[body][letter] transformed by POSE0[body] -- the world-frame
    pivot a JointDef declares, consistent with that body's initial_transform."""
    tx, ty, theta = POSE0[body]
    c, s = math.cos(theta), math.sin(theta)
    lx, ly = LOCAL[body][letter]
    return [c * lx - s * ly + tx, s * lx + c * ly + ty, 0.0]


_Z = [0.0, 0.0, 1.0]

JOINTS = [
    # crank's own axle: the mechanism's driven, frame-fixed pivot
    JointDef("O", "revolute", "crank", None, pivot=_world_point("crank", "O"), axis=_Z),
    # A: crank, AC, AD meet at one pin -- two independent revolutes,
    # both against the crank, fully pin all three together
    JointDef("A_crank_AC", "revolute", "crank", "AC", pivot=_world_point("crank", "A"), axis=_Z),
    JointDef("A_crank_AD", "revolute", "crank", "AD", pivot=_world_point("crank", "A"), axis=_Z),
    JointDef("C", "revolute", "AC", "BCE", pivot=_world_point("AC", "C"), axis=_Z),
    # D: AD, BD, DFG meet at one pin -- two independent revolutes against
    # the ternary link DFG
    JointDef("D_AD_DFG", "revolute", "AD", "DFG", pivot=_world_point("AD", "D"), axis=_Z),
    JointDef("D_BD_DFG", "revolute", "BD", "DFG", pivot=_world_point("BD", "D"), axis=_Z),
    # B: the mechanism's second frame-fixed pivot -- BD and BCE each pin
    # independently to the SAME fixed world point, with no direct joint
    # between them (confirmed: both have zero variance and coincide exactly)
    JointDef("B_ground_BD", "revolute", "BD", None, pivot=_world_point("BD", "B"), axis=_Z),
    JointDef("B_ground_BCE", "revolute", "BCE", None, pivot=_world_point("BCE", "B"), axis=_Z),
    JointDef("E", "revolute", "BCE", "EF", pivot=_world_point("BCE", "E"), axis=_Z),
    JointDef("F", "revolute", "EF", "DFG", pivot=_world_point("EF", "F"), axis=_Z),
    # G (DFG's third corner) is the mechanism's free foot tip -- no joint
]


CASE = AtlasCaseSpec(
    name="jansen",
    mechanism=MechanismSpec(
        # JOINTS above are derived from, and verified against, the specialized
        # generator's own recording (see the comment at their definition) --
        # not asserted independently of it, so declaring them does not gate
        # the build on a constraint the data never actually satisfied.
        bodies=[_body(name) for name in
                ("crank", "AC", "AD", "BD", "BCE", "EF", "DFG")],
        joints=JOINTS,
        up_axis="y",
        gravity=9.8,
    ),
    seeds=[],
    contact_model="none",
    coordinates=[
        CoordinateDef(name="phi", topology="periodic", period=6.283185307179586,
                      source="sample_field", args={"key": "recorded_angle:crank:z"}),
    ],
    actuators=[
        # at d=1 the coordinate IS the crank angle, so a crank torque needs no
        # Jacobian — the degenerate case of the same actuator rule
        # peak gravity torque on the crank is ~7.4 N*m at this scale, so the
        # limit has to clear it with margin or the slider cannot move the leg
        ActuatorDef("crank torque", kind="joint_torque", target="O", hi=40.0),
    ],
    build=BuildOptions(
        # Explore this declared joint mechanism directly.  The lifted crank
        # angle is reconstructed from local continuation edges by sampling.py.
        sampling=SamplingOptions(sampler="analytic"),
        # n_harmonics must be at least the winding of the fastest body over one
        # crank turn, or a multi-turn body simply cannot be represented
        decoder=DecoderOptions(epochs=6000, n_harmonics=6, smooth_lambda=1e-4),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=721),
        expect=Expectations(dimension=1, n_bodies=7),
        # the translation gate is a length, so it rides the 1/100 rescale too;
        # rotation is scale-free and stays where it was
        tolerances=Tolerances(decoder_translation_rmse=0.005,
                              decoder_rotation_deg=1.0),
    ),
    scenarios={"default": {"dt": 1e-3, "initial_u": [0.0]}},
)
