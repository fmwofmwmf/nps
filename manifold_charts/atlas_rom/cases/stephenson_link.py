"""cases/stephenson_link.py — the Stephenson link motion of `Screenshot 2026-08-02 034312.png`
(plate 185 of the old kinematics atlases): a reversing valve gear.

WHAT THE FIGURE SHOWS, right to left:

  * a crankshaft carrying TWO ECCENTRICS side by side (the two straps drawn as
    concentric rings around the hatched shaft),
  * two ECCENTRIC RODS running left to the two ends of a curved, slotted
    EXPANSION LINK,
  * a DIE BLOCK riding in that slot, pinned to the lower end of a ROCKER that
    swings on a fixed pedestal standing on the engine bed,
  * the rocker's upper end driving the VALVE ROD, which slides horizontally
    through a gland into the steam chest (the hatched casting on the left),
  * a LIFTING LINK hung from a REVERSING LEVER — the long arm at the top left
    whose pointer sweeps the notched quadrant — which raises and lowers the
    expansion link and so slides the die from one end of the slot to the other.

That last member is what makes the gear a *reversing* gear, and it is why this
case has two degrees of freedom rather than one: the crank turns, and
independently the reverser sets where in the slot the die sits.  On the real
engine the quadrant's notches latch the lever; here it is left free, so the
quadrant is rendered but carries no constraint.

DOF, counted the way the mechanism actually closes (planar, 10 driven bodies):

    30  =  3 * 10 planar freedoms
   -28  =  14 ideal joints * 2 constraints
   ----
     2  =  crank angle + reverser position

THE SLOT IS A REVOLUTE JOINT.  This case has no contact at all, and that is a
statement about the geometry rather than a convenience.  A Stephenson slot is
struck with a radius equal to the eccentric rod length about a link-fixed
centre (`ARC_C`) — that is what puts the rod pins on the arc — and the die
rides that arc staying radially aligned.  Sliding the die along the slot is
therefore *exactly* a rotation about `ARC_C`, verified to 8.9e-16 m of vertex
mismatch over +-8 degrees of travel.  So the pair is `die_slot`: a revolute
between the link and the die, pivot `ARC_C`, 3.79 m off the die's own body.

That is worth doing rather than modelling the slot with meshes in contact:
`ideal_oracle` handles the whole mechanism analytically (a Gauss-Newton
root-find and one SVD, on the CPU) instead of paying a GPU IPC solve to
rediscover a constraint the spec can simply state.

What contact WOULD have given for free is the closed ends of the slot, and
those are declared instead as `limit_lo`/`limit_hi` on `die_slot` — the joint's
own travel, which is what the stop physically is.  A `WorkspaceBound` cannot
express it: it bounds the die's absolute translation, and because the link
swings underneath, the tightest such bound that admits every legal pose also
admits 80% of the poses where the die has left the link entirely.

GEOMETRY is traced from the figure.  Pixel coordinates of every pin were read
off the screenshot and are kept below as the source data; the world frame puts
the rocker fulcrum at the origin, +y up in the plane of the drawing, and scales
by `SCALE` metres per pixel.  Three quantities are NOT traced but derived, so
that the assembly closes exactly instead of to woodcut accuracy:

  * the slot arc.  In a Stephenson gear the slot is struck with a radius equal
    to the eccentric rod length and the two rod pins lie ON that arc.  `ARC_C`
    is solved from the traced end pins and `ROD_LEN`.
  * the eccentric angles.  `PHASE_FORE` / `PHASE_BACK` are solved so that each
    rod is exactly `ROD_LEN` long at the drawn pose.  They come out 90.8 deg and
    -72.7 deg, i.e. very nearly opposed with an ~8 deg angle of advance, which
    is what a link motion is supposed to look like.
  * the die pin.  Traced slightly off the slot centre line; projected back onto
    the arc so the die actually sits in its slot.

WHAT IT DOES, checked by solving the loop independently of the pipeline (all
four closures to 2e-15 over a full revolution, continued in crank angle):

    reverser   die's place in the slot   valve travel
      -3 deg          0.16                  0.81       full forward gear
       0 deg          0.22                  0.61       <- the pose drawn
       6 deg          0.35                  0.31
      12 deg          0.47                  0.06       mid gear: valve nearly dead
      18 deg          0.60                  0.18
      24 deg          0.73                  0.40
      30 deg          0.86                  0.61       full back gear

which is the whole point of the gear: notching the reverser toward mid gear
shortens the valve's travel to nothing and then reopens it running the other
way.  Both ends of the reverser's range are stopped by the die reaching the
closed end of the slot — a real contact boundary, not a declared bound.  Link
slip over one revolution is 0.14 of the slot in full gear and 0.01 at mid gear.

DEPTH.  The mechanism is planar — every hinge is about +z and gravity acts in
the xy-plane — but the parts are stacked across the drawing's depth the way the
real machine is, so that members which merely cross in the figure do not share
a plane: the eccentric rods straddle the link, the rocker stands in front of
them, the valve rod in front of that, and the lifting link and reversing lever
in front of everything.  `LAYER` is the whole of that bookkeeping; bosses
bridge the gaps where a pin has to reach between two layers.
"""
from __future__ import annotations

import math

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, CanonicalOptions, ChartOptions, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances, CoordinateDef

Z = [0.0, 0.0, 1.0]                       # every pin is a planar hinge about +z
X = [1.0, 0.0, 0.0]                       # the valve rod's gland


# Display-only mapping consumed by both manifold diagnostic viewers.  The
# values are evaluated through CoordinateMap from the recorded poses; this is
# not supplied to sampling, chart discovery, or training.
VISUAL_UVS = {"kind": "driver_coordinates", "coordinates": ("crank", "reverser")}

# ─── the figure ───────────────────────────────────────────────────────────────
# Pixel coordinates read off the screenshot (origin top-left, y down), and the
# map into the world frame: rocker fulcrum at the origin, +y up.
SCALE = 0.02                              # metres per pixel
PX_ORIGIN = (302.6, 150.3)                # the rocker fulcrum


def _w(px, py):
    """Traced pixel -> world xy (z is supplied per-part by `LAYER`)."""
    return ((px - PX_ORIGIN[0]) * SCALE, -(py - PX_ORIGIN[1]) * SCALE)


SHAFT = _w(500.0, 202.0)                  # crankshaft centre
PIN_A = _w(310.5, 178.0)                  # fore eccentric rod -> link, upper end
PIN_B = _w(317.0, 244.0)                  # back eccentric rod -> link, lower end
DIE_TRACED = _w(300.5, 190.0)             # die pin, before projection onto the arc
ROCK_TOP = _w(304.0, 104.0)               # rocker -> valve link
VALVE_PIN = _w(252.4, 110.3)              # valve link -> valve rod
GLAND = _w(207.0, 110.3)                  # where the valve rod enters the chest
LIFT_Y = _w(268.0, 82.0)[1]               # height of the lifting link's upper pin
LEVER_TIP = _w(31.7, 60.0)                # the reversing lever's pointer
BED_Y = _w(500.0, 160.0)[1]               # the engine bed's top face
CASTING_LO = _w(88.0, 271.0)              # steam chest / cylinder casting
CASTING_HI = _w(207.0, 90.0)

# ─── derived geometry ─────────────────────────────────────────────────────────

ROD_LEN = 3.79                            # eccentric rod == slot arc radius
ECC_THROW = 0.30
SLOT_HALF = 0.09                          # half the slot's clear width
RAIL_THICK = 0.24                         # the link's rails, square in section
DIE_RADIAL = 0.06                         # so the die has 0.03 clearance a side
DIE_TANGENT = 0.16
BAR = 0.10                                # a plain bar's section
SADDLE_Z = (0.09, 0.29)                   # the bridge over the slot, clear of the die


def _sub(p, q):
    return (p[0] - q[0], p[1] - q[1])


def _norm(v):
    return math.hypot(v[0], v[1])


def _arc_centre(a, b, radius):
    """Centre of the radius-`radius` arc through `a` and `b`, on the +x side."""
    d = _sub(b, a)
    chord = _norm(d)
    u = (d[0] / chord, d[1] / chord)
    n = (u[1], -u[0])                     # right of a->b, i.e. toward the shaft
    if n[0] < 0.0:
        n = (-n[0], -n[1])
    h = math.sqrt(radius * radius - 0.25 * chord * chord)
    mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    return (mid[0] + h * n[0], mid[1] + h * n[1])


def _on_arc(angle, radius=ROD_LEN):
    return (ARC_C[0] + radius * math.cos(angle), ARC_C[1] + radius * math.sin(angle))


def _eccentric_phase(pin, sign):
    """Shaft angle placing an eccentric exactly `ROD_LEN` from `pin`.

    Law of cosines on the triangle (shaft, eccentric, pin); `sign` picks which
    of the two roots — the fore eccentric stands above the shaft, the back one
    below, as drawn.
    """
    d = _sub(pin, SHAFT)
    r = _norm(d)
    alpha = math.atan2(d[1], d[0])
    cos_off = (r * r + ECC_THROW ** 2 - ROD_LEN ** 2) / (2.0 * ECC_THROW * r)
    return alpha + sign * math.acos(max(-1.0, min(1.0, cos_off)))


ARC_C = _arc_centre(PIN_A, PIN_B, ROD_LEN)


def _arc_angle(p, after=None):
    """Polar angle about `ARC_C`, unwrapped to run forward from `after`.

    The slot straddles the -x axis, so the raw atan2 branch cuts straight
    through it and the end angles would come out in the wrong order.
    """
    ang = math.atan2(p[1] - ARC_C[1], p[0] - ARC_C[0])
    if after is not None:
        while ang < after:
            ang += 2.0 * math.pi
    return ang


ANG_A = _arc_angle(PIN_A)
ANG_B = _arc_angle(PIN_B, after=ANG_A)
ANG_MID = 0.5 * (ANG_A + ANG_B)
# The slot runs a little past both end pins so the die reaches full gear.
ANG_LO, ANG_HI = ANG_A - 0.022, ANG_B + 0.022

ANG_DIE = _arc_angle(DIE_TRACED, after=ANG_A)
DIE_PIN = _on_arc(ANG_DIE)                # the traced die pin, put back in its slot

PHASE_FORE = _eccentric_phase(PIN_A, -1.0)
PHASE_BACK = _eccentric_phase(PIN_B, +1.0)
ECC_FORE = (SHAFT[0] + ECC_THROW * math.cos(PHASE_FORE),
            SHAFT[1] + ECC_THROW * math.sin(PHASE_FORE))
ECC_BACK = (SHAFT[0] + ECC_THROW * math.cos(PHASE_BACK),
            SHAFT[1] + ECC_THROW * math.sin(PHASE_BACK))

# WHERE THE LIFTING LINK PICKS THE LINK UP.  The figure hangs it from a lug that
# reaches about half a metre to the left of the slot, and at that lever arm the
# link's own +-25 deg oscillation slides the whole slot along itself: the die is
# driven a third of the slot's length per revolution and runs off the end well
# before the reverser has been moved anywhere.  So the pin goes where a link
# saddle actually goes — on the slot's centre line at mid length, straddling the
# slot in DEPTH rather than reaching around it in the plane.  Rotation about a
# point on that line moves the slot across itself instead of along itself, and
# the slip drops to what a link motion is supposed to have.
LUG = _on_arc(ANG_MID)
LIFT_TOP = (LUG[0] - 0.35, LIFT_Y)

# The reversing lever is one straight bar with its fulcrum on it.
_t = (_w(201.7, 74.5)[0] - LEVER_TIP[0]) / (LIFT_TOP[0] - LEVER_TIP[0])
LEVER_PIVOT = (LEVER_TIP[0] + _t * (LIFT_TOP[0] - LEVER_TIP[0]),
               LEVER_TIP[1] + _t * (LIFT_TOP[1] - LEVER_TIP[1]))
QUADRANT_R = 2.75                         # the notched sector, struck about the fulcrum

# closure of the traced-plus-derived assembly, asserted rather than trusted
assert abs(_norm(_sub(PIN_A, ECC_FORE)) - ROD_LEN) < 1e-9
assert abs(_norm(_sub(PIN_B, ECC_BACK)) - ROD_LEN) < 1e-9
assert abs(_norm(_sub(PIN_A, ARC_C)) - ROD_LEN) < 1e-9
assert abs(_norm(_sub(PIN_B, ARC_C)) - ROD_LEN) < 1e-9
assert ANG_A < ANG_DIE < ANG_B            # the die starts inside its slot

# ─── depth ────────────────────────────────────────────────────────────────────
# Where each part sits across the drawing.  Members that merely cross in the
# figure are given separate layers; only the die and the slot share one.
#
# The split is the machine's own: the saddle and the reversing gear ride on the
# FRONT of the expansion link, the die's pin comes out the BACK to the rocker
# and the valve train.  That is what lets the die traverse the whole slot
# without running into its own suspension.
LAYER = {
    "valve_rod": -0.78,
    "valve_link": -0.78,
    "rocker": -0.60,
    "rod_back": -0.40,
    "ecc_back": -0.22,
    "expansion_link": 0.00,
    "die_block": 0.00,
    "shaft": 0.00,
    "ecc_fore": +0.22,
    "rod_fore": +0.40,
    "lifting_link": +0.96,
    "reverse_lever": +1.14,
    "quadrant": +1.28,
    "frame": 0.00,
}


def _p(xy, layer):
    return [xy[0], xy[1], LAYER[layer]]


def _pivot(xy, layer_a, layer_b):
    """A pin between two layers is drawn at the midpoint of the two."""
    return [xy[0], xy[1], 0.5 * (LAYER[layer_a] + LAYER[layer_b])]


def _bar(p0, p1, thick=BAR):
    return MeshSpec("link", {"p0": list(p0), "p1": list(p1), "thick": thick})


def _boss(xy, radius, z_lo, z_hi):
    """A cylindrical pin boss bridging two layers."""
    return MeshSpec("cylinder", {"radius": radius, "height": z_hi - z_lo,
                                 "center": [xy[0], xy[1], 0.5 * (z_lo + z_hi)],
                                 "axis": Z, "sections": 20})


def _arc_bars(radius, a0, a1, thick, z, segments=24):
    """A circular rail about `ARC_C`, faceted finely enough that the die does not
    feel the facets (one segment's sagitta is ~1e-4 against 0.03 of clearance)."""
    pts = []
    for i in range(segments + 1):
        ang = a0 + (a1 - a0) * i / segments
        pts.append([ARC_C[0] + radius * math.cos(ang),
                    ARC_C[1] + radius * math.sin(ang), z])
    return [_bar(pts[i], pts[i + 1], thick) for i in range(segments)]


# ─── bodies ───────────────────────────────────────────────────────────────────

STEEL = (0.55, 0.60, 0.66)
BRONZE = (0.80, 0.60, 0.28)
IRON = (0.45, 0.47, 0.50)
RED = (0.86, 0.32, 0.26)
BLUE = (0.36, 0.62, 0.86)

_link_parts = (
    _arc_bars(ROD_LEN + SLOT_HALF + 0.5 * RAIL_THICK, ANG_LO, ANG_HI,
              RAIL_THICK, LAYER["expansion_link"])
    + _arc_bars(ROD_LEN - SLOT_HALF - 0.5 * RAIL_THICK, ANG_LO, ANG_HI,
                RAIL_THICK, LAYER["expansion_link"])
    + [
        # the closed ends of the slot
        _bar(_p(_on_arc(ANG_LO, ROD_LEN - SLOT_HALF - RAIL_THICK), "expansion_link"),
             _p(_on_arc(ANG_LO, ROD_LEN + SLOT_HALF + RAIL_THICK), "expansion_link"),
             RAIL_THICK),
        _bar(_p(_on_arc(ANG_HI, ROD_LEN - SLOT_HALF - RAIL_THICK), "expansion_link"),
             _p(_on_arc(ANG_HI, ROD_LEN + SLOT_HALF + RAIL_THICK), "expansion_link"),
             RAIL_THICK),
        # end-pin bosses, reaching out to the two eccentric rods
        _boss(PIN_A, 0.13, 0.12, LAYER["rod_fore"] + 0.05),
        _boss(PIN_B, 0.13, LAYER["rod_back"] - 0.05, -0.12),
        # the saddle: a bridge thrown across the FRONT of the slot at mid length,
        # standing off far enough that the die passes underneath it, and the
        # column that carries the lifting link's pin up out of the way
        _bar([_on_arc(ANG_MID, ROD_LEN - SLOT_HALF - RAIL_THICK)[0],
              _on_arc(ANG_MID, ROD_LEN - SLOT_HALF - RAIL_THICK)[1],
              0.5 * (SADDLE_Z[0] + SADDLE_Z[1])],
             [_on_arc(ANG_MID, ROD_LEN + SLOT_HALF + RAIL_THICK)[0],
              _on_arc(ANG_MID, ROD_LEN + SLOT_HALF + RAIL_THICK)[1],
              0.5 * (SADDLE_Z[0] + SADDLE_Z[1])],
             SADDLE_Z[1] - SADDLE_Z[0]),
        _boss(LUG, 0.10, SADDLE_Z[1] - 0.06, LAYER["lifting_link"] + 0.05),
    ])

_rocker_parts = [
    _bar(_p(DIE_PIN, "rocker"), _p((0.0, 0.0), "rocker"), BAR),
    _bar(_p((0.0, 0.0), "rocker"), _p(ROCK_TOP, "rocker"), BAR),
    _boss(ROCK_TOP, 0.09, LAYER["valve_link"] - 0.045, LAYER["rocker"] - 0.05),
]

_shaft_parts = [
    MeshSpec("cylinder", {"radius": 0.18, "height": 1.40,
                          "center": [SHAFT[0], SHAFT[1], 0.0], "axis": Z,
                          "sections": 24}),
    MeshSpec("cylinder", {"radius": 0.48, "height": 0.18,
                          "center": _p(ECC_FORE, "ecc_fore"), "axis": Z,
                          "sections": 28}),
    MeshSpec("cylinder", {"radius": 0.48, "height": 0.18,
                          "center": _p(ECC_BACK, "ecc_back"), "axis": Z,
                          "sections": 28}),
]

_lever_parts = [
    _bar(_p(LEVER_TIP, "reverse_lever"), _p(LIFT_TOP, "reverse_lever"), 0.12),
    _boss(LIFT_TOP, 0.09, LAYER["lifting_link"] - 0.05, LAYER["reverse_lever"] + 0.06),
]

_frame_parts = [
    # the cylinder / steam-chest casting the valve rod enters
    MeshSpec("box", {"center": [0.5 * (CASTING_LO[0] + CASTING_HI[0]),
                                0.5 * (CASTING_LO[1] + CASTING_HI[1]), 0.0],
                     "half": [0.5 * abs(CASTING_HI[0] - CASTING_LO[0]),
                              0.5 * abs(CASTING_HI[1] - CASTING_LO[1]), 0.60]}),
    # the engine bed, in two spans: the expansion link swings down through the
    # gap between them, exactly as it does past the frame line in the figure
    _bar([GLAND[0], BED_Y - 0.08, 0.0], [-0.80, BED_Y - 0.08, 0.0], 0.16),
    _bar([0.90, BED_Y - 0.08, 0.0], [SHAFT[0] + 0.4, BED_Y - 0.08, 0.0], 0.16),
    # the rocker's pedestal: two cheeks straddling the lever, not a solid block
    MeshSpec("box", {"center": [0.0, 0.5 * BED_Y - 0.02, LAYER["rocker"] - 0.13],
                     "half": [0.16, 0.5 * abs(BED_Y) + 0.06, 0.05]}),
    MeshSpec("box", {"center": [0.0, 0.5 * BED_Y - 0.02, LAYER["rocker"] + 0.13],
                     "half": [0.16, 0.5 * abs(BED_Y) + 0.06, 0.05]}),
    # the crankshaft's pillow blocks, outboard of the shaft itself
    MeshSpec("box", {"center": [SHAFT[0], 0.5 * (SHAFT[1] + BED_Y), -0.82],
                     "half": [0.22, 0.5 * abs(SHAFT[1] - BED_Y) + 0.02, 0.12]}),
    MeshSpec("box", {"center": [SHAFT[0], 0.5 * (SHAFT[1] + BED_Y), +0.82],
                     "half": [0.22, 0.5 * abs(SHAFT[1] - BED_Y) + 0.02, 0.12]}),
    # the gland the valve rod slides through
    MeshSpec("cylinder", {"radius": 0.16, "height": 0.30,
                          "center": [GLAND[0], GLAND[1], LAYER["valve_rod"]],
                          "axis": X, "sections": 16}),
    # the reversing lever's bracket and its notched quadrant
    MeshSpec("box", {"center": [LEVER_PIVOT[0], LEVER_PIVOT[1] - 0.30,
                                LAYER["reverse_lever"]],
                     "half": [0.15, 0.32, 0.10]}),
] + [
    _bar([LEVER_PIVOT[0] + QUADRANT_R * math.cos(a0),
          LEVER_PIVOT[1] + QUADRANT_R * math.sin(a0), LAYER["quadrant"]],
         [LEVER_PIVOT[0] + QUADRANT_R * math.cos(a1),
          LEVER_PIVOT[1] + QUADRANT_R * math.sin(a1), LAYER["quadrant"]], 0.12)
    for a0, a1 in [(math.radians(158.0 + 4.0 * i), math.radians(158.0 + 4.0 * (i + 1)))
                   for i in range(14)]
]


CASE = AtlasCaseSpec(
    name="stephenson_link",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef("__ground", MeshSpec("compound", {"parts": _frame_parts}),
                    fixed=True, color=IRON),
            BodyDef("shaft", MeshSpec("compound", {"parts": _shaft_parts}),
                    color=STEEL),
            # each rod carries its strap: a ring round the eccentric sheave in
            # the figure, here a disc standing beside it in its own layer
            BodyDef("rod_fore", MeshSpec("compound", {"parts": [
                        _bar(_p(ECC_FORE, "rod_fore"), _p(PIN_A, "rod_fore")),
                        _boss(ECC_FORE, 0.58, LAYER["rod_fore"] - 0.05,
                              LAYER["rod_fore"] + 0.05)]}), color=RED),
            BodyDef("rod_back", MeshSpec("compound", {"parts": [
                        _bar(_p(ECC_BACK, "rod_back"), _p(PIN_B, "rod_back")),
                        _boss(ECC_BACK, 0.58, LAYER["rod_back"] - 0.05,
                              LAYER["rod_back"] + 0.05)]}), color=BLUE),
            BodyDef("expansion_link", MeshSpec("compound", {"parts": _link_parts}),
                    color=(0.36, 0.55, 0.42)),
            BodyDef("die_block",
                    MeshSpec("compound", {"parts": [
                        _bar(_p(_on_arc(ANG_DIE - DIE_TANGENT / ROD_LEN), "die_block"),
                             _p(_on_arc(ANG_DIE + DIE_TANGENT / ROD_LEN), "die_block"),
                             2.0 * DIE_RADIAL),
                        _boss(DIE_PIN, 0.07, LAYER["rocker"] - 0.05, -DIE_RADIAL)]}),
                    color=BRONZE),
            BodyDef("rocker", MeshSpec("compound", {"parts": _rocker_parts}),
                    color=(0.93, 0.53, 0.18)),
            BodyDef("valve_link", _bar(_p(VALVE_PIN, "valve_link"),
                                       _p(ROCK_TOP, "valve_link"), 0.09),
                    color=STEEL),
            BodyDef("valve_rod", _bar([GLAND[0] - 0.45, GLAND[1], LAYER["valve_rod"]],
                                      _p(VALVE_PIN, "valve_rod"), 0.14),
                    color=STEEL),
            BodyDef("lifting_link", _bar(_p(LUG, "lifting_link"),
                                         _p(LIFT_TOP, "lifting_link")),
                    color=(0.72, 0.36, 0.72)),
            BodyDef("reverse_lever", MeshSpec("compound", {"parts": _lever_parts}),
                    color=(0.35, 0.72, 0.45)),
        ],
        joints=[
            # crankshaft and its two eccentrics
            JointDef("crank", "revolute", "__ground", "shaft",
                     pivot=[SHAFT[0], SHAFT[1], 0.0], axis=Z),
            JointDef("ecc_fore", "revolute", "shaft", "rod_fore",
                     pivot=_pivot(ECC_FORE, "ecc_fore", "rod_fore"), axis=Z),
            JointDef("ecc_back", "revolute", "shaft", "rod_back",
                     pivot=_pivot(ECC_BACK, "ecc_back", "rod_back"), axis=Z),
            # the two rods onto the ends of the expansion link
            JointDef("link_fore", "revolute", "rod_fore", "expansion_link",
                     pivot=_pivot(PIN_A, "rod_fore", "expansion_link"), axis=Z),
            JointDef("link_back", "revolute", "rod_back", "expansion_link",
                     pivot=_pivot(PIN_B, "rod_back", "expansion_link"), axis=Z),
            # the die in the slot: a revolute about the slot's own centre of
            # curvature, limited to the arc the slot actually spans
            JointDef("die_slot", "revolute", "expansion_link", "die_block",
                     pivot=[ARC_C[0], ARC_C[1], 0.0], axis=Z,
                     limit_lo=ANG_LO - ANG_DIE, limit_hi=ANG_HI - ANG_DIE),
            # the die's pin, the rocker, and the valve train
            JointDef("die_pin", "revolute", "rocker", "die_block",
                     pivot=_pivot(DIE_PIN, "rocker", "die_block"), axis=Z),
            JointDef("rocker_pin", "revolute", "__ground", "rocker",
                     pivot=_p((0.0, 0.0), "rocker"), axis=Z),
            JointDef("valve_upper", "revolute", "rocker", "valve_link",
                     pivot=_pivot(ROCK_TOP, "rocker", "valve_link"), axis=Z),
            JointDef("valve_lower", "revolute", "valve_link", "valve_rod",
                     pivot=_p(VALVE_PIN, "valve_rod"), axis=Z),
            JointDef("gland", "prismatic", "__ground", "valve_rod",
                     pivot=[GLAND[0], GLAND[1], LAYER["valve_rod"]], axis=X),
            # the reversing gear
            JointDef("reverser", "revolute", "__ground", "reverse_lever",
                     pivot=_p(LEVER_PIVOT, "reverse_lever"), axis=Z),
            JointDef("lift_upper", "revolute", "reverse_lever", "lifting_link",
                     pivot=_pivot(LIFT_TOP, "reverse_lever", "lifting_link"), axis=Z),
            JointDef("lift_lower", "revolute", "lifting_link", "expansion_link",
                     pivot=_p(LUG, "lifting_link"), axis=Z),
        ],
        up_axis="y",                     # planar linkage in xy: gravity is in-plane
        render_up_axis="z",
        gravity=9.8,
    ),
    seeds=[],
    contact_model="none",                # the slot is `die_slot`, not a mesh pair
    # Prior-free: the crank angle is an obvious coordinate and the reverser
    # position is not — the die's travel along the slot is what actually matters
    # and no body's angle reads it off.  Discovery gets poses and the oracle.
    coordinates=[
        CoordinateDef("crank", topology="periodic", period=2.0 * math.pi,
                      source="joint_angle",
                      args={"joint": "crank"}),
        CoordinateDef("reverser", source="joint_angle",
                      args={"joint": "reverser"}),
    ],
    actuators=[
        ActuatorDef("crank torque", kind="joint_torque", target="crank", hi=400.0),
        ActuatorDef("reverser torque", kind="joint_torque", target="reverser", hi=200.0),
    ],
    build=BuildOptions(
        # No contact, so no IPC relaxation to tune: `ideal_oracle` takes this
        # case analytically and the d_hat/kappa/newton knobs never apply.
        sampling=SamplingOptions(
            # The zero-prior advancing-front explorer (atlas_rom.sample.sampling)
            # covers the crank/reverser rectangle itself; `step` is intrinsic
            # spacing over the 2-DOF manifold, so raising it thins the front
            # to roughly the requested sample count instead of the grid's
            # former ~9.4k-candidate raster.
            # Continue until the physical frontier is exhausted (or the
            # explicit time budget expires); do not cut the reachable cover at
            # an arbitrary sample count.  This spacing targets about 1k samples
            # for Stephenson's 2-D domain while preserving the same
            # advancing-front method.
            n_samples=None, step=0.3, tangent_eps=0.01,
            rank_gap_ratio=50., max_seconds=5400.0,
            representation="planar_xytheta", graph_k=32,
            # Drive the two actuated members only; the link, die, rocker and
            # valve train have to be solved from the joints.
            actuated_bodies=["shaft", "reverse_lever"]),
        charts=ChartOptions(mode="coordinate_split", split_centres=[0.0, math.pi],
                            split_half_width=0.60 * math.pi),
        canonical=CanonicalOptions(enabled=False),
        decoder=DecoderOptions(epochs=1600, n_harmonics=3, smooth_lambda=1e-3,
                               closure_weight=3.0),
        dynamics=DynamicsOptions(distill=True, epochs=700, n_teacher=1200),
        # The slot ends bound the reverser at both extremes, so the feasible
        # region is a real boundary the runtime has to hold, not a formality.
        boundary=BoundaryOptions(kind="sdf", k=1e6, margin=0.0,
                                 damping=50.0, epochs=100,
                                 width=128, layers=4),
        expect=Expectations(dimension=2, n_bodies=10),
        tolerances=Tolerances(decoder_translation_rmse=0.06,
                              decoder_rotation_deg=3.0, closure=1.1e-1),
    ),
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "distilled",
                            "boundary_solver": "damped"},
               "distilled": {"dt": 1e-3, "dynamics_backend": "distilled",
                              "boundary_solver": "damped"},
               "analytic": {"dt": 1e-3, "dynamics_backend": "analytic",
                             "boundary_solver": "damped"}},
)
