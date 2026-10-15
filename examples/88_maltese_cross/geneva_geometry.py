"""Procedural mesh generation for the Maltese-cross (Geneva drive) mechanism.

Classic external Geneva drive with n=4 slots (the "Maltese cross"). Two rigid
bodies mesh purely through contact, no joints:

  driver: a disc (fixed-axis-driven by RotatingMotor) carrying one crank pin
          at radius r = a*sin(pi/n) from its own center.
  wheel:  a slotted disc, outer radius R = a*cos(pi/n) from its own center
          (== r for n=4), free-floating except for the contact bearing on
          its fixed axle.

`a` is the center distance between the two axles. The two discs live in
separate, non-overlapping z-layers so their flat faces never touch; only the
crank pin (which spans both layers) can touch the wheel, through whichever
slot it currently occupies. This is the standard non-jamming Geneva geometry:
at the pin's entry/exit angle its path is tangent to the wheel radius, so
each engagement carries the wheel through exactly 360/n degrees with zero
angular velocity at both ends (the characteristic Geneva indexing motion).
The locking disc that classic Geneva wheels use to hold position between
engagements is omitted -- with zero gravity and no other torque on the free
wheel, it simply has nothing to make it drift while disengaged. (A first cut
at that locking ring made things worse: a driver rotated to a mid-idle start
pose immediately interpenetrated the wheel, because the ring's solid arc and
the wheel's matching concave scallops are only built to fit at the four
90-degree-separated rest alignments, not at an arbitrary start angle. Getting
that right needs the two bodies' start poses solved jointly, which wasn't
worth it for a demo -- omitted rather than shipped broken.)
"""

import numpy as np
import trimesh as tm

N_SLOTS = 4
A = 1.0  # center distance between driver axle and wheel axle
R_CRANK = A * np.sin(np.pi / N_SLOTS)   # driver crank-pin orbit radius
R_WHEEL = A * np.cos(np.pi / N_SLOTS)   # wheel outer radius (== R_CRANK at n=4)

T_DISC = 0.08     # driver disc thickness
GAP = 0.02        # axial clearance between driver and wheel layers
T_WHEEL = 0.08    # wheel disc thickness

DRIVER_BORE_R = 0.15
WHEEL_BORE_R = 0.12
AXLE_CLEAR = 0.015   # radial clearance between a bore and its fixed axle

CRANK_PIN_R = 0.045
DRIVER_DISC_R = R_CRANK + CRANK_PIN_R + 0.05

SLOT_W = 2 * CRANK_PIN_R + 0.12   # slot width: pin diameter + 0.06 clearance/side
# radius (from wheel center) the slot must reach: the pin CENTER's closest approach
# is (A - R_CRANK), so the slot must clear that by a full pin radius plus margin, or
# the pin's near edge pokes into material the slot never cut away.
SLOT_INNER = (A - R_CRANK) - CRANK_PIN_R - 0.03

DRIVER_Z = (0.0, T_DISC)
WHEEL_Z = (T_DISC + GAP, T_DISC + GAP + T_WHEEL)


def _cyl(radius, z0, z1, center_xy=(0.0, 0.0), sections=128):
    h = z1 - z0
    c = tm.creation.cylinder(radius=radius, height=h, sections=sections)
    c.apply_translation([center_xy[0], center_xy[1], z0 + h / 2.0])
    return c


def _as_VF(mesh: tm.Trimesh):
    return (np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.int32))


def build_driver():
    """Disc + one radial crank pin, bored for its own fixed axle. Pivot at (0,0)."""
    disc = _cyl(DRIVER_DISC_R, DRIVER_Z[0], DRIVER_Z[1])
    bore = _cyl(DRIVER_BORE_R, DRIVER_Z[0] - 0.02, DRIVER_Z[1] + 0.02)
    disc = disc.difference(bore, engine=None)
    pin = _cyl(CRANK_PIN_R, DRIVER_Z[0], WHEEL_Z[1], center_xy=(R_CRANK, 0.0))
    driver = disc.union(pin, engine=None)
    return _as_VF(driver)


def build_wheel():
    """Slotted disc, bored for its own fixed axle. Pivot at (A,0)."""
    z0, z1 = WHEEL_Z
    disc = _cyl(R_WHEEL, z0, z1, center_xy=(A, 0.0))
    bore = _cyl(WHEEL_BORE_R, z0 - 0.02, z1 + 0.02, center_xy=(A, 0.0))
    wheel = disc.difference(bore, engine=None)
    for k in range(N_SLOTS):
        # k=0 slot points toward the driver (world -x) at rest pose, matching the
        # crank pin's deepest-penetration angle (driver angle 0, pin on the O1-O2 line)
        ang = np.pi + 2 * np.pi * k / N_SLOTS
        slot = tm.creation.box(extents=[R_WHEEL - SLOT_INNER + 0.1, SLOT_W, T_WHEEL + 0.04])
        slot.apply_translation([(R_WHEEL + SLOT_INNER) / 2.0, 0.0, 0.0])
        c, s = np.cos(ang), np.sin(ang)
        Rz = np.eye(4)
        Rz[:2, :2] = [[c, -s], [s, c]]
        slot.apply_transform(Rz)
        slot.apply_translation([A, 0.0, (z0 + z1) / 2.0])
        wheel = wheel.difference(slot, engine=None)
    return _as_VF(wheel)


def build_driver_axle():
    """Fixed axle cylinder the driver disc's bore rides on."""
    z0, z1 = DRIVER_Z
    m = _cyl(DRIVER_BORE_R - AXLE_CLEAR, z0 - 0.03, z1 + 0.03, center_xy=(0.0, 0.0))
    return _as_VF(m)


def build_wheel_axle():
    """Fixed axle cylinder the wheel's bore rides on."""
    z0, z1 = WHEEL_Z
    m = _cyl(WHEEL_BORE_R - AXLE_CLEAR, z0 - 0.03, z1 + 0.03, center_xy=(A, 0.0))
    return _as_VF(m)
