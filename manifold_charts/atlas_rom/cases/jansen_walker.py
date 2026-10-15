"""Eight independently pinned Jansen legs locked to a shared, phased shaft."""
from __future__ import annotations

import math

import numpy as np

from atlas_rom.cases._jansen_array_common import make_case

STATION_DX = 0.90
PLANE_Z0 = 0.06
PLANE_DZ = 0.05
PLACEMENTS = []
for station in range(2):
    x = (station - 0.5) * STATION_DX
    for row in range(2):
        z = PLANE_Z0 + PLANE_DZ * (2 * station + row)
        phase = math.pi * station + 0.5 * math.pi * row
        PLACEMENTS.extend(((f"R{station}{row}", x, +z, phase, False),
                           (f"L{station}{row}", x, -z, phase + math.pi, True)))

CASE = make_case("jansen_walker", 8, PLACEMENTS)
CASE.build.sampling.checkpoint_seconds = 60.


def VIEW_EMBED(q):
    """Display the shared crank phase on its canonical unit circle.

    This is a viewer-only map: the phase is recovered from the recorded
    physical crank transform and is never supplied to sampling or CAE fitting.
    """
    poses = np.asarray(q, float).reshape(len(q), -1, 3, 4)
    crank_index = CASE.mechanism.body_order.index("crank@R00")
    rotation = poses[:, crank_index, :, :3]
    phi = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return np.column_stack((np.cos(phi), np.sin(phi)))
