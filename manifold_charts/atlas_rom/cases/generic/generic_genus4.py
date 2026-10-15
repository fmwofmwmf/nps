"""Closed genus-4 surface: rounded cube frame with one square face filled."""
from __future__ import annotations

from manifold_charts.atlas_rom.cases.generic._generic_surface_common import (observed_pose3, rounded_box_sdf,
                                                     rounded_graph_sdf, smooth_min_sdf,
                                                     state_from_observation3, surface_case)
from atlas_rom.simulate.sdf_oracle import SDFSystem

_SIDE, _TUBE = 1.20, .27
_SEED_Z = -1.475914650829763
_VERTICES = tuple((x, y, z) for z in (-_SIDE, _SIDE)
                  for y in (-_SIDE, _SIDE) for x in (-_SIDE, _SIDE))
_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
          (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def _genus4_sdf(q):
    frame = rounded_graph_sdf(q, _VERTICES, _EDGES, tube_radius=_TUBE, blend=.07)
    # This rounded plate caps the y=-SIDE square.  Attaching one 2-cell to the
    # cube graph removes its dependent face cycle, leaving genus four.
    cap = rounded_box_sdf(q, (0.0, -_SIDE, 0.0), (1.0, .08, 1.0), radius=.30)
    return smooth_min_sdf((frame, cap), blend=.07)


SDF_SYSTEM = SDFSystem(_genus4_sdf, seeds=((0.0, -_SIDE, _SEED_Z),),
                       dimension=3, observe=observed_pose3,
                       state_from_observation=state_from_observation3, surface=True,
                       fingerprint="generic-genus4-square-surface-sdf-v1", step=.22, tangent_eps=.01,
                       max_samples=None, max_seconds=None)
CASE = surface_case("generic_genus4", (0.0, -_SIDE, _SEED_Z))
