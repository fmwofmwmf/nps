"""Closed genus-3 surface: a smoothly rounded tetrahedral wireframe."""
from __future__ import annotations

from manifold_charts.atlas_rom.cases.generic._generic_surface_common import (observed_pose3, rounded_graph_sdf,
                                                     state_from_observation3, surface_case)
from atlas_rom.simulate.sdf_oracle import SDFSystem

_VERTICES = ((0.0, 0.0, 1.35), (1.2, -.8, -.60),
             (-1.2, -.8, -.60), (0.0, 1.35, -.60))
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_TUBE = .32
_SEED_Z = -.2799332717273785


def _genus3_sdf(q):
    return rounded_graph_sdf(q, _VERTICES, _EDGES, tube_radius=_TUBE, blend=.075)


SDF_SYSTEM = SDFSystem(_genus3_sdf, seeds=((0.0, -.8, _SEED_Z),),
                       dimension=3, observe=observed_pose3,
                       state_from_observation=state_from_observation3, surface=True,
                       fingerprint="generic-genus3-triangle-surface-sdf-v1", step=.158, tangent_eps=.01,
                       max_samples=None, max_seconds=None)
CASE = surface_case("generic_genus3", (0.0, -.8, _SEED_Z))
