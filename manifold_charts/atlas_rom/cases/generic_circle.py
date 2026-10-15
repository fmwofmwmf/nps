"""SDF zero-level circle S^1 in R^2 with physical pose observations.

The smallest closed manifold in the generic family: intrinsic dimension 1,
Lusternik--Schnirelmann category 2, so a faithful atlas needs two charts and
no fewer.  Cheapest possible check that chart count tracks topology.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, Expectations, MechanismSpec, MeshSpec, SamplingOptions
from atlas_rom.simulate.sdf_oracle import SDFSystem

_RADIUS = 2.0

# Evaluation ground truth.  Deliberately not part of CASE, SDF_SYSTEM, or any
# artifact: the fitter must derive both numbers from measured evidence.
GROUND_TRUTH = {"topology": "S^1", "intrinsic_dimension": 1, "min_charts_ls_category": 2}


def _circle_sdf(q):
    return float(np.linalg.norm(q) - _RADIUS)


def _observed_pose(q):
    """Physical pose emitted by the surface-response simulator."""
    q = np.asarray(q, float)
    return np.column_stack((np.eye(3), np.array([q[0], q[1], 0.0])))[None]


def _state_from_observation(pose):
    return np.asarray(pose, float).reshape(1, 3, 4)[0, :2, 3]


SDF_SYSTEM = SDFSystem(_circle_sdf, seeds=((_RADIUS, 0.0),), dimension=2,
                       observe=_observed_pose, surface=True,
                       state_from_observation=_state_from_observation,
                       fingerprint="generic-circle-surface-sdf-v1", step=.25,
                       max_samples=300, max_seconds=300.)


CASE = AtlasCaseSpec(
    name="generic_circle",
    mechanism=MechanismSpec(bodies=[
        BodyDef(name="probe", mesh=MeshSpec("sphere", {"radius": .12, "subdiv": 2}),
                initial_transform=[[1, 0, 0, _RADIUS], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]],
                color=(.25, .55, .90)),
    ]),
    # System-observation declarations only; SDF sampling writes no ``u`` and
    # the chart builder derives its rank from measured pose secants.
    coordinates=[CoordinateDef("obs_x", source="body_translation", args={"body": "probe", "axis": "x"}),
                 CoordinateDef("obs_y", source="body_translation", args={"body": "probe", "axis": "y"})],
    contact_model="none",
    build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.15),
                       expect=Expectations(n_bodies=1)),
)
