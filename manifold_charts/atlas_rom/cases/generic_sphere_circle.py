"""SDF zero-level S^2 x S^1 in R^4 with physical pose observations.

The middle rung of the intrinsic-dimension-3 topology ladder: S^3 has
Lusternik--Schnirelmann category 2, this has 3, and T^3 has 4, all at the same
dimension and ambient size.  Together they test whether chart count tracks
topology with everything else held fixed.

The surface is the tube of radius ``_TUBE`` around the 2-sphere of radius
``_RADIUS`` sitting in the w = 0 hyperplane.  That sphere has codimension 2 in
R^4 and a trivial normal bundle, so the tube is S^2 x S^1 rather than a twisted
bundle, and ``hypot(|xyz| - R, w)`` is its exact distance function.

Observation packing follows ``generic_sphere3``: ``probe`` carries (q0, q1, q2)
and ``probe_w`` carries q3 in its x slot.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, Expectations, MechanismSpec, MeshSpec, SamplingOptions
from atlas_rom.simulate.sdf_oracle import SDFSystem

_RADIUS = 2.0
_TUBE = 1.2
_SEED_X = _RADIUS + _TUBE

# Evaluation ground truth.  Deliberately not part of CASE, SDF_SYSTEM, or any
# artifact: the fitter must derive both numbers from measured evidence.
GROUND_TRUTH = {"topology": "S^2 x S^1", "intrinsic_dimension": 3,
                "min_charts_ls_category": 3}


def _sphere_circle_sdf(q):
    q = np.asarray(q, float)
    return float(np.hypot(np.linalg.norm(q[:3]) - _RADIUS, q[3]) - _TUBE)


def _observed_pose(q):
    """Physical pose emitted by the surface-response simulator."""
    q = np.asarray(q, float)
    return np.stack((np.column_stack((np.eye(3), q[:3])),
                     np.column_stack((np.eye(3), np.array([q[3], 0.0, 0.0])))))


def _state_from_observation(pose):
    T = np.asarray(pose, float).reshape(2, 3, 4)
    return np.array([T[0, 0, 3], T[0, 1, 3], T[0, 2, 3], T[1, 0, 3]])


SDF_SYSTEM = SDFSystem(_sphere_circle_sdf, seeds=((_SEED_X, 0.0, 0.0, 0.0),), dimension=4,
                       observe=_observed_pose, surface=True,
                       state_from_observation=_state_from_observation,
                       fingerprint="generic-sphere-circle-surface-sdf-v1", step=.55,
                       max_samples=2000, max_seconds=1800.)


CASE = AtlasCaseSpec(
    name="generic_sphere_circle",
    mechanism=MechanismSpec(bodies=[
        BodyDef(name="probe", mesh=MeshSpec("sphere", {"radius": .14, "subdiv": 2}),
                initial_transform=[[1, 0, 0, _SEED_X], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]],
                color=(.25, .55, .90)),
        BodyDef(name="probe_w", mesh=MeshSpec("sphere", {"radius": .14, "subdiv": 2}),
                initial_transform=[[1, 0, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]],
                color=(.90, .55, .25)),
    ]),
    coordinates=[CoordinateDef("obs_x", source="body_translation", args={"body": "probe", "axis": "x"}),
                 CoordinateDef("obs_y", source="body_translation", args={"body": "probe", "axis": "y"}),
                 CoordinateDef("obs_z", source="body_translation", args={"body": "probe", "axis": "z"}),
                 CoordinateDef("obs_w", source="body_translation", args={"body": "probe_w", "axis": "x"})],
    contact_model="none",
    build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.26),
                       expect=Expectations(n_bodies=2)),
)
