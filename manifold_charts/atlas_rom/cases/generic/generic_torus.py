"""SDF zero-level torus T^2 in R^3 with physical pose observations.

Same intrinsic dimension as ``generic_sphere`` but genus 1: Lusternik--
Schnirelmann category 3, so a faithful atlas needs three charts where the
sphere needs two.  This is the case that separates "found the right number of
charts" from "cut the manifold somewhere convenient" -- a near-global chart
has to fold in two independent directions to cover it.

The SDF is the exact distance to the core circle's tube, built by wrapping a
tube of radius ``_TUBE`` around the circle of radius ``_RING`` in the xy-plane.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, Expectations, MechanismSpec, MeshSpec, SamplingOptions
from atlas_rom.simulate.sdf_oracle import SDFSystem

_RING = 2.0
_TUBE = 0.8
_SEED_X = _RING + _TUBE

# Evaluation ground truth.  Deliberately not part of CASE, SDF_SYSTEM, or any
# artifact: the fitter must derive both numbers from measured evidence.
GROUND_TRUTH = {"topology": "T^2 (genus 1)", "intrinsic_dimension": 2,
                "min_charts_ls_category": 3}


def _torus_sdf(q):
    q = np.asarray(q, float)
    return float(np.hypot(np.hypot(q[0], q[1]) - _RING, q[2]) - _TUBE)


def _observed_pose(q):
    """Physical pose emitted by the surface-response simulator."""
    return np.column_stack((np.eye(3), np.asarray(q, float)))[None]


def _state_from_observation(pose):
    return np.asarray(pose, float).reshape(1, 3, 4)[0, :, 3]


SDF_SYSTEM = SDFSystem(_torus_sdf, seeds=((_SEED_X, 0.0, 0.0),), dimension=3,
                       observe=_observed_pose, surface=True,
                       state_from_observation=_state_from_observation,
                       fingerprint="generic-torus-surface-sdf-v1", step=.28,
                       max_samples=1000, max_seconds=900.)

# Read exclusively by ``generic_manifold_diagnostics`` after training.  This
# display parametrization is deliberately not part of CASE or any artifact.
VISUAL_UVS = {"kind": "torus_ring_tube", "body": "probe",
              "ring_radius": _RING, "tube_radius": _TUBE}


CASE = AtlasCaseSpec(
    name="generic_torus",
    mechanism=MechanismSpec(bodies=[
        BodyDef(name="probe", mesh=MeshSpec("sphere", {"radius": .14, "subdiv": 2}),
                initial_transform=[[1, 0, 0, _SEED_X], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]],
                color=(.25, .55, .90)),
    ]),
    coordinates=[CoordinateDef("obs_x", source="body_translation", args={"body": "probe", "axis": "x"}),
                 CoordinateDef("obs_y", source="body_translation", args={"body": "probe", "axis": "y"}),
                 CoordinateDef("obs_z", source="body_translation", args={"body": "probe", "axis": "z"})],
    contact_model="none",
    build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.20),
                       expect=Expectations(n_bodies=1)),
)
