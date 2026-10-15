"""Shared physical SDF fixtures for topology-only atlas cases.

The helpers define only the simulator's feasible surface and pose observation.
No topology, parametrization, chart count, or membership label crosses into the
fitter or packaged runtime.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, Expectations, MechanismSpec, MeshSpec, SamplingOptions


def observed_pose3(q):
    return np.column_stack((np.eye(3), np.asarray(q, float)))[None]


def state_from_observation3(pose):
    return np.asarray(pose, float).reshape(1, 3, 4)[0, :, 3]


def surface_case(name: str, seed: tuple[float, float, float]) -> AtlasCaseSpec:
    """One movable physical probe; the SDF system supplies feasible states."""
    return AtlasCaseSpec(
        name=name,
        mechanism=MechanismSpec(bodies=[
            BodyDef(name="probe", mesh=MeshSpec("sphere", {"radius": .14, "subdiv": 2}),
                    initial_transform=[[1, 0, 0, seed[0]], [0, 1, 0, seed[1]],
                                       [0, 0, 1, seed[2]], [0, 0, 0, 1]],
                    color=(.25, .55, .90)),
        ]),
        coordinates=[
            CoordinateDef("obs_x", source="body_translation", args={"body": "probe", "axis": "x"}),
            CoordinateDef("obs_y", source="body_translation", args={"body": "probe", "axis": "y"}),
            CoordinateDef("obs_z", source="body_translation", args={"body": "probe", "axis": "z"}),
        ],
        contact_model="none",
        build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.24),
                           expect=Expectations(n_bodies=1)),
    )


def planar_pose(q):
    q = np.asarray(q, float)
    return np.column_stack((np.eye(3), [q[0], q[1], 0.0]))[None]


def planar_state(pose):
    return np.asarray(pose, float).reshape(1, 3, 4)[0, :2, 3]


def planar_case(name: str, seed: tuple[float, float]) -> AtlasCaseSpec:
    return AtlasCaseSpec(
        name=name,
        mechanism=MechanismSpec(bodies=[
            BodyDef(name="probe", mesh=MeshSpec("sphere", {"radius": .14, "subdiv": 2}),
                    initial_transform=[[1, 0, 0, seed[0]], [0, 1, 0, seed[1]],
                                       [0, 0, 1, 0], [0, 0, 0, 1]],
                    color=(.25, .55, .90)),
        ]),
        coordinates=[
            CoordinateDef("obs_x", source="body_translation", args={"body": "probe", "axis": "x"}),
            CoordinateDef("obs_y", source="body_translation", args={"body": "probe", "axis": "y"}),
        ],
        contact_model="none",
        build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.18),
                           expect=Expectations(n_bodies=1)),
    )


def point_segment_distance(q, a, b):
    q = np.asarray(q, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    direction = b - a
    t = np.clip(np.dot(q - a, direction) / max(float(direction @ direction), 1e-20), 0.0, 1.0)
    return float(np.linalg.norm(q - (a + t * direction)))


def smooth_min_sdf(parts, *, blend: float) -> float:
    """Smooth union of signed-distance primitives (negative is inside)."""
    values = np.asarray(parts, float)
    low = float(values.min())
    return float(low - blend * np.log(np.exp(-(values - low) / blend).sum()))


def torus_sdf(q, center, *, major_radius: float, tube_radius: float) -> float:
    q = np.asarray(q, float) - np.asarray(center, float)
    return float(np.hypot(np.hypot(q[0], q[1]) - major_radius, q[2]) - tube_radius)


def rounded_graph_sdf(q, vertices, edges, *, tube_radius: float, blend: float) -> float:
    """Smooth regular neighbourhood of an embedded graph.

    The surface genus is the graph's first Betti number.  This construction is
    useful for the tetrahedral and capped-cube fixtures because every handle is
    a visible, ordinary tunnel rather than a connection assembled from charts.
    """
    vertices = np.asarray(vertices, float)
    parts = [point_segment_distance(q, vertices[left], vertices[right]) - tube_radius
             for left, right in edges]
    return smooth_min_sdf(parts, blend=blend)


def rounded_box_sdf(q, center, half_extent, *, radius: float) -> float:
    """Distance field of a box with every edge and corner rounded."""
    p = np.abs(np.asarray(q, float) - np.asarray(center, float)) - np.asarray(half_extent, float)
    outside = float(np.linalg.norm(np.maximum(p, 0.0)))
    inside = min(float(np.max(p)), 0.0)
    return outside + inside - radius
