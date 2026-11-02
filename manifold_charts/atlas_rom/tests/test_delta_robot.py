"""Analytic-constraint and evaluator-only tests for the delta robot case."""
from __future__ import annotations

import numpy as np

from atlas_rom.cases import load_spec
from atlas_rom.cases.delta_robot import EVALUATOR_REFERENCE
from atlas_rom.simulate import ideal_oracle


def _oracle():
    return ideal_oracle.build_ideal_oracle(load_spec("delta_robot"), log=lambda *_: None)


def _platform(oracle, q):
    """Evaluator readout: the platform's (rotation, translation)."""
    T = np.asarray(q).reshape(oracle.n_bodies, 3, 4)[oracle.order.index("platform")]
    return T[:, :3], T[:, 3]


def _walk(oracle, q, direction, steps, h=0.1):
    """March along a transported tangent direction, projecting every step."""
    v = None
    for _ in range(steps):
        frame, _, _ = oracle.tangent_frame(q)
        v = frame @ (frame.T @ (direction if v is None else v))
        v /= np.linalg.norm(v)
        q = oracle.project(q + h * v)
    return q


def test_delta_robot_is_prior_free_and_analytic():
    spec = load_spec("delta_robot")
    assert spec.coordinates == []
    assert spec.build.expect.dimension is None
    assert ideal_oracle.supports(spec)
    assert "EVALUATOR_REFERENCE" not in spec.to_dict()
    assert EVALUATOR_REFERENCE["intrinsic_dimension"] == 3


def test_rest_pose_is_assembled_and_three_dimensional():
    oracle = _oracle()
    q = oracle.pack()
    assert np.linalg.norm(oracle.residual(q), np.inf) < 1e-12
    _, _, dimension = oracle.tangent_frame(q)
    assert dimension == 3


def test_platform_translates_three_ways_without_rotating():
    oracle = _oracle()
    rest = oracle.pack()
    R0, p0 = _platform(oracle, rest)
    frame, _, _ = oracle.tangent_frame(rest)
    displacements = []
    for k in range(3):
        q = _walk(oracle, rest, frame[:, k], 5)
        assert np.linalg.norm(oracle.residual(q), np.inf) < 1e-9
        assert oracle.tangent_frame(q)[2] == 3
        R, p = _platform(oracle, q)
        assert np.linalg.norm(R - R0) < 1e-6          # parallelograms hold orientation
        displacements.append(p - p0)
    s = np.linalg.svd(np.array(displacements), compute_uv=False)
    assert s[-1] > 0.05                                # three independent finite translations
