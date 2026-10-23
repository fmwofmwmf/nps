"""Analytic-constraint and evaluator-only tests for the drivetrain cases (inline-4 and V8)."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from atlas_rom.cases import load_spec
from atlas_rom.cases.drivetrain import EVALUATOR_REFERENCE
from atlas_rom.simulate import ideal_oracle
from atlas_rom.spec import geometry


CASES = ("drivetrain", "drivetrain_v8")
case = pytest.mark.parametrize("name", CASES)


def _oracle(name):
    return ideal_oracle.build_ideal_oracle(load_spec(name), log=lambda *_: None)


def _angle_x(oracle, q, name):
    """Evaluator readout: a body's spin about its own bearing axis -- world x for
    the carrier and axles, the `crank_main` axis for the crankshaft."""
    axis = (np.asarray(oracle.spec.mechanism.joint("crank_main").axis, float)
            if name == "crankshaft" else np.array([1.0, 0.0, 0.0]))
    u = np.cross(axis, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    R = np.asarray(q).reshape(oracle.n_bodies, 3, 4)[oracle.order.index(name), :, :3]
    v = R @ u
    return float(np.arctan2(np.cross(u, v) @ axis, u @ v))


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _walk(oracle, q, direction, steps, h=0.15):
    """March along a transported tangent direction, projecting every step."""
    v = None
    for _ in range(steps):
        frame, _, _ = oracle.tangent_frame(q)
        v = frame @ (frame.T @ (direction if v is None else v))
        v /= np.linalg.norm(v)
        q = oracle.project(q + h * v)
    return q


@case
def test_drivetrain_is_prior_free_and_analytic(name):
    spec = load_spec(name)
    assert spec.coordinates == []
    assert spec.build.expect.dimension is None
    assert ideal_oracle.supports(spec)
    assert "EVALUATOR_REFERENCE" not in spec.to_dict()
    assert EVALUATOR_REFERENCE["intrinsic_dimension"] == 2


@case
def test_rest_pose_is_assembled_and_two_dimensional(name):
    oracle = _oracle(name)
    q = oracle.pack()
    assert np.linalg.norm(oracle.residual(q), np.inf) < 1e-12
    _, _, dimension = oracle.tangent_frame(q)
    assert dimension == 2


@case
def test_both_freedoms_travel_finitely_and_obey_the_gear_laws(name):
    oracle = _oracle(name)
    rest = oracle.pack()
    frame, _, _ = oracle.tangent_frame(rest)
    for k in range(2):
        q = _walk(oracle, rest, frame[:, k], 60)
        assert np.linalg.norm(oracle.residual(q), np.inf) < 1e-9
        assert oracle.tangent_frame(q)[2] == 2
        crank, carrier = _angle_x(oracle, q, "crankshaft"), _angle_x(oracle, q, "carrier")
        left, right = _angle_x(oracle, q, "axle_left"), _angle_x(oracle, q, "axle_right")
        assert abs(_wrap(crank + 2.0 * carrier)) < 1e-6           # 2:1 final drive (spur or bevel)
        assert abs(_wrap(left + right - 2.0 * carrier)) < 1e-6    # open differential
        # finite travel, not a first-order flex: some body turned a large angle
        assert max(abs(crank), abs(left), abs(right)) > 1.0


@case
def test_split_freedom_moves_the_wheels_against_each_other(name):
    """Holding the engine, one wheel forward means the other backward."""
    oracle = _oracle(name)
    rest = oracle.pack()
    frame, _, _ = oracle.tangent_frame(rest)
    readout = lambda q: np.array([_angle_x(oracle, q, "carrier"),
                                  _angle_x(oracle, q, "axle_left")])
    h = 1e-6
    A = np.stack([(readout(rest + h * frame[:, j]) - readout(rest - h * frame[:, j])) / (2 * h)
                  for j in range(2)], 1)
    split = frame @ np.linalg.solve(A, np.array([0.0, 1.0]))      # carrier fixed, left +1
    q = _walk(oracle, rest, split, 25)
    assert abs(_angle_x(oracle, q, "carrier")) < 1e-3
    assert abs(_angle_x(oracle, q, "crankshaft")) < 1e-3
    assert abs(_wrap(_angle_x(oracle, q, "axle_left") + _angle_x(oracle, q, "axle_right"))) < 1e-6
    assert abs(_angle_x(oracle, q, "axle_left")) > 1.0


@case
def test_gear_constraints_have_consistent_jacobians(name):
    oracle = _oracle(name)
    rest = oracle.pack()
    q = _walk(oracle, rest, oracle.tangent_frame(rest)[0].sum(1), 7)
    analytic = oracle.jacobian(q)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for column in range(q.size):
        delta = np.zeros_like(q)
        delta[column] = step
        numeric[:, column] = (oracle.residual(q + delta) - oracle.residual(q - delta)) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


@case
def test_functional_meshes_are_closed_positive_solids(name):
    """Inertia comes from these meshes, so none may be inside out or open."""
    spec = load_spec(name)
    for body in spec.mechanism.driven_bodies:
        parts = (body.mesh.params["parts"] if body.mesh.kind == "compound" else [body.mesh])
        for part in parts:
            V, F = geometry.build_mesh(part)
            mesh = trimesh.Trimesh(V, F, process=False)
            mesh.merge_vertices()
            assert mesh.volume > 0.0, (body.name, part.kind)
