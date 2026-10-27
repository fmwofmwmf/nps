"""product_split: Riemannian-product detection on fixtures with known answers.

Stages 1-2 run on analytic point clouds with exact tangent frames, each frame
rotated by a random gauge inside its tangent space.  Stage 3 runs on small
closed-form `PhysicalSystem`s.  The known answers live only here."""
from __future__ import annotations

import numpy as np
import pytest

from atlas_rom.sample.product_split import certify_split, detect_product_split, infer_frame
from atlas_rom.simulate.oracle_base import PhysicalObservation, PhysicalResponse


# ── stages 1-2: point clouds ─────────────────────────────────────────────────
def _cloud(embed, params, seed=0, eps=1e-6):
    rng = np.random.default_rng(seed)
    q = np.stack([embed(p) for p in params])
    frames = []
    for p in params:
        J = np.column_stack([(embed(p + eps * e) - embed(p - eps * e)) / (2 * eps)
                             for e in np.eye(len(p))])
        gauge = np.linalg.qr(rng.standard_normal((len(p), len(p))))[0]
        frames.append(np.linalg.qr(J)[0] @ gauge)
    return q, np.stack(frames)


def _grid(*axes):
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(axes))


def _fib_sphere(n):
    k = np.arange(n) + 0.5
    return np.column_stack((np.arccos(1 - 2 * k / n), (np.pi * (1 + 5 ** 0.5) * k) % (2 * np.pi)))


def _sphere(a):
    return np.array([np.sin(a[0]) * np.cos(a[1]), np.sin(a[0]) * np.sin(a[1]), np.cos(a[0])])


_LOOP = np.linspace(0, 2 * np.pi, 60, endpoint=False)
_S2 = _fib_sphere(200)


def _hopf_cloud(n=2000, seed=0):
    h = np.random.default_rng(seed).uniform(size=(n, 3)) * np.array([1, 2 * np.pi, 2 * np.pi])
    h[:, 0] = np.arcsin(np.sqrt(h[:, 0]))
    return h


FIXTURES = {
    # name: (embedding, parameters, expected split kind, expected factor dims)
    "cylinder": (lambda p: np.array([np.cos(p[0]), np.sin(p[0]), p[1]]),
                 _grid(_LOOP, np.linspace(0, 1.5, 15)), "flat_boundary", [1, 1]),
    "torus_x_interval": (
        lambda p: np.array([np.cos(p[0]), np.sin(p[0]), .7 * np.cos(p[1]), .7 * np.sin(p[1]), p[2]]),
        _grid(np.linspace(0, 2 * np.pi, 24, endpoint=False),
              np.linspace(0, 2 * np.pi, 17, endpoint=False), np.linspace(0, 1.2, 5)),
        "flat_boundary", [2, 1]),
    "sphere_x_circle": (
        lambda p: np.concatenate((_sphere(p[:2]), .4 * np.array([np.cos(p[2]), np.sin(p[2])]))),
        np.array([[a, b, c] for a, b in _S2 for c in np.linspace(0, 2 * np.pi, 10, endpoint=False)]),
        "holonomy", [1, 2]),
    # a topological product whose metric is WARPED: not a Riemannian product
    "warped_tube": (
        lambda p: np.concatenate(((1 + .6 * np.cos(p[2])) * _sphere(p[:2]), [.6 * np.sin(p[2])])),
        np.array([[a, b, c] for a, b in _S2 for c in np.linspace(0, 2 * np.pi, 14, endpoint=False)]),
        "irreducible", [3]),
    "revolution": (
        lambda p: np.array([(1 + .3 * p[1] ** 2) * np.cos(p[0]), (1 + .3 * p[1] ** 2) * np.sin(p[0]), p[1]]),
        _grid(_LOOP, np.linspace(0, 1.5, 15)), "irreducible", [2]),
    "three_sphere": (
        lambda p: np.array([np.cos(p[0]) * np.cos(p[1]), np.cos(p[0]) * np.sin(p[1]),
                            np.sin(p[0]) * np.cos(p[2]), np.sin(p[0]) * np.sin(p[2])]),
        _hopf_cloud(), "irreducible", [3]),
}


@pytest.mark.parametrize("name", FIXTURES)
def test_split_kind_and_factors(name):
    embed, params, kind, factors = FIXTURES[name]
    report = detect_product_split(*_cloud(embed, params))
    assert report["split"]["kind"] == kind, report
    assert sorted(report["split"]["factors"]) == sorted(factors), report


def test_parallel_field_counts():
    """One near-zero connection eigenvalue per parallel field, behind a gap."""
    expected = {"cylinder": 2, "torus_x_interval": 3, "sphere_x_circle": 1,
                "warped_tube": 0, "revolution": 0, "three_sphere": 0}
    for name, count in expected.items():
        embed, params, *_ = FIXTURES[name]
        assert detect_product_split(*_cloud(embed, params))["parallel"]["count"] == count, name


def test_boundary_axis_is_the_interval():
    """The flat cylinder's split puts the boundary normal along its axis."""
    embed, params, *_ = FIXTURES["cylinder"]
    q, frames = _cloud(embed, params)
    report = detect_product_split(q, frames)
    interval = frames[0] @ report["_bases"][1][:, 0]        # root frame -> ambient
    assert abs(abs(interval[2]) - 1.0) < 1e-6


def test_split_does_not_depend_on_the_frame_gauge():
    embed, params, *_ = FIXTURES["sphere_x_circle"]
    reports = [detect_product_split(*_cloud(embed, params, seed=s)) for s in (1, 2)]
    assert [r["split"] for r in reports] == [reports[0]["split"]] * 2


# ── stage 3: closed-form physical systems ────────────────────────────────────
class _Surface:
    """A surface of revolution x = (r(z) cos u, r(z) sin u, z), z in [0, top(u)],
    as a response-only system: a command moves the point and retracts it."""

    def __init__(self, radius, top):
        self.radius, self.top = radius, top

    def _retract(self, p):
        u, z = np.arctan2(p[1], p[0]), p[2]
        return np.array([self.radius(z) * np.cos(u), self.radius(z) * np.sin(u), z]), u, z

    def seed(self):
        return PhysicalObservation(self._retract(np.array([1.0, 0.0, 0.6]))[0])

    def restore(self, observation):
        return np.asarray(observation.pose, float)

    def command(self, session, direction, scale, *, parent=None):
        pose, u, z = self._retract(session + scale * np.asarray(direction, float))
        valid = 0.0 <= z <= self.top(u)
        return PhysicalResponse(PhysicalObservation(pose), True, bool(valid), {}, parent)

    def command_many(self, session, directions, scale, *, parent=None):
        return [self.command(session, d, scale, parent=parent) for d in directions]


SYSTEMS = {
    # name: (system, commuting gap must vanish, width must be constant)
    "cylinder": (_Surface(lambda z: 1.0, lambda u: 1.2), True, True),
    # curved: meridians are all the same length, so only commuting catches it
    "revolution": (_Surface(lambda z: 1.0 + .3 * z ** 2, lambda u: 1.2), False, True),
    # flat: every split commutes, so only the width catches it
    "slanted_top": (_Surface(lambda z: 1.0, lambda u: 1.0 + .2 * np.cos(u)), True, False),
}


def _split_at(system, observation, frame):
    """Evaluator readout: the meridian (z) direction is the candidate interval."""
    b = frame.T @ np.array([0.0, 0.0, 1.0])
    b /= np.linalg.norm(b)
    return np.array([[-b[1]], [b[0]]]), b[:, None]


@pytest.mark.parametrize("name", SYSTEMS)
def test_physical_certificate(name):
    system, commutes, constant_width = SYSTEMS[name]
    rng = np.random.default_rng(0)
    obs = system.seed()
    frame = infer_frame(system, obs, 2, 1e-3, rng=rng)
    A, B = _split_at(system, obs, frame)
    width_bases = []
    for u in (0.0, 1.5, 3.0):
        o = PhysicalObservation(system._retract(np.array([np.cos(u), np.sin(u), 0.6]))[0])
        f = infer_frame(system, o, 2, 1e-3, rng=rng)
        width_bases.append((o, f, _split_at(system, o, f)[1][:, 0]))
    report = certify_split(system, obs, frame, [A, B], length=0.4, substeps=(4, 8, 16),
                           width_bases=width_bases, width_substep=0.05, rng=rng)
    ratio = report["gap_refinement_ratio"]
    assert (ratio < 0.6) == commutes, report
    assert (report["width_cv"] < 1e-3) == constant_width, report
