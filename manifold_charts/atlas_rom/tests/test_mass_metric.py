"""Mass metric from case meshes, its whitened system, and aimed frame measurement."""
from __future__ import annotations

import numpy as np
import pytest

from atlas_rom.cases import load_spec
from atlas_rom.sample.mass_metric import MassWhitenedSystem, body_moments, mass_metric
from atlas_rom.sample.physical_advancing_front import AdvancingFrontOptions, explore
from atlas_rom.simulate.oracle_base import PhysicalObservation, PhysicalResponse


@pytest.fixture(scope="module")
def spec():
    return load_spec("circle_arm_contact")


def _skew(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])


def test_translation_length_is_the_displacement(spec):
    metric, _masses = mass_metric(spec)
    shift = np.array([.3, -.2, .1])
    dq = np.zeros(12 * len(spec.mechanism.body_order))
    for i in range(len(spec.mechanism.body_order)):
        dq[12 * i:12 * (i + 1)] = np.hstack([np.zeros((3, 3)), shift[:, None]]).ravel()
    assert np.isclose(np.linalg.norm(metric.weigh(dq)), np.linalg.norm(shift))


def test_spin_about_the_centroid_is_rotational_energy(spec):
    import trimesh
    from atlas_rom.spec import geometry

    metric, masses = mass_metric(spec)
    i, name = 3, spec.mechanism.body_order[3]
    V, F = geometry.build_mesh(spec.mechanism.body(name).mesh)
    mesh = trimesh.Trimesh(V, F, process=False)
    mesh.density = spec.mechanism.body(name).density
    w = np.array([.2, -.1, .4])
    dq = np.zeros(12 * len(masses))
    dq[12 * i:12 * (i + 1)] = np.hstack([_skew(w), (-_skew(w) @ mesh.center_mass)[:, None]]).ravel()
    expected = w @ mesh.moment_inertia @ w / masses.sum()
    assert np.isclose(np.sum(metric.weigh(dq) ** 2), expected, rtol=1e-9)


def test_moments_reject_open_meshes(spec, monkeypatch):
    from atlas_rom.spec import geometry
    V, F = geometry.build_mesh(spec.mechanism.body("arm").mesh)
    monkeypatch.setattr(geometry, "build_mesh", lambda _mesh: (V, F[:-10]))
    with pytest.raises(ValueError, match="closed volume"):
        body_moments(spec)


class _Recorder:
    """Inner system that records packed commands and answers with the target."""

    def __init__(self, ambient):
        self.ambient = ambient
        self.calls = []

    def seed(self):
        return PhysicalObservation(np.zeros(self.ambient))

    def restore(self, observation):
        return np.asarray(observation.pose, float)

    def command(self, session, direction, scale, *, parent=None):
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        self.calls.append(scale * d)
        return PhysicalResponse(PhysicalObservation(session + scale * d), True, True, {}, parent)


def test_whitened_command_moves_y_by_the_scale(spec):
    metric, masses = mass_metric(spec)
    inner = _Recorder(12 * len(masses))
    system = MassWhitenedSystem(inner, metric, masses)
    direction = np.random.default_rng(0).standard_normal(inner.ambient)
    session = system.restore(system.seed())
    response = system.command(session, direction, 2.5)
    unit = direction / np.linalg.norm(direction)
    assert np.allclose(response.observation.pose, 2.5 * unit)
    assert np.allclose(inner.calls[0], system.to_packed(2.5 * unit))


def test_generic_directions_push_every_body_equally(spec):
    metric, masses = mass_metric(spec)
    system = MassWhitenedSystem(_Recorder(12 * len(masses)), metric, masses)
    directions = system.generic_directions(np.random.default_rng(1), 5)
    assert np.allclose(np.linalg.norm(directions, axis=0), 1.0)
    for column in directions.T:
        rms = [np.linalg.norm(column[12 * i:12 * (i + 1)]) / np.sqrt(m / masses.sum())
               for i, m in enumerate(masses)]
        assert np.allclose(rms, rms[0])


class _Torus:
    """A flat torus in R^4 seen through a projection-like response."""

    def __init__(self):
        self._state = {}

    def _obs(self, x):
        pose = np.array([np.cos(x[0]), np.sin(x[0]), np.cos(x[1]), np.sin(x[1])])
        self._state[pose.tobytes()] = np.asarray(x, float)
        return PhysicalObservation(pose)

    def seed(self):
        return self._obs(np.zeros(2))

    def restore(self, observation):
        return self._state[np.asarray(observation.pose, float).reshape(-1).tobytes()]

    def command(self, session, direction, scale, *, parent=None):
        J = np.array([[-np.sin(session[0]), 0.], [np.cos(session[0]), 0.],
                      [0., -np.sin(session[1])], [0., np.cos(session[1])]])
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        return PhysicalResponse(self._obs(session + scale * (J.T @ d)), True, True, {}, parent)

    def command_many(self, session, directions, scale, *, parent=None):
        return [self.command(session, d, scale, parent=parent) for d in directions]


def test_aimed_measurement_keeps_certified_frames():
    options = AdvancingFrontOptions(step=.5, probe_scale=.05, max_samples=40, max_seconds=120,
                                    aimed_probe_scale=.05, aimed_random_count=2)
    q, _edges, frames, audit = explore(_Torus(), options=options)
    records = [entry for entry in audit if "index" in entry]
    modes = [record["diagnostics"]["probe_mode"] for record in records]
    # the seed is measured generically first; its record holds its latest (aimed) re-measurement
    assert "aimed" in modes
    assert all(record["outcome"] == "certified_regular" and record["rank"] == 2
               for record in records)


def test_aimed_measurement_is_off_by_default():
    options = AdvancingFrontOptions(step=.5, probe_scale=.05, max_samples=12, max_seconds=120)
    _q, _edges, _frames, audit = explore(_Torus(), options=options)
    records = [entry for entry in audit if "index" in entry]
    assert {record["diagnostics"]["probe_mode"] for record in records} == {"generic"}


class _CountingTorus(_Torus):
    """Counts how many frame measurements (h and h/2 batch pairs) each pose receives."""

    def __init__(self):
        super().__init__()
        self.batches = {}

    def command_many(self, session, directions, scale, *, parent=None):
        key = np.asarray(session, float).tobytes()
        self.batches[key] = self.batches.get(key, 0) + 1
        return super().command_many(session, directions, scale, parent=parent)


def test_aimed_mode_does_not_remeasure_certified_nodes():
    system = _CountingTorus()
    options = AdvancingFrontOptions(step=.5, probe_scale=.05, max_samples=30, max_seconds=120,
                                    aimed_probe_scale=.05, aimed_random_count=2)
    _q, _edges, _frames, audit = explore(system, options=options)
    records = [entry for entry in audit if "index" in entry]
    assert all(record["outcome"] == "certified_regular" for record in records)
    # one measurement per pose: two batches (h and h/2), or four after a fallback
    assert max(system.batches.values()) <= 4
    assert sum(count == 2 for count in system.batches.values()) >= len(records) - 1


def test_certification_measures_in_the_supplied_metric():
    from atlas_rom.sample.sample_graph import certified_edges

    class Flat:
        """Every midpoint is on the set except along axis 1, which is off by its length."""
        def residuals(self, poses):
            return [np.array([0.0, abs(p[1]), 0.0]) for p in poses]

    q = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    edges = np.empty((0, 2), int)
    plain, _ = certified_edges(q, edges, Flat(), k=2, audit=0, log=lambda *_: None)
    # stretching axis 1 by 10 makes its chords long, so their small residual passes
    stretch = lambda rows: np.asarray(rows, float) * np.array([1., 10., 1.])
    weighed, _ = certified_edges(q, edges, Flat(), k=2, audit=0, log=lambda *_: None,
                                 metric=stretch)
    plain, weighed = {tuple(e) for e in plain}, {tuple(e) for e in weighed}
    assert (0, 1) in plain and (0, 2) not in plain
    assert (0, 2) in weighed
