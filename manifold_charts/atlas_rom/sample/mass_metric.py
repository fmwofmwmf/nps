"""The kinetic-energy (mass) metric for the advancing front, from each body's mesh.

``factored.KineticMetric`` is the metric itself: a body's pose rows move a point
p by dA [p;1], so the mass-weighted squared displacement is sum_r dA_r Mbar dA_r^T
with Mbar = int rho [p;1][p;1]^T.  This module supplies Mbar from the case's
meshes and densities, divided by the total mass, so euclidean length in the
weighed coordinates is the RMS material displacement over the whole mechanism,
in the case's length units.

The metric depends only on simulator geometry and density.  It does not depend
on the world origin or on where a body's local frame sits, which plain packed
distance does: a rigid rotation about the origin leaves every packed translation
unchanged.  It declares no dimension, coordinate, topology or chart.

``MassWhitenedSystem`` exposes a ``PhysicalSystem`` in the weighed coordinates,
so an explorer that measures with plain euclidean distance measures in this
metric without knowing it.
"""
from __future__ import annotations

import inspect

import numpy as np

from atlas_rom.sample.factored import KineticMetric
from atlas_rom.simulate.oracle_base import PhysicalObservation, PhysicalResponse


def body_moments(spec) -> tuple[np.ndarray, np.ndarray]:
    """(n, 4, 4) second moments Mbar and (n,) masses of the driven bodies, from mesh and density."""
    import trimesh

    from atlas_rom.spec import geometry

    moments, masses = [], []
    for name in spec.mechanism.body_order:
        body = spec.mechanism.body(name)
        V, F = geometry.build_mesh(body.mesh)
        mesh = trimesh.Trimesh(V, F, process=False)
        if not mesh.is_volume:
            raise ValueError(f"{spec.name}: body {name!r} mesh is not a closed volume; "
                             "its mass moments are undefined")
        mesh.density = float(body.density)
        mass = float(mesh.mass)
        centre = np.asarray(mesh.center_mass, float)
        inertia = np.asarray(mesh.moment_inertia, float)
        second = np.trace(inertia) / 2.0 * np.eye(3) - inertia      # central second moment
        moment = np.zeros((4, 4))
        moment[:3, :3] = second + mass * np.outer(centre, centre)
        moment[:3, 3] = moment[3, :3] = mass * centre
        moment[3, 3] = mass
        moments.append(moment)
        masses.append(mass)
    return np.asarray(moments), np.asarray(masses)


def mass_metric(spec) -> tuple[KineticMetric, np.ndarray]:
    """The kinetic metric normalised to RMS displacement, and the body masses."""
    moments, masses = body_moments(spec)
    return KineticMetric(moments / masses.sum()), masses


class MassWhitenedSystem:
    """A ``PhysicalSystem`` seen in mass-weighed coordinates y.

    Commands are taken in y and carried to the packed backend; observations are
    returned in y.  ``generic_directions`` draws probe directions that push every
    body by the same RMS material displacement, so a thin or light body is not
    asked for a large motion along a direction it barely resists.
    """

    def __init__(self, inner, metric: KineticMetric, masses: np.ndarray):
        self.inner = inner
        self.metric = metric
        self.masses = np.asarray(masses, float)
        batched = getattr(getattr(inner, "oracle", None), "directional_responses", None)
        self._batched = (batched if batched is not None
                         and "scales" in inspect.signature(batched).parameters else None)

    def to_whitened(self, q):
        """Packed pose (ambient,) or vectors (ambient, k) to weighed coordinates."""
        return self.metric.weigh(np.asarray(q, float))

    def to_packed(self, y):
        return self.metric.unweigh(np.asarray(y, float))

    def _observation(self, observation):
        pose = np.asarray(observation.pose, float).reshape(-1)
        return PhysicalObservation(self.to_whitened(pose), observation.diagnostics)

    def _response(self, response):
        return PhysicalResponse(self._observation(response.observation), response.converged,
                                response.valid, response.diagnostics, response.parent)

    def _packed_command(self, direction, scale):
        """The packed direction and euclidean scale that move y by scale along direction."""
        unit = np.asarray(direction, float).reshape(-1)
        unit = unit / np.linalg.norm(unit)
        packed = self.to_packed(unit)
        return packed, float(scale) * float(np.linalg.norm(packed))

    def seed(self):
        return self._observation(self.inner.seed())

    def restore(self, observation):
        pose = np.asarray(observation.pose, float).reshape(-1)
        return self.inner.restore(PhysicalObservation(self.to_packed(pose), observation.diagnostics))

    def command(self, session, direction, scale, *, parent=None):
        packed, packed_scale = self._packed_command(direction, scale)
        return self._response(self.inner.command(session, packed, packed_scale, parent=parent))

    def command_many(self, session, directions, scale, *, parent=None):
        commands = [self._packed_command(direction, scale) for direction in directions]
        if self._batched is None:
            return [self._response(self.inner.command(session, packed, size, parent=parent))
                    for packed, size in commands]
        answers = self._batched(session, [packed for packed, _size in commands], None,
                                scales=[size for _packed, size in commands])
        return [self._response(PhysicalResponse(PhysicalObservation(np.asarray(a.result, float),
                                                                    a.diagnostics),
                                                a.converged, a.valid, a.diagnostics, parent))
                for a in answers]

    def generic_directions(self, generator, count):
        """Orthonormal draws in the packed space, each body's part rescaled to the
        same RMS material displacement, returned as unit columns in y."""
        ambient = 12 * len(self.masses)
        draws, _ = np.linalg.qr(generator.standard_normal((ambient, count)), mode="reduced")
        weighed = self.to_whitened(draws)
        fraction = np.sqrt(self.masses / self.masses.sum())
        for i, share in enumerate(fraction):
            block = weighed[12 * i:12 * (i + 1)]
            norms = np.linalg.norm(block, axis=0)
            # equal RMS displacement per body: ||y_b||^2 = (m_b / M) rms_b^2
            block *= np.where(norms > 0.0, share / np.maximum(norms, 1e-300), 0.0)
        return weighed / np.linalg.norm(weighed, axis=0)
