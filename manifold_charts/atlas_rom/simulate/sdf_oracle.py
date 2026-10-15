"""Pure SDF directional oracle and bounded explorer.

This module intentionally knows only vectors and a signed-distance function.
Negative values are valid; no mechanism, mesh, topology, or simulator state is
part of a sampling decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from atlas_rom.simulate.oracle_base import DirectionalResponse, ManifoldOracle, PhysicalObservation, PhysicalResponse, ProjectionLedger


@dataclass(frozen=True)
class SDFSystem:
    sdf: Callable[[np.ndarray], float]
    seeds: tuple[tuple[float, ...], ...]
    dimension: int
    fingerprint: str = "sdf-system-v1"
    tolerance: float = 1e-8
    lipschitz: float = 1.0
    batched_sdf: Callable[[np.ndarray], np.ndarray] | None = None
    observe: Callable[[np.ndarray], np.ndarray] | None = None
    state_from_observation: Callable[[np.ndarray], np.ndarray] | None = None
    surface: bool = False
    # Optional physical admissibility predicate for an open subset of a zero
    # level surface.  It belongs to the oracle only: learned chart code sees
    # accepted frames, never this membership function.
    surface_accept: Callable[[np.ndarray], bool] | None = None
    step: float = .16
    # Local tangent-estimation scale.  When omitted, preserve the historical
    # coupling to ``step``; surface cases with tight curvature can override it
    # without making advancing-front coverage arbitrarily fine.
    tangent_eps: float | None = None
    max_samples: int | None = 320
    max_seconds: float | None = 5.0


class SDFDirectionalOracle(ManifoldOracle):
    """Segment-traced responses over a region defined solely by ``phi <= 0``."""

    def __init__(self, system: SDFSystem, *, ledger: ProjectionLedger | None = None):
        super().__init__(ledger=ledger)
        self.system = system
        self.dim_ambient = int(system.dimension)
        self.query_count = 0
        self.batch_query_count = 0

    def oracle_fingerprint(self): return self.system.fingerprint

    def tangent_measurement(self, p, k=None, eps=1e-3, n_probe=None,
                            rel_tol=0.05, rng=None, scales=(1.0,),
                            witnessed_secants=None, proposal_directions=None):
        """Measure a zero-level-surface tangent from local SDF probes.

        Random paired surface responses are unreliable when a proposed command
        has a normal component: retraction then has no signed branch to follow.
        A centred finite-difference gradient is direct local physical evidence
        from the SDF, and its null space supplies only the measured tangent
        directions.  No case coordinate, topology, or chart information enters
        this calculation.
        """
        if not self.system.surface:
            return super().tangent_measurement(
                p, k=k, eps=eps, n_probe=n_probe, rel_tol=rel_tol, rng=rng,
                scales=scales, witnessed_secants=witnessed_secants,
                proposal_directions=proposal_directions)
        point = np.asarray(p, float).reshape(-1)
        h = max(1e-7, min(float(eps), .01 * float(self.system.step)))
        gradient = np.empty(len(point), float)
        for axis in range(len(point)):
            offset = np.zeros(len(point)); offset[axis] = h
            gradient[axis] = (self._phi(point + offset) - self._phi(point - offset)) / (2.0 * h)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm <= 1e-10:
            return np.empty((len(point), 0)), np.empty(0), 0, {
                "probe_outcome": "uncertain", "probe_method": "surface_sdf_gradient",
                "gradient_norm": norm,
            }
        normal = gradient / norm
        _left, _singular, right = np.linalg.svd(normal.reshape(1, -1), full_matrices=True)
        basis = right[1:].T
        return basis, np.ones(basis.shape[1]), int(basis.shape[1]), {
            "probe_outcome": "certified_regular", "probe_method": "surface_sdf_gradient",
            "gradient_norm": norm,
        }

    def _phi(self, q):
        self.query_count += 1
        return float(self.system.sdf(np.asarray(q, float)))

    def _surface_allowed(self, q) -> bool:
        accept = self.system.surface_accept
        return accept is None or bool(accept(np.asarray(q, float)))

    def _phis(self, q):
        q = np.asarray(q, float)
        if self.system.batched_sdf is None:
            return np.asarray([self._phi(row) for row in q])
        self.batch_query_count += 1
        return np.asarray(self.system.batched_sdf(q), float)

    def project(self, q):
        q = np.asarray(q, float)
        if self.system.surface:
            return self._surface_project(q)
        if self._phi(q) > self.system.tolerance:
            raise ValueError("SDF oracle cannot project an invalid point; use a directional response")
        return q.copy()

    def _surface_project(self, q, *, center=None, trust_radius: float | None = None):
        """Retract onto ``phi == 0`` using only local SDF evaluations.

        Open surfaces have valid zero-level edges where an unsigned distance
        field is non-smooth.  A raw Newton update can oscillate there, so every
        update is accepted only if it reduces field distance; a bounded local
        descent stencil supplies physical field evidence when its gradient is
        not reliable.
        """
        point = np.asarray(q, float).copy()
        center = point.copy() if center is None else np.asarray(center, float).copy()
        if trust_radius is not None and trust_radius <= 0.0:
            raise ValueError("surface projection trust radius must be positive")

        def in_trust_region(candidate):
            return trust_radius is None or np.linalg.norm(candidate - center) <= trust_radius
        base_h = max(1e-6, .01 * float(self.system.step))
        for _ in range(48):
            value = self._phi(point)
            if abs(value) <= self.system.tolerance:
                if self._surface_allowed(point):
                    return point
                raise ValueError("surface projection lies outside the physical surface subset")
            # Once a descent step reaches a sheet edge closely, a fixed probe
            # wider than the remaining distance straddles unrelated nearest
            # features.  Shrink the finite-difference radius with the residual
            # while retaining a floating-point floor.
            h = min(base_h, max(1e-8, .25 * abs(value)))
            gradient = np.empty(len(point))
            for axis in range(len(point)):
                offset = np.zeros(len(point)); offset[axis] = h
                gradient[axis] = (self._phi(point + offset) - self._phi(point - offset)) / (2 * h)
            norm2 = float(gradient @ gradient)
            accepted = None
            if norm2 > 1e-20:
                newton = -value * gradient / norm2
                for fraction in (1.0, .5, .25, .125, .0625, .03125):
                    candidate = point + fraction * newton
                    if in_trust_region(candidate) and abs(self._phi(candidate)) < abs(value):
                        accepted = candidate
                        break
            if accepted is None:
                # The nearest feature can switch at a sheet edge.  Search a
                # small ambient stencil for a descent witness rather than
                # treating that non-smoothness as an absent physical surface.
                radius = max(h, min(abs(value), .5 * float(self.system.step)))
                axes = np.eye(len(point))
                candidates = []
                for fraction in (1.0, .5, .25, .125):
                    candidates.extend(point + fraction * radius * direction
                                      for direction in np.vstack((axes, -axes)))
                values = [abs(self._phi(candidate)) if in_trust_region(candidate) else np.inf
                          for candidate in candidates]
                best = int(np.argmin(values))
                if values[best] < abs(value):
                    accepted = candidates[best]
            if accepted is None:
                raise RuntimeError("surface SDF retraction found no local distance descent")
            point = accepted
        if abs(self._phi(point)) > 1e-6:
            raise RuntimeError("surface SDF retraction did not converge")
        if not self._surface_allowed(point):
            raise ValueError("surface projection lies outside the physical surface subset")
        return point

    def _surface_response(self, baseline, direction, scale):
        """Trace a continuous zero-set branch using local trust-region retractions.

        A nearest zero-set projection has no branch identity: on tightly folded
        surfaces it can jump to another nearby sheet.  Here every correction is
        restricted around a predictor, and ambiguous long moves are retried as
        progressively smaller continuous substeps.  The choice is based only on
        physical displacement and SDF residuals, never on case topology.
        """
        for subdivisions in (1, 2, 4, 8, 16):
            current = baseline.copy()
            substep = scale / subdivisions
            residuals, radii = [], []
            try:
                for _ in range(subdivisions):
                    prediction = current + substep * direction
                    residual = abs(float(self._phi(prediction)))
                    # A generic physical command can have a substantial normal
                    # component.  The retraction neighbourhood must contain
                    # the zero set at the predictor; shrinking both the normal
                    # error and the trust radius with substep cannot achieve
                    # that.  The residual is physical SDF evidence, not a
                    # declared tangent or topology prior.
                    trust_radius = max(.25 * substep, 1.25 * residual)
                    result = self._surface_project(prediction, center=prediction,
                                                   trust_radius=trust_radius)
                    residuals.append(residual); radii.append(trust_radius)
                    secant = result - current
                    # A correction that moves back across the requested local
                    # direction is not a continuous response to this command.
                    if float(secant @ direction) <= 1e-10 * max(substep, 1.0):
                        raise RuntimeError("surface branch correction reversed local progress")
                    current = result
            except (RuntimeError, ValueError):
                continue
            return current, {"surface_retraction": True, "subdivisions": subdivisions,
                             "substep": substep, "trust_radius": radii[-1],
                             "prediction_residual": residuals[-1],
                             "max_prediction_residual": max(residuals),
                             "max_trust_radius": max(radii)}
        raise RuntimeError("surface response remained branch-ambiguous after bounded bisection")

    def directional_response(self, snapshot, direction, scale, *, parent=None, reuse=True):
        baseline = np.asarray(snapshot, float).copy()
        direction = np.asarray(direction, float).reshape(-1)
        direction /= max(float(np.linalg.norm(direction)), 1e-300)
        scale = float(scale); fingerprint = self.oracle_fingerprint()
        cached = self.ledger.lookup(fingerprint, baseline, direction, scale) if reuse else None
        self._response_sequence += 1
        if cached is not None:
            return DirectionalResponse(baseline, direction, scale, np.asarray(cached["result"], float),
                                       bool(cached["converged"]), bool(cached["valid"]),
                                       dict(cached["diagnostics"]), self._response_sequence, parent, True)
        phi0 = self._phi(baseline)
        if self.system.surface:
            try:
                result, diagnostics = self._surface_response(baseline, direction, scale)
                valid = converged = True
                diagnostics.update({"start_phi": phi0, "end_phi": self._phi(result),
                                    "hit_boundary": False})
            except Exception as error:
                result = baseline; valid = converged = False
                diagnostics = {"start_phi": phi0, "error": f"{type(error).__name__}: {error}"}
        elif phi0 > self.system.tolerance:
            result, valid, converged = baseline, False, False
            diagnostics = {"start_phi": phi0, "reason": "invalid_baseline", "sdf_queries": 1}
        else:
            result, diagnostics = self._trace(baseline, direction, scale, phi0)
            valid = True; converged = True
        self.ledger.store(fingerprint, baseline, direction, scale, result, converged, valid,
                          diagnostics, sequence=self._response_sequence, parent=parent)
        return DirectionalResponse(baseline, direction, scale, result, converged, valid,
                                   diagnostics, self._response_sequence, parent)

    def _trace(self, base, direction, scale, phi0):
        t, last_t, last_phi, steps = 0.0, 0.0, phi0, 0
        target = 1.0
        while t < target:
            # True-SDF conservative advance: cannot cross a zero while its
            # distance bound says we remain inside.  A safety factor handles
            # finite arithmetic and non-ideal caller SDFs.
            advance = max(abs(last_phi) / max(self.system.lipschitz, 1e-300) / scale * .8, 1e-6)
            t = min(target, last_t + advance)
            phi = self._phi(base + t * scale * direction); steps += 1
            if phi > self.system.tolerance:
                lo, hi = last_t, t
                for _ in range(48):
                    mid = .5 * (lo + hi)
                    if self._phi(base + mid * scale * direction) <= 0.0: lo = mid
                    else: hi = mid
                return base + lo * scale * direction, {"start_phi": phi0, "end_phi": phi,
                    "hit_boundary": True, "first_zero": lo, "segment_fraction": lo,
                    "sdf_queries": steps + 48}
            last_t, last_phi = t, phi
        return base + scale * direction, {"start_phi": phi0, "end_phi": last_phi,
            "hit_boundary": False, "first_zero": None, "segment_fraction": 1.0,
            "sdf_queries": steps}

    def directional_responses(self, snapshot, directions, scale):
        # The response policy is identical in scalar and batch modes; batched
        # SDF providers merely reduce function-call overhead, not semantics.
        return [self.directional_response(snapshot, d, scale) for d in np.asarray(directions)]


class SDFPhysicalOracle:
    """Physical-oracle facade for SDF tracing used by the shared strategy."""

    def __init__(self, system: SDFSystem, *, ledger=None, rng=None):
        self.system = system
        self.backend = SDFDirectionalOracle(system, ledger=ledger)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self._observations: list[PhysicalObservation] = []
        self._state_by_pose: dict[bytes, np.ndarray] = {}

    @property
    def query_count(self):
        return self.backend.query_count

    def project(self, state):
        return self.backend.project(state)

    def directional_response(self, snapshot, direction, scale, **kwargs):
        return self.backend.directional_response(snapshot, direction, scale, **kwargs)

    def _observe(self, state):
        if self.system.observe is None:
            return np.asarray(state, float)
        return np.asarray(self.system.observe(np.asarray(state, float)), float)

    def _remember(self, state):
        pose = self._observe(state)
        self._state_by_pose[np.asarray(pose, float).reshape(-1).tobytes()] = np.asarray(state, float).copy()
        return pose

    def seed(self):
        state = np.asarray(self.system.seeds[0], float)
        return PhysicalObservation(self._remember(state), {"state": state})

    def restore(self, observation):
        state = self._state_by_pose.get(np.asarray(observation.pose, float).reshape(-1).tobytes())
        if state is None:
            raise ValueError("SDF session can only be restored from a simulator-produced observation")
        _base, token = self.backend.baseline(state)
        return token

    def _state_direction(self, state, direction):
        """Keep the observation-to-state differential private to the backend."""
        direction = np.asarray(direction, float).reshape(-1)
        if self.system.observe is None:
            return direction
        epsilon = 1e-6
        origin = self._observe(state).reshape(-1)
        columns = [(self._observe(state + epsilon * np.eye(state.size)[i]).reshape(-1) - origin) / epsilon
                   for i in range(state.size)]
        jacobian = np.column_stack(columns)
        return np.linalg.pinv(jacobian, rcond=1e-10) @ direction

    def command(self, session, direction, step, *, parent=None):
        state_direction = self._state_direction(np.asarray(session, float), direction)
        response = self.backend.directional_response(session, state_direction, step, parent=parent)
        pose = self._remember(response.result)
        state_observation = PhysicalObservation(pose, {"state": np.asarray(response.result, float)})
        self._observations.append(state_observation)
        return PhysicalResponse(state_observation, response.converged, response.valid,
                                response.diagnostics, parent)

    def command_many(self, session, directions, step, *, parent=None):
        return [self.command(session, direction, step, parent=parent) for direction in directions]


def explore(system: SDFSystem, *, step=.1, max_samples=256, max_seconds=5., ledger_path=None,
            rng=None, return_frames=False, direction_schedule=None):
    """Run the shared advancing front with SDF tracing as the backend."""
    from atlas_rom.sample.physical_advancing_front import AdvancingFrontOptions, explore as front_explore
    oracle = SDFPhysicalOracle(system, ledger=ProjectionLedger(ledger_path), rng=rng)
    observations = [PhysicalObservation(oracle._remember(seed), {"state": np.asarray(seed, float)})
                    for seed in system.seeds]
    q, edges, _frames, records = front_explore(
        oracle,
        options=AdvancingFrontOptions(step=step, max_samples=max_samples, max_seconds=max_seconds, seed=0,
                                      probe_scale=system.tangent_eps),
        seeds=observations)
    states = np.asarray([oracle._state_by_pose.get(np.asarray(pose, float).reshape(-1).tobytes(), pose)
                         for pose in q], float)
    if not return_frames:
        # Preserve the established low-level API: points are SDF states and the
        # backend exposes query accounting and directional-response methods.
        return states, edges, oracle.backend
    measured = np.asarray(_frames, float)
    if not len(measured):
        raise RuntimeError("SDF exploration retained no tangent evidence")
    ranks = {frame.shape[1] for frame in measured}
    if len(ranks) != 1:
        raise RuntimeError("SDF exploration observed nonuniform local rank; split the dataset before training")
    return states, edges, oracle.backend, measured
