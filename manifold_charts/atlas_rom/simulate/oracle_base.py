"""Solver-neutral directional-response interface for physical chart sampling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class PhysicalObservation:
    """A simulator-produced physical pose exposed to exploration."""

    pose: np.ndarray
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class PhysicalResponse:
    """One settled physical response to an opaque command."""

    observation: PhysicalObservation
    converged: bool
    valid: bool
    diagnostics: dict[str, Any]
    parent: int | None = None


class PhysicalSystem(Protocol):
    """Response-only backend consumed by the physical advancing-front strategy.

    Sessions are intentionally opaque.  Both observations and command
    directions live in the observation space; any simulator-specific state
    differential stays inside the backend.
    """

    def seed(self) -> PhysicalObservation: ...
    def restore(self, observation: PhysicalObservation) -> Any: ...
    def command(self, session: Any, direction: np.ndarray, scale: float,
                *, parent: int | None = None) -> PhysicalResponse: ...
    def command_many(self, session: Any, directions: Sequence[np.ndarray], scale: float,
                     *, parent: int | None = None) -> list[PhysicalResponse]: ...


class LegacyPhysicalOracleAdapter:
    """Expose an existing ``ManifoldOracle`` through ``PhysicalSystem``.

    The adapter is deliberately mechanical: all solver state and command
    interpretation remain in the backend, while the strategy sees only poses,
    measured frames, and settled responses.
    """

    def __init__(self, oracle: "ManifoldOracle", *, observe: Callable[[np.ndarray], np.ndarray] | None = None):
        self.oracle = oracle
        self._observe = observe or (lambda value: np.asarray(value, float))

    def _observation(self, value, diagnostics=None):
        return PhysicalObservation(np.asarray(self._observe(np.asarray(value, float)), float), diagnostics)

    def seed(self):
        return self._observation(self.oracle.seed())

    def restore(self, observation):
        _settled, token = self.oracle.baseline(np.asarray(observation.pose, float))
        return token

    def command(self, session, direction, scale, *, parent=None):
        response = self.oracle.directional_response(session, np.asarray(direction, float), scale, parent=parent)
        observation = self._observation(response.result, response.diagnostics)
        return PhysicalResponse(observation, response.converged, response.valid,
                                response.diagnostics, parent)

    def command_many(self, session, directions, scale, *, parent=None):
        return [self.command(session, direction, scale, parent=parent) for direction in directions]


@dataclass(frozen=True)
class DirectionalResponse:
    """One settled observation made from a restored physical baseline."""

    baseline: np.ndarray
    direction: np.ndarray
    scale: float
    result: np.ndarray
    converged: bool
    valid: bool
    diagnostics: dict[str, Any]
    sequence: int
    parent: int | None = None
    reused: bool = False

    @property
    def displacement(self) -> np.ndarray:
        return self.result - self.baseline


class ProjectionLedger:
    """Bounded response-reuse cache with an optional append-only JSONL audit.

    The bound is memory, not a fixed count: the explorer re-measures a pose's
    frame as neighbours arrive, and a 4096-entry cap evicted the probes those
    re-measurements repeat (measured: 48% of twin-piston solves were repeats).
    ``max_entries`` still caps the count when given.
    """

    _ENTRY_OVERHEAD_BYTES = 1024          # key string, dict and array headers

    def __init__(self, path: str | Path | None = None, *, max_entries: int | None = None,
                 max_bytes: int = 256 * 2 ** 20):
        self.path = None if path is None else Path(path)
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("ProjectionLedger max_bytes must be positive")
        if max_entries is not None and int(max_entries) <= 0:
            raise ValueError("ProjectionLedger max_entries must be positive")
        self._entry_cap = None if max_entries is None else int(max_entries)
        self.max_entries = None                 # resolved from the first entry's size
        self._by_key: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.hits = self.misses = self.evictions = 0
        if self.path is not None and self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self._cache(json.loads(line))

    @staticmethod
    def _key(fingerprint, baseline, direction, scale):
        payload = np.concatenate((np.asarray(baseline, float).ravel(),
                                  np.asarray(direction, float).ravel(),
                                  [float(scale)])).astype("<f8", copy=False).tobytes()
        return f"{fingerprint}:{hashlib.sha256(payload).hexdigest()}"

    def _cache(self, record):
        key = record["key"]
        self._by_key[key] = {"result": np.asarray(record["result"], float),
                             "converged": bool(record["converged"]),
                             "valid": bool(record["valid"]),
                             "diagnostics": dict(record.get("diagnostics", {}))}
        self._by_key.move_to_end(key)
        if self.max_entries is None:
            entry_bytes = self._by_key[key]["result"].nbytes + self._ENTRY_OVERHEAD_BYTES
            self.max_entries = max(1, self.max_bytes // entry_bytes)
            if self._entry_cap is not None:
                self.max_entries = min(self.max_entries, self._entry_cap)
        while len(self._by_key) > self.max_entries:
            self._by_key.popitem(last=False)
            self.evictions += 1

    def lookup(self, fingerprint, baseline, direction, scale):
        key = self._key(fingerprint, baseline, direction, scale)
        record = self._by_key.get(key)
        if record is None:
            self.misses += 1
            return None
        self.hits += 1
        self._by_key.move_to_end(key)
        return record

    @property
    def statistics(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "entries": len(self._by_key), "capacity": self.max_entries,
                "hit_rate": self.hits / total if total else 0.0}

    def store(self, fingerprint, baseline, direction, scale, result, converged, valid,
              diagnostics, *, sequence=None, parent=None):
        """Cache a minimal response and optionally emit its complete audit row."""
        key = self._key(fingerprint, baseline, direction, scale)
        cached = {"key": key, "result": np.asarray(result, float),
                  "converged": bool(converged), "valid": bool(valid),
                  "diagnostics": dict(diagnostics)}
        self._cache(cached)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                audit = {"key": key, "baseline": np.asarray(baseline, float).tolist(),
                         "command": np.asarray(direction, float).tolist(),
                         "direction": np.asarray(direction, float).tolist(), "scale": float(scale),
                         "result": np.asarray(result, float).tolist(),
                         "settled_result": np.asarray(result, float).tolist(),
                         "converged": bool(converged), "valid": bool(valid),
                         "diagnostics": diagnostics, "oracle_fingerprint": fingerprint,
                         "sequence": sequence, "parent": parent, "later_use": {"reused": False}}
                fh.write(json.dumps(audit, sort_keys=True) + "\n")


class ManifoldOracle:
    """A sampler-facing oracle: restore, push in one direction, report outcome.

    Backends only need to implement :meth:`project`.  Stateful backends should
    override snapshot/restore; the default token is the settled packed state.
    """

    dim_ambient: int = 0

    def __init__(self, *, ledger: ProjectionLedger | None = None):
        self.ledger = ledger or ProjectionLedger()
        self._response_sequence = 0

    def project(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def response_validity(self, result: np.ndarray) -> tuple[bool, dict[str, Any]]:
        """Whether a settled response remains in this system's physical domain.

        The default has no inequalities beyond a finite projected pose.  Backends
        that model travel or workspace limits override this without exposing that
        information to the response-only exploration strategy.
        """
        return True, {}

    def oracle_fingerprint(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}:v1"

    def snapshot(self, settled: np.ndarray):
        """Capture enough state to make subsequent responses comparable."""
        return np.asarray(settled, float).copy()

    def restore(self, snapshot):
        """Restore exactly the baseline captured by :meth:`snapshot`."""
        return np.asarray(self.project(np.asarray(snapshot, float)), float)

    def baseline(self, q: np.ndarray):
        settled = np.asarray(self.project(q), float)
        return settled, self.snapshot(settled)

    def directional_response(self, snapshot, direction, scale, *, parent=None,
                             reuse=True) -> DirectionalResponse:
        """Restore, command one displacement, settle, and restore again.

        The returned displacement is an observed secant, not a finite
        difference.  It is therefore meaningful for arbitrary response
        backends and remains valid when their response is nonlinear.
        """
        # Third-party/test backends written before this interface commonly do
        # not call ``super().__init__``; remain source compatible.
        if not hasattr(self, "ledger"):
            self.ledger = ProjectionLedger()
        if not hasattr(self, "_response_sequence"):
            self._response_sequence = 0
        # Key reuse to the *captured* baseline, never to a baseline that was
        # re-settled during restore.  A stateful solver can move by round-off
        # on that settle; using that value made logically identical requests
        # miss the cache.
        baseline = (np.asarray(snapshot["packed"], float).copy()
                    if isinstance(snapshot, dict) and "packed" in snapshot
                    else np.asarray(snapshot, float).copy())
        direction = np.asarray(direction, float).reshape(-1)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("directional response requires a finite nonzero direction")
        direction = direction / norm
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("directional response scale must be positive")
        fingerprint = self.oracle_fingerprint()
        cached = self.ledger.lookup(fingerprint, baseline, direction, scale) if reuse else None
        self._response_sequence += 1
        sequence = self._response_sequence
        # Restore once before each physical command.  Do not restore again on
        # return: the next command restores its own baseline, and tangent
        # measurement performs one final restore when it is finished.
        self.restore(snapshot)
        if cached is not None:
            result = np.asarray(cached["result"], float)
            response = DirectionalResponse(baseline, direction, scale, result,
                                           bool(cached["converged"]), bool(cached["valid"]),
                                           dict(cached.get("diagnostics", {})), sequence,
                                           parent, reused=True)
            return response
        try:
            result = np.asarray(self.project(baseline + scale * direction), float)
            valid = bool(result.shape == baseline.shape and np.isfinite(result).all())
            converged = valid
            diagnostics = {"response_norm": float(np.linalg.norm(result - baseline))}
            if valid:
                domain_valid, domain_diagnostics = self.response_validity(result)
                valid = bool(domain_valid)
                diagnostics.update(domain_diagnostics)
        except Exception as error:  # preserve failed physical measurements in the ledger
            result = baseline.copy()
            valid = converged = False
            diagnostics = {"error": f"{type(error).__name__}: {error}"}
        self.ledger.store(fingerprint, baseline, direction, scale, result, converged, valid,
                          diagnostics, sequence=sequence, parent=parent)
        return DirectionalResponse(baseline, direction, scale, result, converged,
                                   valid, diagnostics, sequence, parent)

    def tangent_measurement(self, p, k=None, eps=1e-3, n_probe=None,
                            rel_tol=0.05, rng=None, scales=(1.0,),
                            witnessed_secants: Iterable[np.ndarray] | None = None,
                            proposal_directions: Iterable[np.ndarray] | None = None):
        """Measure a local tangent using paired, differential responses.

        A one-sided cloud of finite response secants spans both tangent and
        curvature directions.  For each probe direction this instead measures
        ``(P(q+h v)-P(q-h v))/(2h)`` at independent batches and at ``h`` and
        ``h/2``.  The paired difference cancels the leading even curvature
        term.  Rank is the largest principal subspace that agrees across those
        independent measurements; no declared dimension or fraction of the
        strongest singular value selects it.
        """
        n = self.dim_ambient
        scales = tuple(sorted({float(s) for s in scales if float(s) > 0.0}))
        if not scales:
            raise ValueError("tangent probe scales must contain a positive value")
        baseline, snapshot = self.baseline(p)
        if rng is None:
            digest = hashlib.blake2b(np.asarray(baseline, float).tobytes(), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
        h = float(eps) * scales[0]
        if h <= 0.0:
            raise ValueError("tangent measurement radius must be positive")

        # Parent frames and walked secants are *physical observations*, not
        # declared coordinates.  They are proposals only: paired responses
        # still decide whether they are valid tangent evidence at this pose.
        proposals = []
        for value in tuple(proposal_directions or ()) + tuple(witnessed_secants or ()):
            direction = np.asarray(value, float).reshape(-1)
            norm = float(np.linalg.norm(direction))
            if direction.size == n and np.isfinite(norm) and norm > 1e-12:
                proposals.append(direction / norm)

        def draw(count):
            values = list(proposals[:count])
            if len(values) < count:
                random = rng.standard_normal((count - len(values), n))
                values.extend(random / np.maximum(np.linalg.norm(random, axis=1, keepdims=True), 1e-300))
            return np.asarray(values, float)

        def responses(directions, radius):
            commands = np.vstack((directions, -directions))
            if hasattr(self, "directional_responses"):
                return self.directional_responses(snapshot, commands, radius)
            return [self.directional_response(snapshot, direction, radius) for direction in commands]

        def observe(directions, radius):
            paired = responses(directions, radius)
            rows = []
            for direction, plus, minus in zip(directions, paired[:len(directions)], paired[len(directions):]):
                if not (plus.valid and minus.valid) or plus.result.shape != baseline.shape or minus.result.shape != baseline.shape:
                    continue
                odd = (plus.result - minus.result) / (2.0 * radius)
                even = (plus.result + minus.result - 2.0 * baseline) / (radius * radius)
                signal = float(np.linalg.norm(odd))
                noise = max(.5 * radius * float(np.linalg.norm(even)),
                            np.finfo(float).eps * max(signal, 1.0))
                rows.append((direction, odd, even, signal / noise))
            return rows

        def rank_basis(rows):
            accepted = [row for row in rows if row[3] >= 4.0 and np.linalg.norm(row[1]) > 1e-12]
            if len(accepted) < 2:
                return None, None, accepted
            # Rank is estimated from unit derivatives.  A large response can
            # no longer erase a weaker but repeatable independent response.
            normalized = np.asarray([row[1] / np.linalg.norm(row[1]) for row in accepted])
            _u, singular, vt = np.linalg.svd(normalized, full_matrices=False)
            floor = 1e-8 * max(float(singular[0]), 1.0)
            rank = int(np.count_nonzero(singular > floor))
            if rank > 1:
                gaps = singular[:-1] / np.maximum(singular[1:], floor)
                rank = min(rank, int(np.argmax(gaps)) + 1)
            return vt.T[:, :rank], singular, accepted

        def paired_rows(directions):
            """Pool one direction's h and h/2 evidence into one derivative."""
            high = observe(directions, h)
            low = observe(directions, h / 2.0)
            pooled = []
            for high_row, low_row in zip(high, low):
                # Directions are submitted in identical order.  Both scales
                # must pass their dimensional SNR check before they can supply
                # a Richardson derivative or a locality estimate.
                if not np.allclose(high_row[0], low_row[0]):
                    continue
                derivative = (4.0 * low_row[1] - high_row[1]) / 3.0
                disagreement = float(np.linalg.norm(low_row[1] - high_row[1]))
                signal = float(np.linalg.norm(derivative))
                if signal <= 1e-12 or signal / max(disagreement, 1e-12 * max(signal, 1.0)) < 2.0:
                    continue
                pooled.append((high_row[0], derivative, high_row[2], min(high_row[3], low_row[3]),
                               disagreement / max(signal, 1e-300)))
            return pooled, high, low

        def select_confirmation(rows, basis):
            if basis is None:
                return np.asarray([row[0] for row in rows[:4]], float)
            selected, span = [], np.empty((n, 0))
            for row in sorted(rows, key=lambda item: item[3], reverse=True):
                candidate = row[1] / max(np.linalg.norm(row[1]), 1e-300)
                residual = candidate - span @ (span.T @ candidate) if span.size else candidate
                if np.linalg.norm(residual) > .15:
                    selected.append(row[0]); span = np.column_stack((span, candidate / np.linalg.norm(candidate)))
                if len(selected) >= basis.shape[1]:
                    break
            selected.extend(row[0] for row in rows if 2.0 <= row[3] < 8.0)
            return np.asarray(selected[:max(4, basis.shape[1])], float)

        def cosine(a, b, dim):
            if a is None or b is None or dim <= 0 or a.shape[1] < dim or b.shape[1] < dim:
                return 0.0
            return float(np.linalg.svd(a[:, :dim].T @ b[:, :dim], compute_uv=False).min(initial=1.0))

        # Candidate discovery is one pooled set, not competing per-batch rank
        # votes.  Each accepted direction costs the paired +/- h and +/- h/2
        # responses needed to make it comparable.
        pooled, high_rows, low_rows = paired_rows(draw(4))
        basis, singular, accepted = rank_basis(pooled)
        for _ in range(3):
            if basis is not None and len(accepted) >= basis.shape[1] + 2:
                break
            more, high, low = paired_rows(draw(2))
            pooled.extend(more); high_rows.extend(high); low_rows.extend(low)
            basis, singular, accepted = rank_basis(pooled)

        # Holdout commands are new in command space, modulo sign: +/- v is
        # the same paired experiment.  Whether one adds a dimension is judged
        # solely from its *measured response* and its h-vs-h/2 error.
        seen_commands = [row[0] for row in pooled]
        holdout, holdout_high, holdout_low = [], [], []
        no_new = 0
        current = list(pooled)
        current_basis = basis
        attempts = 0
        while no_new < 2 and len(holdout) < 4 and attempts < 24:
            attempts += 1
            command = rng.standard_normal(n)
            command /= max(float(np.linalg.norm(command)), 1e-300)
            if any(abs(float(command @ old)) > .98 for old in seen_commands):
                continue
            seen_commands.append(command)
            rows, high, low = paired_rows(np.asarray([command]))
            holdout_high.extend(high); holdout_low.extend(low)
            if not rows:
                continue
            row = rows[0]
            holdout.append(row)
            normalized = row[1] / max(float(np.linalg.norm(row[1])), 1e-300)
            residual = (float(np.linalg.norm(normalized - current_basis @ (current_basis.T @ normalized)))
                        if current_basis is not None and current_basis.size else 1.0)
            error_tolerance = max(2.0 * float(row[4]), .02)
            if residual > error_tolerance:
                expanded_basis, _expanded_singular, _expanded_rows = rank_basis(current + [row])
                if expanded_basis is None or current_basis is None or expanded_basis.shape[1] <= current_basis.shape[1]:
                    no_new = 0
                else:
                    current.append(row); current_basis = expanded_basis; no_new = 0
            else:
                no_new += 1

        combined_basis, combined_singular, combined_accepted = rank_basis(current)
        dim = 0 if basis is None else basis.shape[1]
        agreement = cosine(basis, combined_basis, dim)
        regular = (dim > 0 and len(accepted) >= dim + 2 and no_new >= 2
                   and combined_basis is not None and combined_basis.shape[1] == dim
                   and agreement >= .98)
        outcome = "certified_regular" if regular else "uncertain"
        chosen = combined_basis if regular else basis
        singular = combined_singular if combined_singular is not None else (singular if singular is not None else np.empty(0))
        kept = accepted + holdout
        odd_norms = np.asarray([np.linalg.norm(row[1]) for row in kept], float)
        even_norms = np.asarray([np.linalg.norm(row[2]) for row in kept], float)
        diagnostics = {
            "probe_scale_disagreement": float(1.0 - agreement),
            "probe_signed_asymmetry": float(np.median(even_norms / np.maximum(odd_norms, 1e-300))) if len(kept) else np.inf,
            "probe_settle_repeatability": 0.0,
            "probe_confidence": agreement if regular else 0.0,
            "probe_retained_fraction": float(len(kept)) / max(float(len(high_rows) + len(low_rows) + len(holdout_high) + len(holdout_low)), 1.0),
            "probe_observation_count": len(kept),
            "probe_direction_count": len(high_rows) + len(low_rows) + len(holdout_high) + len(holdout_low),
            "probe_witness_count": len(proposals),
            "probe_candidate_rank": dim,
            "probe_validated_rank": 0 if combined_basis is None else combined_basis.shape[1],
            "probe_holdout_no_new": no_new,
            "probe_subspace_agreement": agreement,
            "probe_outcome": outcome,
            "probe_method": "adaptive_paired_differential_response",
        }
        self.restore(snapshot)
        if not regular:
            return np.empty((n, 0)), singular, 0, diagnostics
        return chosen[:, :dim], singular, dim, diagnostics

    def tangent_frame(self, p, k=None, eps=1e-3, n_probe=None,
                      rel_tol=0.05, rng=None):
        frame, sigma, dim, _diag = self.tangent_measurement(
            p, k=k, eps=eps, n_probe=n_probe, rel_tol=rel_tol, rng=rng)
        return frame, sigma, dim


class CallbackDirectionalOracle(ManifoldOracle):
    """General fast-backend seam for contact or any other response engine.

    A backend can submit a batch of independent restored directional commands
    (for example to a worker pool or accelerator) and return settled poses.
    There is no mechanism, contact-mode, or topology knowledge here.  Passing
    no batch callback leaves the normal ``project`` fallback authoritative.
    """

    def __init__(self, dim_ambient: int, project: Callable[[np.ndarray], np.ndarray], *,
                 snapshot: Callable[[np.ndarray], Any] | None = None,
                 restore: Callable[[Any], np.ndarray] | None = None,
                 batch_response: Callable[[Any, np.ndarray, np.ndarray], Iterable[np.ndarray]] | None = None,
                 fingerprint: str = "callback-directional-v1", ledger=None):
        super().__init__(ledger=ledger)
        self.dim_ambient = int(dim_ambient)
        self._project = project; self._snapshot = snapshot; self._restore = restore
        self._batch_response = batch_response; self._fingerprint = fingerprint

    def project(self, q): return np.asarray(self._project(np.asarray(q, float)), float)
    def snapshot(self, settled): return self._snapshot(settled) if self._snapshot else super().snapshot(settled)
    def restore(self, token): return np.asarray(self._restore(token), float) if self._restore else super().restore(token)
    def oracle_fingerprint(self): return self._fingerprint

    def directional_responses(self, snapshot, directions, scale):
        """Optional batched implementation that still writes normal ledger rows."""
        directions = np.asarray(directions, float)
        if self._batch_response is None:
            return [self.directional_response(snapshot, d, scale) for d in directions]
        baseline = (np.asarray(snapshot["packed"], float).copy()
                    if isinstance(snapshot, dict) and "packed" in snapshot
                    else np.asarray(snapshot, float).copy())
        unit = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-300)
        fingerprint = self.oracle_fingerprint()
        cached = [self.ledger.lookup(fingerprint, baseline, d, scale) for d in unit]
        missing = [i for i, row in enumerate(cached) if row is None]
        landed = {}
        if missing:
            self.restore(snapshot)
            landed = dict(zip(missing, self._batch_response(snapshot, unit[missing],
                                                              np.full(len(missing), scale))))
        out = []
        for i, direction in enumerate(unit):
            self._response_sequence += 1
            row = cached[i]
            reused = row is not None
            result = np.asarray(row["result"] if row is not None else landed[i], float)
            valid = bool(result.shape == baseline.shape and np.isfinite(result).all())
            diagnostics = {"response_norm": float(np.linalg.norm(result - baseline)),
                           "batched": True}
            if row is None:
                self.ledger.store(fingerprint, baseline, direction, scale, result, valid, valid,
                                  diagnostics, sequence=self._response_sequence)
            out.append(DirectionalResponse(baseline, direction, float(scale), result, valid,
                                           valid, diagnostics, self._response_sequence,
                                           reused=reused))
        return out
